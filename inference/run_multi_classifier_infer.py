"""
  This script provides an exmaple to wrap UER-py for classification inference.
"""
import math
import sys
import os
import torch
import argparse
import collections
import torch.nn as nn


uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)
from tool.f1 import __convert_to_validation__,__get_f1__
from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from uer.opts import infer_opts
from finetune.run_classifier_multi import Classifier


def batch_loader(batch_size, src, tgt, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, tgt_batch, seg_batch




def read_dataset(args, path):
    dataset, columns = [], {}
    lines = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            tgt = [0] * args.labels_num
            # print(line[columns["text_a"]])
            lines.append(line[columns["text_a"]])
            for l in line[columns["label"]].split(","):
                tgt[int(l)] = 1
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return lines, dataset



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--labels_num", type=int, required=True,
                        help="Number of prediction labels.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")
    parser.add_argument("--threshold", default="0.5", help="set special threshold for each class")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    model = Classifier(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    lines, dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)

    model.eval()
    TP, FN, FP,perplexity = 0, 0, 0, 0
    thresholds = args.threshold.split(",")
    assert len(thresholds) == args.labels_num
    print(f"thresholds: {thresholds}")
    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        f.write(f"答案\t模型預測\t內文")
        f.write("\n")
        for i, (src_batch, tgt_batch, seg_batch) in enumerate(batch_loader(batch_size, src, tgt, seg)):
            print(f"row: {(i + 1) * batch_size}", end="\r")

            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            with torch.no_grad():
                loss, logits = model(src_batch, tgt_batch, seg_batch, None)

            loss = loss.item() * tgt_batch.view(-1).shape[0]
            perplexity += math.exp(loss)
            # print(f"{math.exp(loss)}")


            losser = nn.Sigmoid()#nn.Softmax(dim=1)
            prob = losser(logits)
            pred = []
            anss = [ass.tolist() for ass in tgt_batch]

            for answer_label in losser(logits):
                pred.append([1 if i > float(thresholds[idx]) else 0 for idx, i in enumerate(answer_label)])

            t_TP, t_FN, t_FP = __convert_to_validation__(anss, pred)
            TP += t_TP
            FN += t_FN
            FP += t_FP

            for j in range(len(pred)):
                f.write(f"{anss[j]}\t{pred[j]}\t{lines[(i*batch_size)+j]}")
                if args.output_logits:
                    f.write("\t" + " ".join([str(float(v)) for v in prob[j]]))
                f.write("\n")

    print(f"perplexity: {perplexity/instances_num}")

    f1, recall, precision = __get_f1__(TP, FN, FP)
    print(f"f1: {f1}\trecall: {recall}\tprecision: {precision}\t")



if __name__ == "__main__":
    main()
