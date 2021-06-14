"""
This script provides an exmaple to wrap UER-py for classification.
"""
import math
import sys
import os
import random
import argparse
import torch
import torch.nn as nn


uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)
from tool.f1 import __convert_to_validation__, __get_f1__
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size x labels_num]
            seg: [batch_size x seq_length]
        """


        # Embedding.
        emb = self.embedding(src, seg)
        # print(f"emb: {len(emb)},{emb}")
        # print(f"emb - 0: {len(emb[0])},{emb[0]}")
        # Encoder.
        output = self.encoder(emb, seg)

        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        # print(f"output: {len(output)},{output}")
        # print(f"output - 0: {len(output[0])},{output[0]}")
        logits = self.output_layer_2(output)
        # print(f"logits: {logits}")
        output = nn.Sigmoid()(logits)
        # print(f"output: {output}")
        # print(f"tgt: {tgt}")

        label_ans = tgt.float()
        loss = nn.BCELoss()(output, label_ans)




        # fake_ = torch.tensor([list(i).index(1) for i in label_ans])
        # print(math.exp(nn.CrossEntropyLoss()(nn.Softmax()(logits),fake_)*5))
        return loss, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            labels = line[columns["label"]].split(",")
            # print(f"labels:{labels}")
            for label in labels:
                labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = [0]*args.labels_num
            if line[columns["label"]] != "-1":
                for l in line[columns["label"]].split(","):
                    tgt[int(l)] = 1

            #
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

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)

    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, max_num=-1):
    if max_num == -1:
        src = torch.LongTensor([sample[0] for sample in dataset])
        tgt = torch.LongTensor([sample[1] for sample in dataset])
        seg = torch.LongTensor([sample[2] for sample in dataset])
    else:
        src = torch.LongTensor([sample[0] for sample in dataset][:max_num])
        tgt = torch.LongTensor([sample[1] for sample in dataset][:max_num])
        seg = torch.LongTensor([sample[2] for sample in dataset][:max_num])

    batch_size = args.batch_size

    args.model.eval()
    TP, FP, FN = 0, 0, 0
    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        # pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        # print(f"logits: {logits}")
        # print(f"tgt_batch: {tgt_batch}")
        losser = nn.Sigmoid() #nn.Softmax(dim=1)
        pred = losser(logits)
        pred = pred.round()

        gold = tgt_batch.float()
        t_TP, t_FN, t_FP = __convert_to_validation__(gold,pred)
        TP += t_TP
        FN += t_FN
        FP += t_FP

    f1, recall, precision = __get_f1__(TP, FN, FP)
    return f1



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    parser.add_argument("--save_checkpoint_steps", type=int, default=100,
                        help="Specific steps to save model checkpoint.")

    parser.add_argument("--saved_model_eva_num", type=int, default=-1,
                        help="Specific steps to save model checkpoint.")
    parser.add_argument("--cuda", type=bool, default=False,
                        help="")
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model = model.to(args.device)

    # Training phase.
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
    else:
        soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result, skip_cnt, n = 0.0, 0.0, 0.0, 0, 0

    print("Start training.")

    with open("train_loss.log", "w") as loffile, open("train_f1.log", "w") as recallloffile:
        recallloffile.write(f"train\tdev\n")
        for epoch in range(1, args.epochs_num + 1):
            model.train()
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                #total_loss += loss.item()
                total_loss += loss.item()*tgt_batch.view(-1).shape[0]
                # l_sum += loss.item() * target.view(-1).shape[0]
                n += tgt_batch.view(-1).shape[0]
                avg_loss = total_loss / args.report_steps

                if (i + 1) % args.report_steps == 0:
                    sys_report = f"Epoch id:{epoch}\tTraining steps:{i + 1}\tAvg loss:{avg_loss}\tperplexity:{math.exp(avg_loss)}"
                    print(sys_report)
                    loffile.write(f"{sys_report}\n")
                    total_loss = 0.0

                if (i + 1) % args.save_checkpoint_steps == 0:

                    result_train = evaluate(args, read_dataset(args, args.dev_path), args.saved_model_eva_num)
                    result_dev = evaluate(args, read_dataset(args, args.train_path), args.saved_model_eva_num)
                    recallloffile.write(f"{result_train}\t{result_dev}\n")
                    if result_train > best_result or skip_cnt > 3:
                        print(f"updated model.... {result_train}:{result_dev}")
                        best_result = result_train
                        save_model(model, args.output_model_path)
                        skip_cnt = 0
                    else:
                        skip_cnt += 1



if __name__ == "__main__":
    main()
