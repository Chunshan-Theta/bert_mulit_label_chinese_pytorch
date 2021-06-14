# --pretrained_model_path models/chinese_wwm_ext_pytorch/pytorch_model.bin \
# --pretrained_model_path models/trained_model.bin \
python3 inference/run_multi_classifier_infer.py --load_model_path models/trained_model.bin \
                                          --vocab_path models/chinese_wwm_ext_pytorch/vocab.txt \
                                          --test_path test.tsv \
                                          --prediction_path predict.tsv \
                                          --batch_size 1 \
                                          --labels_num 5 \
                                          --seq_length 128 \
                                          --output_logits --output_prob \
                                          --threshold 0.5,0.5,0.5,0.5,0.5 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible