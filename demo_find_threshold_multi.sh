python3 inference/run_multi_classifier_find_threshold.py --load_model_path models/trained_model.bin \
                                          --vocab_path models/chinese_wwm_ext_pytorch/vocab.txt \
                                          --train_path train.tsv \
                                          --output_file_path best_threshold.tsv \
                                          --batch_size 4 \
                                          --labels_num 5 \
                                          --seq_length 128 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible

