# --pretrained_model_path models/chinese_wwm_ext_pytorch/pytorch_model.bin \
# --pretrained_model_path models/trained_model.bin \
python3 finetune/run_classifier_multi.py --pretrained_model_path models/chinese_wwm_ext_pytorch/pytorch_model.bin \
                                   --vocab_path models/chinese_wwm_ext_pytorch/vocab.txt \
                                   --train_path train.tsv \
                                   --dev_path test.tsv \
                                   --test_path test.tsv \
                                   --epochs_num 1 \
                                   --batch_size 2 \
                                   --report_steps 1 \
                                   --embedding word_pos_seg \
                                   --encoder transformer \
                                   --mask fully_visible \
                                   --output_model_path models/trained_model.bin \
                                   --save_checkpoint_steps 25 \
                                   --saved_model_eva_num 100
