#!/usr/bin/env bash

sh generate_soft_label.sh

python run_classifier_soft_label_kwls.py \
  --data_dir=data \
  --task_name=sim \
  --vocab_file=/tf/data/bert/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/tf/data/bert/uncased_L-12_H-768_A-12/bert_config.json \
  --output_dir=data/output \
  --do_train=true \
  --do_eval=false \
  --do_predict=true \
  --init_checkpoint=/tf/data/bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=160 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3 \
  --CUDA_VISIBLE_DEVICES=2 \
  --train_soft_label_file=data/soft_label.npy \
  --do_lower_case=True \
  --do_soft_label=True \
  --alpha=1

python -u print_metrics.py