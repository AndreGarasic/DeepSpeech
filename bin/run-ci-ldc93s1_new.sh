#!/bin/sh

set -xe


epoch_count=$1
audio_sample_rate=$2


# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py --noshow_progressbar --noearly_stop \
  --train_files data/test1/test_excel.csv --train_batch_size 1 \
  --dev_files data/test1/test_excel.csv --dev_batch_size 1 \
  --test_files data/test1/test_excel.csv --test_batch_size 1 \
  --n_hidden 100 --epochs $epoch_count \
  --max_to_keep 1 --checkpoint_dir 'DeepSpeech\data\test1' \
  --learning_rate 0.001 --dropout_rate 0.05  --export_dir 'DeepSpeech\data\test1' \
  --audio_sample_rate ${audio_sample_rate}
