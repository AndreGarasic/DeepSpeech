##!/bin/sh
set -xe
#if [ ! -f DeepSpeech.py ]; then
#    echo "Please make sure you run this from DeepSpeech's top level directory."
#    exit 1
#fi;
#
#if [ ! -f "data/test1/izlaz.csv" ]; then
#    echo "Downloading and preprocessing LDC93S1 example data, saving in ./data/ldc93s1."
#    python -u bin/import_ldc93s1.py ./data/test1
#fi
#
#if [ -d "${COMPUTE_KEEP_DIR}" ]; then
#    checkpoint_dir=$COMPUTE_KEEP_DIR
#else
#    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/ldc93s1"))')
#fi
# We should try to minimise forks, especially on Windows where they are
# unreasonably slow, so skip the feature probes when bash or zsh are
# being used:


# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -u DeepSpeech.py --noshow_progressbar \
  --train_files data/audio_files_andre/andre.csv \
  --test_files data/audio_files_andre/andre.csv \
  --train_batch_size 1 \
  --test_batch_size 1 \
  --n_hidden 50 \
  --epochs 100 \
  --export_dir 'DeepSpeech\data\audio_files_andre' \