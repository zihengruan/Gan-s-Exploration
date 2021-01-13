#! /bin/bash

seeds="16 256 1024 2048 8192"
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 mode
# $3 minlen
# $4 maxlen
# $5 gross_result name
for seed in ${seeds} ; do
  python -m app.run_oodp_realnessgan_length \
  --model=realness_gan  \
  --seed=${seed}  \
  --D_lr=2e-5 \
  --G_lr=2e-5 \
  --num_outcomes=5 \
  --D_updates=1 \
  --G_updates=3 \
  --beta1=0.5 \
  --beta2=0.999 \
  --G_h_size=32 \
  --D_h_size=32 \
  --bert_lr=2e-5 \
  --positive_skew=1.0 \
  --negative_skew=-1.0 \
  --relativisticG \
  --fine_tune \
  --n_epoch=500 \
  --patience=10 \
  --train_batch_size=32 \
  --bert_type=bert-base-chinese \
  --dataset=smp \
  --data_file=$1 \
  --output_dir=oodp-gan/oodp-realness_gan-smp_mode$2_minlen$3_maxlen$4_s${seed} \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --feature_dim=768 \
  --G_z_dim=1024  \
  --mode=$2  \
  --minlen=$3 \
  --maxlen=$4 \
  --result=$5
  rm -rf oodp-gan/oodp-realness_gan-smp_mode$2_minlen$3_maxlen$4_s${seed}/save
done
exit 0
