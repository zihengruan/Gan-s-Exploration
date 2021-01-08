#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192"
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 mode
# $3 gross_result name
##--do_vis \
for seed in ${seeds} ; do
  python -m app.run_oodp_realnessgan_length \
  --model=realness_gan  \
  --seed=${seed}  \
  --D_lr=2e-4 \
  --G_lr=2e-4 \
  --beta1=0.5 \
  --beta2=0.999 \
  --G_h_size=32 \
  --D_h_size=32 \
  --bert_lr=2e-5 \
  --positive_skew=1.0 \
  --negative_skew=-1.0 \
  --num_outcomes=5 \
  --relativisticG \
  --fine_tune \
  --n_epoch=500 \
  --patience=10 \
  --train_batch_size=32 \
  --bert_type=bert-base-chinese \
  --dataset=smp \
  --data_file=$1 \
  --output_dir=oodp-gan/oodp-realness_gan-smp_mode$2_s${seed} \
  --do_train \
  --do_eval \
  --do_test \
  --feature_dim=768 \
  --G_z_dim=1024  \
  --mode=$2  \
  --result=$3
  rm -rf oodp-gan/oodp-realness_gan-smp_mode$2_s${seed}/save
done
exit 0
