#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192"
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 beta
# $3 mode
# $4 maxlen
# $5 minlen
# $6 optim_mode
# $7 length_weight
# $8 sample_weight
# $9 gross_result name
for seed in ${seeds} ; do
  python -m app.run_oodp_gan_length \
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
  --output_dir=oodp-gan/oodp-gan-smp_beta$2_maxlen$4_minlen$5_mode$3_optim_mode$6_lw$7_sw$8_s${seed} \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --feature_dim=768 \
  --G_z_dim=1024  \
  --beta=$2  \
  --mode=$3  \
  --maxlen=$4 \
  --minlen=$5 \
  --optim_mode=$6 \
  --length_weight=$7 \
  --sample_weight=$8 \
  --result=$9
  rm -rf oodp-gan/oodp-gan-smp_beta$2_maxlen$4_minlen$5_mode$3_optim_mode$6_lw$7_sw$8_s${seed}/save
done
exit 0
