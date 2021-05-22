#! /bin/bash

seeds="16 256 1024 2048 8192"
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 mode
# $3 num_outcomes
# $4 G_updates
# $5 gross_result name
# $6 save_path
# $7 alpha
for seed in ${seeds} ; do
  python -m app.run_oodp_realness1gan_length \
  --model=realness1gan  \
  --seed=${seed}  \
  --D_lr=2e-5 \
  --G_lr=2e-5 \
  --num_outcomes=${3} \
  --D_updates=1 \
  --G_updates=${4} \
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
  --data_file=${1} \
  --output_dir=oodp-gan/oodp-realness_gan-smp_mode${2}_outcome${3}_Gupdate${4}_alpha${7}_s${seed} \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --feature_dim=768 \
  --G_z_dim=1024  \
  --mode=${2}  \
  --result=${5}  \
  --alpha=${7}  \
  --manual_knowledge  \
  --remove_entity \
  --entity_mode=2

  rm -rf oodp-gan/oodp-realness_gan-smp_mode${2}_outcome${3}_Gupdate${4}_alpha${7}_s${seed}/save

  cp realness_GQOGAN_mode${2}_outcome${3}_Gupdate${4}_alpha${6}_gross_result.csv oodp-gan/oodp-realness_gan-smp_mode${2}_outcome${3}_Gupdate${4}_alpha${7}_s${seed}
  cp -r "/content/Gan-s-Exploration/oodp-gan/oodp-realness_gan-smp_mode${2}_outcome${3}_Gupdate${4}_alpha${7}_s${seed}" "${6}"
done
exit 0
