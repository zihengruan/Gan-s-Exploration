#! /bin/bash

seeds="16 256 1024 2048 8192"
## dataset_file="binary_undersample, binary_wiki_aug"
## $1 dataset_file
## $2 num_outcomes
## $3 G_updates
## $4 gross_result name
## $5 save_path
for seed in ${seeds} ; do
  python -m app.run_oodp_realness1gan_length \
  --model=realness1gan  \
  --seed=${seed}  \
  --D_lr=2e-5 \
  --G_lr=2e-5 \
  --num_outcomes=${2} \
  --D_updates=1 \
  --G_updates=${3} \
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
  --bert_type=bert-large-uncased \
  --dataset=oos-eval \
  --data_file=${1} \
  --output_dir=oodp-gan/oos-${1}-oodp-gan_outcome${2}_Gupdate${3}_s${seed} \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --feature_dim=1024 \
  --G_z_dim=1024  \
  --result=${4}
  rm -rf oodp-gan/oos-${1}-oodp-gan_outcome${2}_Gupdate${3}_s${seed}/save

  cp oos-${1}-realness_GQOGAN_outcome${2}_Gupdate${3}_gross_result.csv oodp-gan/oos-${1}-oodp-gan_outcome${2}_Gupdate${3}_s${seed}
  cp -r "/content/Gan-s-Exploration/oodp-gan/oos-${1}-oodp-gan_outcome${2}_Gupdate${3}_s${seed} " "${5}"
done
exit 0
