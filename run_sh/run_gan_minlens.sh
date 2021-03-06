#! /bin/bash

lengths="-1"
betas="-1" # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 beta
# $3 mode
# $4 maxlen
# $5 minlen
# $6 optim_mode
# $7 length_weight
# $8 sample_weight
# $9 save_path of my computer
for len in ${lengths} ; do
  for beta in ${betas} ; do
    bash run_sh/run_gan_length.sh ${1} ${beta} ${3} ${4} ${len} ${6} ${7} ${8} GQOGAN-beta${beta}_maxlen${4}_minlen${len}_mode${3}_optim_mode${6}_ls${7}_sw${8}
    mv oodp-gan GQOGAN-beta${beta}_maxlen${4}_minlen${len}_mode${3}_optim_mode${6}_ls${7}_sw${8}
    mv GQOGAN-beta${beta}_maxlen${4}_minlen${len}_mode${3}_optim_mode${6}_ls${7}_sw${8}_gross_result.csv GQOGAN-beta${beta}_maxlen${4}_minlen${len}_mode${3}_optim_mode${6}_ls${7}_sw${8}
    cp -r "/content/Gan-s-Exploration/GQOGAN-beta${beta}_maxlen${4}_minlen${len}_mode${3}_optim_mode${6}_ls${7}_sw${8}" "$9"
  done
done
exit 0
