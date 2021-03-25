#! /bin/bash

lengths="-1"
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 mode
# $3 num_outcomes
# $4 G_updates
# $5 save_path of my computer
# $6 beta

for len in ${lengths} ; do
  bash run_sh/run_oodp_realness_gan_length.sh ${1} ${2} ${3} ${4} realness_GQOGAN_mode${2}_outcome${3}_Gupdate${4}_beta${6} ${5} ${6}
  mv oodp-gan realness_GQOGAN_mode${2}_outcome${3}_Gupdate${4}_beta${6}
  mv realness_GQOGAN_mode${2}_outcome${3}_Gupdate${4}_beta${6}_gross_result.csv realness_GQOGAN_mode${2}_outcome${3}_Gupdate${4}_beta${6}
  cp -r "/content/Gan-s-Exploration/realness_GQOGAN_mode${2}_outcome${3}_Gupdate${4}_beta${6}" "${5}"
done
exit 0