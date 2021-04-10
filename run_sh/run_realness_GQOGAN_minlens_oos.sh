#! /bin/bash

lengths="-1"
# dataset_file="binary_undersample, binary_wiki_aug"
# $1 dataset_file
# $2 num_outcomes
# $3 G_updates
# $4 save_path of my computer

for len in ${lengths} ; do
  bash run_sh/run_oodp_realness_gan_length_oos.sh ${1} ${2} ${3} oos-${1}-realness_GQOGAN_outcome${2}_Gupdate${3} ${4}
  mv oodp-gan oos-${1}-realness_GQOGAN_outcome${2}_Gupdate${3}
  mv oos-${1}-realness_GQOGAN_outcome${2}_Gupdate${3}_gross_result.csv oos-${1}-realness_GQOGAN_outcome${2}_Gupdate${3}
  cp -r "/content/Gan-s-Exploration/oos-${1}-realness_GQOGAN_outcome${2}_Gupdate${3}" "${4}"
done
exit 0
