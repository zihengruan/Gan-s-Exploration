#! /bin/bash

lengths="-1"
# dataset_file="binary_undersample"
# $1 dataset_file
# $2 num_outcomes
# $3 G_updates
# $4 save_path of my computer


for len in ${lengths} ; do
  bash run_sh/run_oodp_realness_gan_length_oos.sh ${1} ${2} ${3} oos-realness_GQOGAN_outcome${2}_Gupdate${3} ${4}
  mv oodp-gan oos-realness_GQOGAN_outcome${2}_Gupdate${3}
  mv oos-realness_GQOGAN_outcome${2}_Gupdate${3}_gross_result.csv oos-realness_GQOGAN_outcome${2}_Gupdate${3}
  cp -r "/content/Gan-s-Exploration/oos-realness_GQOGAN_outcome${2}_Gupdate${3}" "${4}"
done
exit 0
