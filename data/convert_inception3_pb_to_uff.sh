#!/bin/bash
PB=inception_v3_2016_08_28_frozen.pb
UFF=inception_v3.uff

if [ ! -z $1 ] && [ ! -z $2 ]
then
  python3 /usr/lib/python3.5/dist-packages/uff/bin/convert_to_uff.py tensorflow --input-file $PB -o $UFF -I $1 -O $2
elif [ ! -z $1 ]
then
  python3 /usr/lib/python3.5/dist-packages/uff/bin/convert_to_uff.py tensorflow --input-file $PB -o $UFF -O $1
else
  python3 /usr/lib/python3.5/dist-packages/uff/bin/convert_to_uff.py tensorflow --input-file $PB -l
fi
