#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=8,3 \
--fabric-offsets=4,1 --params=N_per_PE:100,NZ_per_N:10 -o out --memcpy --channels 1
cs_python run.py --name out
