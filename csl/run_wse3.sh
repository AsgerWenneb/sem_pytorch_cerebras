#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=17,12 \
--fabric-offsets=4,1 --params=N_kernel:5,M_kernel:5,N_per_PE:50,max_nz_per_n:100 -o out --memcpy --channels 1
cs_python run.py --name out
