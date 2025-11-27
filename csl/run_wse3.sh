#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=9,4 \
--fabric-offsets=4,1 --params=N_kernel:2,M_kernel:2,N_per_PE:2,max_nz_per_n:2 -o out --memcpy --channels 1
cs_python run.py --name out
