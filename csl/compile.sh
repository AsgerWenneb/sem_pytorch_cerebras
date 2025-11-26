#!/usr/bin/env bash

set -e

cslc --arch=wse3 ./layout.csl --fabric-dims=9,3 \
--fabric-offsets=4,1 --params=N_per_PE:5,NZ_per_N:2 -o out --memcpy --channels 1
