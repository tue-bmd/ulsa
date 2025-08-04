#!/bin/bash

# Send ulsa image to snellius
TMP_IMAGE_TAR=/tmp/ulsa.tar
TMP_IMAGE_SIF=/tmp/ulsa.sif
SNELLIUS_USER=wvannierop
SNELLIUS_ADDRESS=snellius.surf.nl

# build docker image
docker build . -t ulsa:latest
# save docker image to file.
docker save -o $TMP_IMAGE_TAR ulsa:latest
# convert docker image to apptainer image
apptainer build $TMP_IMAGE_SIF docker-archive://$TMP_IMAGE_TAR
# copy apptainer image to snellius
scp $TMP_IMAGE_SIF $SNELLIUS_USER@$SNELLIUS_ADDRESS:/projects/0/prjs0966/ulsa.sif