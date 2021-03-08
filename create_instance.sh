#!/bin/sh

export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="europe-west4-a"
export INSTANCE_NAME="pytorch-instance"
export INSTANCE_TYPE="n1-highmem-8"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-v100,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --metadata="install-nvidia-driver=True"