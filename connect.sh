#!/bin/sh

export ZONE="europe-west4-a"
export INSTANCE_NAME="pytorch-instance"

gcloud compute instances start $INSTANCE_NAME --zone=$ZONE
sleep 10
gcloud compute ssh --zone=$ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080
gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE