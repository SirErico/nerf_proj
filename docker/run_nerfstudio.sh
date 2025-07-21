#!/bin/bash

# Ensure XAUTH is set
export XAUTH=${XAUTH:-$HOME/.Xauthority}

if [ -z "$SUDO_USER" ]
then
      user=$USER
else
      user=$SUDO_USER
fi
# Allow container to use the host X11 server
xhost +local:root

# Run Docker
docker run --gpus all \
	   --name=eryk_nerf \
	   --env="QT_X11_NO_MITSHM=1" \
	   --env DISPLAY=$DISPLAY \
           --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
 	   --env="XAUTHORITY=$XAUTH" \
  	   --volume="$XAUTH:$XAUTH" \
	   -v /shared/datasets:/workspace/datasets \
	   -v /home/erykv/nerf_comparsions:/workspace/nerf_comparsions \
	   -v /home/erykv/.cache/:/home/user/.cache/ \
 	   --env="NVIDIA_VISIBLE_DEVICES=all" \
	   --env="NVIDIA_DRIVER_CAPABILITIES=all" \
	   --privileged \
	   --network=host \
           -it \
           --shm-size=64gb \
           nerf_colmap311 \
	   bash
