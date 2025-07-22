#!/bin/bash

# This script runs a Docker container for Nerfstudio with COLMAP 3.11 support.
# Ensure XAUTH is set
export XAUTH=${XAUTH:-$HOME/.Xauthority}

if [ -z "$SUDO_USER" ]
then
      user=$USER
else
      user=$SUDO_USER
fi

echo "Running container for user: $user"

# Allow container to use the host X11 server
xhost +local:docker

# Run Docker
# Ensures access to x11, display, just change your paths
docker run --gpus all \
	   --name=${user}_nerf \
	   --env="QT_X11_NO_MITSHM=1" \
	   --env="DISPLAY=$DISPLAY" \
           --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
 	   --env="XAUTHORITY=$XAUTH" \
  	   --volume="$XAUTH:$XAUTH" \
	   -v /shared/datasets:/workspace/datasets \
	   -v /home/$user/.cache/:/home/user/.cache/ \
 	   --env="NVIDIA_VISIBLE_DEVICES=all" \
	   --env="NVIDIA_DRIVER_CAPABILITIES=all" \
	   --privileged \
	   --network=host \
           -it \
           --shm-size=64gb \
           nerf_colmap311 \
	   bash
