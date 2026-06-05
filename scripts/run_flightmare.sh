#!/usr/bin/env bash

cd ~/fpv_ws || exit 1

export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/ros/noetic/bin:$PATH
unset PYTHONPATH
unset PYTHONHOME

source /opt/ros/noetic/setup.bash
source ~/fpv_ws/devel/setup.bash

roslaunch flightros rotors_gazebo.launch
