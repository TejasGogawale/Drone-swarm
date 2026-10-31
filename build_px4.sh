#!/usr/bin/env bash
set -e

export PATH="$HOME/.local/bin:$PATH"
source /opt/ros/jazzy/setup.bash

# Export CMAKE_PREFIX_PATH with all Gazebo vendor packages from ROS Jazzy
GZ_PATHS=$(find /opt/ros/jazzy/opt -name extra_cmake | tr '\n' ':')
export CMAKE_PREFIX_PATH="${GZ_PATHS}/opt/ros/jazzy:${CMAKE_PREFIX_PATH}"

echo "Configuring PX4 SITL with Gazebo Harmonic support..."
cd /home/tejas/PX4-Autopilot

# Run make px4_sitl_default
DONT_RUN=1 make px4_sitl_default
