#!/bin/bash
# Start the full 4-Drone Decentralized Swarm Simulation Stack in the background:
#   1. MicroXRCEAgent (UDP 8888 bridge between all 4 PX4 instances and ROS 2)
#   2. 4x PX4 SITL instances + Gazebo Harmonic (UAV-1..4 in drone_demo world)
#   3. ROS 2 drone_agent launch (ros_gz_bridge + YOLOv8 + 4-Drone GNN/Heuristic Brain + Live 2D/3D Swarm GUI)

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DISPLAY="${DISPLAY:-:0}"
export PATH="/opt/ros/jazzy/opt/gz_tools_vendor/bin:$HOME/.local/bin:$PATH"
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_INDEX_BUFFER_STRIDE_2_BYTE=1
export MESA_GL_VERSION_OVERRIDE=4.5
export MESA_GLSL_VERSION_OVERRIDE=450
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
export GZ_CONFIG_PATH="/opt/ros/jazzy/opt/gz_sim_vendor/share/gz:/opt/ros/jazzy/opt/sdformat_vendor/share/gz:/opt/ros/jazzy/opt/gz_gui_vendor/share/gz:/opt/ros/jazzy/opt/gz_transport_vendor/share/gz:${GZ_CONFIG_PATH}"

PX4_DIR="$HOME/PX4-Autopilot"
GZ_BRIDGE_DIR="$PX4_DIR/build/px4_sitl_default/src/modules/simulation/gz_bridge"
PX4_BIN="$PX4_DIR/build/px4_sitl_default/bin/px4"

export GZ_SIM_RESOURCE_PATH="$DIR/worlds:$HOME/.gz/fuel/fuel.gazebosim.org/openrobotics/models:$HOME/.gz/fuel/fuel.ignitionrobotics.org/openrobotics/models:$HOME/.gz/models:$PX4_DIR/Tools/simulation/gz/models:$PX4_DIR/Tools/simulation/gz/worlds:${GZ_SIM_RESOURCE_PATH}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$GZ_BRIDGE_DIR:${GZ_SIM_SYSTEM_PLUGIN_PATH}"

# Clean up any stale instances first
pkill -9 -f "map_gui_node" 2>/dev/null || true
pkill -9 -f "brain_node" 2>/dev/null || true
pkill -9 -f "yolo_detector" 2>/dev/null || true
pkill -9 -f "parameter_bridge" 2>/dev/null || true
pkill -9 -f "octomap_server" 2>/dev/null || true
pkill -9 -f "pointcloud_to_laserscan" 2>/dev/null || true
pkill -9 -f "MicroXRCEAgent" 2>/dev/null || true
pkill -9 -f "px4" 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
rm -f /tmp/px4_lock-* /tmp/px4-sock-* 2>/dev/null || true
sleep 2

echo "[1/3] Starting MicroXRCEAgent (UDP 8888 for 4 PX4 Swarm Drones)..."
setsid MicroXRCEAgent udp4 -p 8888 </dev/null >/tmp/xrce_agent.log 2>&1 &

echo "[2/3] Starting 4x PX4 SITL Instances + Gazebo Harmonic (UAV-1..4 in drone_demo world)..."
GZ_BRIDGE_DIR="$HOME/PX4-Autopilot/build/px4_sitl_default/src/modules/simulation/gz_bridge"
PX4_BIN="$HOME/PX4-Autopilot/build/px4_sitl_default/bin/px4"

SWARM_WINGMAN_MODEL="gz_x500_mono_cam"

for i in 0 1 2 3; do
  rm -rf "/tmp/px4_run_$i"
  mkdir -p "/tmp/px4_run_$i"
  ln -sf "$GZ_BRIDGE_DIR"/* "/tmp/px4_run_$i/" 2>/dev/null || true
  rm -rf "/tmp/px4_run_$i/fs"
  mkdir -p "/tmp/px4_run_$i/fs"
done

# Launch UAV-1 (Instance 0, Lead Drone with Camera + 3D LiDAR) at ENU (0, 3, 0.3) -> NED (+3N, 0E)
cd "/tmp/px4_run_0"
setsid env PX4_SIM_MODEL=gz_x500_mono_cam PX4_GZ_WORLD=drone_demo PX4_GZ_MODEL_POSE="0,3,0.3,0,0,0" PX4_GZ_NO_FOLLOW=1 GZ_IP=127.0.0.1 "$PX4_BIN" -i 0 -d </dev/null >/tmp/px4_sitl.log 2>&1 &

echo "Waiting for Gazebo Harmonic world (drone_demo) to initialize before spawning UAV-2, UAV-3, UAV-4..."
sleep 6

# Launch UAV-2 (Instance 1) at ENU (3, 0, 0.3) -> NED (0N, +3E)
cd "/tmp/px4_run_1"
setsid env PX4_SIM_MODEL="$SWARM_WINGMAN_MODEL" PX4_GZ_WORLD=drone_demo PX4_GZ_MODEL_POSE="3,0,0.3,0,0,0" PX4_GZ_NO_FOLLOW=1 GZ_IP=127.0.0.1 "$PX4_BIN" -i 1 -d </dev/null >/tmp/px4_sitl_1.log 2>&1 &
sleep 1

# Launch UAV-3 (Instance 2) at ENU (0, -3, 0.3) -> NED (-3N, 0E)
cd "/tmp/px4_run_2"
setsid env PX4_SIM_MODEL="$SWARM_WINGMAN_MODEL" PX4_GZ_WORLD=drone_demo PX4_GZ_MODEL_POSE="0,-3,0.3,0,0,0" PX4_GZ_NO_FOLLOW=1 GZ_IP=127.0.0.1 "$PX4_BIN" -i 2 -d </dev/null >/tmp/px4_sitl_2.log 2>&1 &
sleep 1

# Launch UAV-4 (Instance 3) at ENU (-3, 0, 0.3) -> NED (0N, -3E)
cd "/tmp/px4_run_3"
setsid env PX4_SIM_MODEL="$SWARM_WINGMAN_MODEL" PX4_GZ_WORLD=drone_demo PX4_GZ_MODEL_POSE="-3,0,0.3,0,0,0" PX4_GZ_NO_FOLLOW=1 GZ_IP=127.0.0.1 "$PX4_BIN" -i 3 -d </dev/null >/tmp/px4_sitl_3.log 2>&1 &
sleep 2

echo "Waiting for PX4 SITL instances to initialize..."
sleep 6

for i in 0 1 2 3; do
  "$PX4_BIN-param" --instance $i set SIM_BAT_DRAIN 0 2>/dev/null || true
  "$PX4_BIN-param" --instance $i set COM_LOW_BAT_ACT 0 2>/dev/null || true
  "$PX4_BIN-param" --instance $i set CBRK_SUPPLY_CHK 894281 2>/dev/null || true
  "$PX4_BIN-param" --instance $i set CBRK_IO_SAFETY 22027 2>/dev/null || true
  "$PX4_BIN-param" --instance $i set COM_ARM_WO_GPS 1 2>/dev/null || true
  "$PX4_BIN-param" --instance $i set COM_PREARM_MODE 0 2>/dev/null || true
  "$PX4_BIN-param" --instance $i set NAV_DLL_ACT 0 2>/dev/null || true
  "$PX4_BIN-param" --instance $i set NAV_RCL_ACT 0 2>/dev/null || true
  "$PX4_BIN-param" --instance $i set COM_RCL_EXCEPT 4 2>/dev/null || true
done

echo "[3/3] Starting 4-Drone Swarm Agent (ROS 2 Decentralized GNN/Heuristic Brain + YOLOv8 + Live 2D/3D Swarm GUI)..."
cd "$DIR"
setsid ./launch_drone.sh </dev/null >/tmp/drone_agent.log 2>&1 &

echo "Waiting for nodes to initialize and stream offboard setpoints..."
for t in $(seq 1 30); do
  if ros2 topic echo /swarm/state --once >/dev/null 2>&1; then
    echo "Swarm state live! (took ${t}s)"
    break
  fi
  sleep 1
done
sleep 2

for i in 0 1 2 3; do
  "$PX4_BIN-commander" --instance $i arm -f 2>/dev/null || true
  "$PX4_BIN-commander" --instance $i mode offboard 2>/dev/null || true
done
ps aux | grep -E "MicroXRCEAgent|px4|gz sim|brain_node|yolo_detector|map_gui_node" | grep -v grep || true
echo ""
echo "=== 4-DRONE DECENTRALIZED SWARM RUNNING! ==="
echo "Logs: /tmp/px4_sitl.log (/tmp/px4_sitl_1..3.log) | /tmp/drone_agent.log | /tmp/xrce_agent.log"
echo "You can control the 4-drone swarm via the Live 2D/3D Swarm Map GUI window,"
echo "or in any terminal via: python3 $DIR/prompt_drone.py"
echo ""
echo "Streaming /tmp/drone_agent.log (press Ctrl+C to stop)..."
trap "pkill -9 -f 'px4|gz sim|MicroXRCEAgent|brain_node|map_gui_node|yolo_detector' 2>/dev/null || true; exit 0" SIGINT SIGTERM
tail -n 20 -f /tmp/drone_agent.log
