#!/bin/bash
# Launch the drone_agent nodes (brain_node [4-drone GNN+Heuristic swarm] + yolo_detector + ros_gz_bridge + map_gui_node).
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env
if [ -f "$HOME/Desktop/swarm/.env" ]; then
  set -a
  . "$HOME/Desktop/swarm/.env"
  set +a
fi

export DISPLAY="${DISPLAY:-:0}"
export GALLIUM_DRIVER=d3d12
# WSL: Mesa's d3d12 driver picks adapter 0 (the Intel iGPU on Optimus laptops), which left the RTX unused
# and Gazebo at ~0.1% real time. Select the NVIDIA adapter by name (override by exporting the variable).
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
export MESA_D3D12_DEFAULT_INDEX_BUFFER_STRIDE_2_BYTE=1
export PATH="$HOME/.local/bin:$PATH"
export LLM_BACKEND="${LLM_BACKEND:-ollama}"
export LLM_MODEL="${LLM_MODEL:-qwen2.5:3b}"

source /opt/ros/jazzy/setup.bash
source "$DIR/install/setup.bash"

# Add swarm root so `train.model` (GraphSAGE GNN) is importable while keeping system/.local torch intact for YOLOv8
export PYTHONPATH="/home/tushar/Desktop/swarm:${PYTHONPATH}"

echo "LLM backend: $LLM_BACKEND ($LLM_MODEL)"
exec ros2 launch drone_agent drone_agent.launch.py "$@"
