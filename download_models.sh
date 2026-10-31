#!/usr/bin/env bash
set -e
source /opt/ros/jazzy/setup.bash

MODELS=(
  "https://fuel.gazebosim.org/1.0/openrobotics/models/collapsed%20fire%20station"
  "https://fuel.gazebosim.org/1.0/openrobotics/models/collapsed%20police%20station"
  "https://fuel.gazebosim.org/1.0/openrobotics/models/collapsed%20house"
  "https://fuel.gazebosim.org/1.0/openrobotics/models/ambulance"
  "https://fuel.gazebosim.org/1.0/openrobotics/models/prius%20hybrid"
  "https://fuel.gazebosim.org/1.0/openrobotics/models/suv"
  "https://fuel.gazebosim.org/1.0/openrobotics/models/standing%20person"
  "https://fuel.gazebosim.org/1.0/openrobotics/models/casual%20female"
  "https://fuel.gazebosim.org/1.0/mingfei/models/actor"
)

for m in "${MODELS[@]}"; do
  echo "Downloading $m..."
  gz fuel download -u "$m" || true
done

BASE_OR="$HOME/.gz/fuel/fuel.gazebosim.org/openrobotics/models"
if [ -d "$BASE_OR" ]; then
  cd "$BASE_OR"
  for dir in *%20*; do
    if [ -d "$dir" ]; then
      target="${dir//%20/ }"
      echo "Linking '$dir' -> '$target'"
      ln -sfn "$dir" "$target"
    fi
  done
fi

echo "Models downloaded and linked successfully!"
