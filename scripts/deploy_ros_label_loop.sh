#!/usr/bin/env bash

set -euo pipefail

ACTION="install"
SERVICE_NAME="opentau-ros-label-loop"
REPO_DIR="${OPENTAU_REPO_DIR:-$(pwd)}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TASK=""
MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"
DEVICE="cuda"
IMAGE_TOPIC="/camera1/image_compressed"
INSTRUCTION_TOPIC="/opentau/pi05_instruction"
STOP_TOPIC="/opentau/pi05_stop"
CYCLE_PERIOD_S="1.0"
CAPTURE_TIMEOUT_S="5.0"
RAW_IMAGE="0"
PUBLISH_JSON="0"
MAX_CYCLES=""

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_ros_label_loop.sh [options]

Options:
  --action install|start|restart|stop|status
  --service-name NAME
  --repo-dir PATH
  --ros-setup PATH
  --python-bin PATH
  --task TEXT
  --model-name NAME
  --device cuda|cpu
  --image-topic TOPIC
  --instruction-topic TOPIC
  --stop-topic TOPIC
  --cycle-period-s SECONDS
  --capture-timeout-s SECONDS
  --raw-image
  --publish-json
  --max-cycles N
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action) ACTION="$2"; shift 2 ;;
    --service-name) SERVICE_NAME="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --ros-setup) ROS_SETUP="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --image-topic) IMAGE_TOPIC="$2"; shift 2 ;;
    --instruction-topic) INSTRUCTION_TOPIC="$2"; shift 2 ;;
    --stop-topic) STOP_TOPIC="$2"; shift 2 ;;
    --cycle-period-s) CYCLE_PERIOD_S="$2"; shift 2 ;;
    --capture-timeout-s) CAPTURE_TIMEOUT_S="$2"; shift 2 ;;
    --raw-image) RAW_IMAGE="1"; shift ;;
    --publish-json) PUBLISH_JSON="1"; shift ;;
    --max-cycles) MAX_CYCLES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! "$ACTION" =~ ^(install|start|restart|stop|status)$ ]]; then
  echo "Invalid --action: $ACTION" >&2
  exit 2
fi

if [[ "$ACTION" =~ ^(install|start|restart)$ && -z "$TASK" ]]; then
  echo "--task is required for action '$ACTION'." >&2
  exit 2
fi

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS setup file not found: $ROS_SETUP" >&2
  exit 2
fi

if [[ ! -d "$REPO_DIR/src/opentau" ]]; then
  echo "OpenTau source tree not found under: $REPO_DIR" >&2
  exit 2
fi

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="${SUDO:-sudo}"
fi

quote_args() {
  printf "%q " "$@"
}

build_exec_command() {
  local repo_abs
  repo_abs="$(cd "$REPO_DIR" && pwd)"

  local args=(
    --task "$TASK"
    --model-name "$MODEL_NAME"
    --device "$DEVICE"
    --image-topic "$IMAGE_TOPIC"
    --instruction-topic "$INSTRUCTION_TOPIC"
    --stop-topic "$STOP_TOPIC"
    --cycle-period-s "$CYCLE_PERIOD_S"
    --capture-timeout-s "$CAPTURE_TIMEOUT_S"
  )

  if [[ "$RAW_IMAGE" == "1" ]]; then
    args+=(--raw-image)
  fi
  if [[ "$PUBLISH_JSON" == "1" ]]; then
    args+=(--publish-json)
  fi
  if [[ -n "$MAX_CYCLES" ]]; then
    args+=(--max-cycles "$MAX_CYCLES")
  fi

  printf "source %q && cd %q && export PYTHONPATH=%q:\${PYTHONPATH:-} && exec %q -m opentau.scripts.ros_label_action_loop %s" \
    "$ROS_SETUP" \
    "$repo_abs" \
    "$repo_abs/src" \
    "$PYTHON_BIN" \
    "$(quote_args "${args[@]}")"
}

install_service() {
  local exec_command
  exec_command="$(build_exec_command)"
  local service_file="/tmp/${SERVICE_NAME}.service"

  cat > "$service_file" <<EOF
[Unit]
Description=OpenTau ROS1 VLM label action loop
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$(cd "$REPO_DIR" && pwd)
Environment=PYTHONUNBUFFERED=1
ExecStart=/bin/bash -lc '$exec_command'
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

  $SUDO mv "$service_file" "/etc/systemd/system/${SERVICE_NAME}.service"
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable "${SERVICE_NAME}.service"
  echo "Installed systemd service: ${SERVICE_NAME}.service"
}

case "$ACTION" in
  install)
    install_service
    ;;
  start)
    install_service
    $SUDO systemctl start "${SERVICE_NAME}.service"
    ;;
  restart)
    install_service
    $SUDO systemctl restart "${SERVICE_NAME}.service"
    ;;
  stop)
    $SUDO systemctl stop "${SERVICE_NAME}.service"
    ;;
  status)
    $SUDO systemctl status "${SERVICE_NAME}.service" --no-pager
    ;;
esac
