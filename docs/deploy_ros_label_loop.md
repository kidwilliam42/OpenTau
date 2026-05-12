# Deploy the ROS1 label action loop

This deployment path is for a humanoid robot terminal server running ROS1
(`rospy`). It deploys the OpenTau source tree from GitHub Actions over SSH and
installs a systemd service that runs:

```bash
PYTHONPATH=/opt/opentau/src python3 -m opentau.scripts.ros_label_action_loop ...
```

## Required GitHub secrets

Configure these secrets in the GitHub repository or in the `robot-server`
environment:

| Secret | Required | Example | Description |
| --- | --- | --- | --- |
| `ROBOT_SERVER_HOST` | yes | `robot-gpu.company.net` | SSH host/IP for the robot terminal server. |
| `ROBOT_SERVER_USER` | yes | `robot` | SSH user. |
| `ROBOT_SERVER_SSH_KEY` | yes | private key | SSH private key with access to the server. |
| `ROBOT_SERVER_PORT` | no | `22` | SSH port. Defaults to `22`. |
| `ROBOT_DEPLOY_PATH` | no | `/opt/opentau` | Remote checkout/deploy directory. Defaults to `/opt/opentau`. |
| `ROBOT_ROS_SETUP` | no | `/opt/ros/noetic/setup.bash` | ROS1 setup script. Defaults to Noetic. |
| `ROBOT_PYTHON_BIN` | no | `/home/robot/miniconda/envs/opentau/bin/python` | Python executable with Qwen/PyTorch/ROS deps. Defaults to `python3`. |

The SSH user must be able to install/update the systemd service, usually via
passwordless `sudo` for `systemctl` and writing `/etc/systemd/system`.

## Server prerequisites

On the robot terminal server:

1. ROS1 is installed and `rospy`, `sensor_msgs`, and `std_msgs` are importable
   from the configured Python environment.
2. Qwen3-VL runtime dependencies are installed in `ROBOT_PYTHON_BIN`
   (`torch`, `transformers` with `Qwen3VLForConditionalGeneration`, `Pillow`,
   `numpy`, etc.).
3. The robot publishes a camera topic:
   - compressed: `sensor_msgs/CompressedImage`, e.g. `/camera1/image_compressed`
   - raw: `sensor_msgs/Image`, e.g. `/camera1/image_raw`
4. The robot terminal server subscribes to instruction/stop topics:
   - `std_msgs/String` instruction topic, default `/opentau/pi05_instruction`
   - `std_msgs/String` stop topic, default `/opentau/pi05_stop`

## GitHub workflow

Run the **Deploy ROS Label Loop** workflow manually.

Actions:

- `install`: copy source and install/update the systemd service, but do not start it.
- `start`: install/update and start the service. Requires `confirm_start=START_ROBOT`.
- `restart`: install/update and restart the service. Requires `confirm_start=START_ROBOT`.
- `stop`: stop the systemd service.
- `status`: show systemd service status.

Example workflow inputs:

```text
action: start
confirm_start: START_ROBOT
task: Pick up the target object and place it into target slot A1.
image_topic: /camera1/image_compressed
raw_image: false
instruction_topic: /opentau/pi05_instruction
stop_topic: /opentau/pi05_stop
publish_json: false
cycle_period_s: 1.0
device: cuda
```

If the terminal server expects JSON payloads, enable `publish_json`. The
instruction payload will be:

```json
{"command": "execute", "instruction": "Navigate to the target slot."}
```

and stop will be:

```json
{"command": "stop", "instruction": "STOP"}
```

## Manual server command

After deployment, you can also manage the service directly on the server:

```bash
cd /opt/opentau
scripts/deploy_ros_label_loop.sh \
  --action restart \
  --task "Pick up the target object and place it into target slot A1." \
  --image-topic /camera1/image_compressed \
  --instruction-topic /opentau/pi05_instruction \
  --stop-topic /opentau/pi05_stop \
  --cycle-period-s 1.0 \
  --device cuda
```

View logs:

```bash
journalctl -u opentau-ros-label-loop -f
```
