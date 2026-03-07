# isaac-hand-teleop

**Real-time bimanual hand teleoperation of a Franka Panda robot arm in NVIDIA Isaac Sim**

> Right hand controls XY position on the table. Left hand controls height and gripper. Pick up objects and place them using only a webcam.

---

## Demo

*[Insert split-screen demo GIF here — webcam left, Isaac Sim right]*

---

## What This Is

A real-time teleoperation system where hand movements captured by a standard webcam are mapped to end-effector commands for a Franka Panda robot arm simulated in NVIDIA Isaac Sim 5.1. No depth camera, no VR headset, no special hardware — just a webcam.

The system is designed as a VR-style puppeteering interface: you are the controller, the robot is your digital arm. This architecture is directly applicable to imitation learning data collection, which is the natural next step (Phase 4).

---

## System Architecture

```
[Webcam]
    │
    ▼
[MediaPipe HandLandmarker]
  Two hands detected, classified as Left/Right
  Right hand → wrist XY position (normalized image coords)
  Left hand  → wrist Y position (Z height) + pinch (gripper)
    │
    ▼
[ROS2 Hand Tracking Node]  ←── system Python 3.10
  Publishes: /hand/eef_pose, /hand/gripper_cmd
  Also sends UDP JSON to localhost:5005
    │  (UDP — bypasses Python version conflict)
    ▼
[Isaac Sim Teleop Node]    ←── conda Python 3.11
  UDP listener (background thread)
  Velocity clamping + workspace bounds
  PhysX Jacobian → Differential IK (DLS)
  joint position targets → Franka arm
    │
    ▼
[Isaac Sim 5.1 + Isaac Lab]
  Franka Panda in Simple Room environment
  Table + cup + target zone
  Physics simulation at 100Hz
```

---

## Tech Stack

| Tool | Role |
|------|------|
| NVIDIA Isaac Sim 5.1 | Physics simulation environment |
| Isaac Lab 0.54.3 | Robot framework, differential IK |
| MuJoCo 3.5 | Phase 0 prototyping and IK validation |
| MediaPipe 0.10.21 | Two-hand landmark detection at 30fps |
| OpenCV | Webcam capture and overlay |
| ROS2 Humble | Hand tracking node, topic publishing |
| PyTorch (nightly cu128) | GPU tensor operations for IK |
| Python 3.10 (system) | ROS2 hand tracking process |
| Python 3.11 (conda) | Isaac Sim process |

---

## Control Mapping

```
RIGHT HAND
  Move left/right    → Robot arm moves left/right (Y axis)
  Move up in frame   → Robot arm moves forward on table (X axis)
  Move down in frame → Robot arm moves backward on table (X axis)

LEFT HAND
  Move up in frame   → End effector rises (Z axis)
  Move down in frame → End effector lowers toward table (Z axis)
  Pinch thumb+index  → Gripper closes
  Open fingers       → Gripper opens
```

---

## Project Structure

```
isaac-hand-teleop/
├── mujoco_prototype/
│   ├── franka_kitchen_env.py      # MuJoCo Franka env (Phase 0)
│   ├── hand_teleop_mujoco.py      # Hand tracking → MuJoCo (Phase 0)
│   └── requirements.txt
├── ros2_ws/src/hand_tracking/
│   ├── hand_tracking/
│   │   └── hand_tracking_node.py  # MediaPipe → ROS2 + UDP
│   ├── package.xml
│   └── setup.py
├── isaac_sim/scripts/
│   ├── load_kitchen_scene.py      # Isaac Sim scene (no teleop)
│   ├── isaac_teleop_node.py       # Main teleop: UDP → IK → Franka
│   └── joint_tuner.py             # Interactive joint angle tuner
├── docs/
│   └── challenges.md              # Technical challenges and solutions
└── README.md
```

---

## Setup

### Requirements
- Ubuntu 22.04
- NVIDIA GPU with driver ≥ 580.65 (tested on RTX 5060 Blackwell)
- CUDA 12.8+
- Miniconda

### Environment 1 — Hand Tracking (ROS2, Python 3.10)

```bash
# Install ROS2 Humble (if not already installed)
# https://docs.ros.org/en/humble/Installation.html

# Install MediaPipe and OpenCV on system Python
/usr/bin/python3 -m pip install mediapipe==0.10.21 opencv-python==4.9.0.80

# Build ROS2 package
mkdir -p ~/ros2_ws/src
cp -r ros2_ws/src/hand_tracking ~/ros2_ws/src/
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select hand_tracking --symlink-install
```

### Environment 2 — Isaac Sim (Python 3.11)

```bash
conda create -n isaac_sim python=3.11 -y
conda activate isaac_sim
pip install --upgrade pip
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com

# PyTorch nightly for Blackwell GPU support
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Isaac Lab from source
cd ~/projects
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
pip install -e source/isaaclab
pip install -e source/isaaclab_assets
pip install h5py  # optional, suppresses warning
```

### Phase 0 — MuJoCo Prototype (optional validation)

```bash
conda create -n robot_kitchen python=3.10 -y
conda activate robot_kitchen
pip install -r mujoco_prototype/requirements.txt
cd mujoco_prototype
python hand_teleop_mujoco.py
```

---

## Running

**Terminal 1 — Hand Tracking:**
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run hand_tracking hand_tracking_node
```

**Terminal 2 — Isaac Sim:**
```bash
conda activate isaac_sim
cd ~/projects/isaac-hand-teleop
python isaac_sim/scripts/isaac_teleop_node.py
```

A webcam window opens showing hand detection overlays. Isaac Sim opens with the kitchen scene. Show both hands to the camera to begin teleoperation.

---

## Known Limitations

- **Z depth from image only** — no depth camera used. Left hand height in image controls robot Z, which is intuitive but not metric.
- **Gripper orientation fixed** — end effector always points straight down. Wrist rotation tracking not yet implemented.
- **Lighting sensitive** — MediaPipe hand detection degrades in low light or with fast movement.
- **Single environment** — Isaac Sim runs one robot instance (8GB VRAM constraint).

---

## Key Technical Challenges

See `docs/challenges.md` for detailed writeups on:
- RTX 5060 Blackwell rendering workaround
- Python 3.10/3.11 version conflict solved with UDP bridge
- Isaac Lab `init_state` bug and fix
- PhysX Jacobian access for differential IK
- IK instability and oscillation debugging
- MediaPipe depth limitations and bimanual solution
- One-hand to two-hand control architecture evolution

---

## Hardware Used

- Lenovo Legion 7i, Ubuntu 22.04
- NVIDIA RTX 5060 Laptop GPU (Blackwell, 8GB VRAM)
- Intel Core i9 12th Gen, 32GB RAM
- Standard USB webcam

---

## What's Next

- **Phase 4 — Imitation Learning:** Collect demonstrations via teleoperation and train a Behavior Cloning policy to perform pick and place autonomously
- **Wrist orientation tracking:** Map hand rotation to end effector orientation
- **Depth camera integration:** RealSense D435 for metric Z control
- **Docker setup:** Containerize Isaac Sim + Isaac Lab stack
