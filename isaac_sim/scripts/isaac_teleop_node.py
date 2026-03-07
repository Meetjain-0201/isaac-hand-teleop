"""
isaac_teleop_node.py
Receives hand pose via UDP → moves Franka in Isaac Sim.
Uses position-only differential IK (more stable, elbow configuration preserved).
"""

import argparse
import socket
import json
import threading
import time
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Hand Teleop")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv

FRANKA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
)
ROOM_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/Environments/Simple_Room/simple_room.usd"
)
TABLE_H       = 0.8
UDP_PORT      = 5005
ARM_JOINT_IDS = list(range(7))
HOME_POS      = np.array([0.4, 0.0, 1.05])
MAX_DELTA     = 0.02       # max meters per sim step
NO_HAND_TIMEOUT = 2.0


class TeleopState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.target_pos     = HOME_POS.copy()
        self.gripper_norm   = 1.0
        self.last_hand_time = time.time()


state = TeleopState()


def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(1.0)
    print(f"UDP listener on port {UDP_PORT}")
    while True:
        try:
            data, _ = sock.recvfrom(256)
            p = json.loads(data.decode())
            with state.lock:
                state.target_pos     = np.array([p["x"], p["y"], p["z"]])
                state.gripper_norm   = float(p["g"])
                state.last_hand_time = time.time()
        except socket.timeout:
            continue
        except Exception as e:
            print(f"UDP error: {e}")


def build_scene():
    sim_utils.UsdFileCfg(usd_path=ROOM_USD).func(
        "/World/Room", sim_utils.UsdFileCfg(usd_path=ROOM_USD))

    cfg_t = sim_utils.CuboidCfg(
        size=(0.9, 0.6, TABLE_H),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
    )
    cfg_t.func("/World/Table", cfg_t, translation=(0.5, 0.0, TABLE_H / 2.0))

    cup = RigidObject(RigidObjectCfg(
        prim_path="/World/Cup",
        spawn=sim_utils.CylinderCfg(
            radius=0.03, height=0.08,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.1, TABLE_H + 0.04)),
    ))

    cfg_tz = sim_utils.CylinderCfg(
        radius=0.07, height=0.005,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
    )
    cfg_tz.func("/World/TargetZone", cfg_tz,
                translation=(0.45, -0.2, TABLE_H + 0.003))

    franka = Articulation(ArticulationCfg(
        prim_path="/World/Franka",
        spawn=sim_utils.UsdFileCfg(usd_path=FRANKA_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_H),
            joint_pos={
                "panda_joint1":  0.0,  "panda_joint2": -0.569,
                "panda_joint3":  0.0,  "panda_joint4": -2.810,
                "panda_joint5":  0.0,  "panda_joint6":  3.037,
                "panda_joint7":  0.785,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"], stiffness=800, damping=40),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"], stiffness=800, damping=40),
        },
    ))
    return franka, cup


def main():
    threading.Thread(target=udp_listener, daemon=True).start()

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    sim.set_camera_view(eye=[0.2, -1.2, 1.6], target=[0.45, 0.0, 0.85])

    franka, cup = build_scene()
    sim.reset()

    # Position-only IK — more stable, preserves arm configuration
    ik_controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="position",       # position only, not pose
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.1},
        ),
        num_envs=1, device="cuda:0"
    )
    ik_controller.reset()

    hand_idx   = franka.find_bodies("panda_hand")[0][0]
    jacobi_idx = hand_idx - 1

    print("Isaac teleop ready. Waiting for hand data on UDP port 5005...")

    smooth_target = HOME_POS.copy()

    # Skip first step — Jacobians not valid until after first sim.step()
    sim.step()
    franka.update(sim.get_physics_dt())

    step = 0
    while simulation_app.is_running():

        # Edge case: no hand → return to home
        with state.lock:
            raw_target   = state.target_pos.copy()
            gripper_norm = state.gripper_norm
            hand_age     = time.time() - state.last_hand_time

        if hand_age > NO_HAND_TIMEOUT:
            raw_target   = HOME_POS.copy()
            gripper_norm = 1.0

        # Velocity clamp
        delta      = raw_target - smooth_target
        d_norm     = np.linalg.norm(delta)
        if d_norm > MAX_DELTA:
            delta = delta * (MAX_DELTA / d_norm)
        smooth_target = smooth_target + delta

        # Workspace clamp
        smooth_target[0] = np.clip(smooth_target[0], 0.2,  0.7)
        smooth_target[1] = np.clip(smooth_target[1], -0.4, 0.4)
        smooth_target[2] = np.clip(smooth_target[2], TABLE_H + 0.05, 1.4)

        # Jacobian (world → base frame)
        jacobian_w = franka.root_physx_view.get_jacobians()[:, jacobi_idx, :, ARM_JOINT_IDS]
        root_rot   = matrix_from_quat(quat_inv(franka.data.root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(root_rot, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(root_rot, jacobian_b[:, 3:, :])

        # EEF in base frame
        eef_pos_b, eef_quat_b = subtract_frame_transforms(
            franka.data.root_pos_w, franka.data.root_quat_w,
            franka.data.body_pos_w[:, hand_idx],
            franka.data.body_quat_w[:, hand_idx],
        )

        # Target position in base frame
        tgt_w = torch.tensor(smooth_target, dtype=torch.float32, device="cuda:0").unsqueeze(0)
        tgt_b, _ = subtract_frame_transforms(
            franka.data.root_pos_w, franka.data.root_quat_w, tgt_w)

        # IK — position only command (3D)
        ik_controller.set_command(tgt_b, ee_quat=eef_quat_b)
        joint_pos_des = ik_controller.compute(
            eef_pos_b, eef_quat_b,
            jacobian_b[:, :3, :],   # position rows only for position command
            franka.data.joint_pos[:, ARM_JOINT_IDS],
        )

        # Apply
        franka.set_joint_position_target(joint_pos_des, joint_ids=ARM_JOINT_IDS)
        franka.set_joint_position_target(
            torch.tensor([[gripper_norm * 0.04, gripper_norm * 0.04]], device="cuda:0"),
            joint_ids=[7, 8]
        )

        franka.write_data_to_sim()
        sim.step()
        franka.update(sim.get_physics_dt())
        cup.update(sim.get_physics_dt())
        step += 1

        if step % 200 == 0:
            status = "HAND" if hand_age < NO_HAND_TIMEOUT else "HOME"
            print(f"Step {step} | {status} | Smooth: {smooth_target.round(3)} | Gripper: {gripper_norm:.2f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
