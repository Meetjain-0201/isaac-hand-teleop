"""
pyramid_scene.py
3-2-1 pyramid stacking task.
5 cups pre-placed: 3 on bottom row, 2 on middle row.
You pick the white cup and place it on top to complete the pyramid.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pyramid stacking scene")
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
import threading, socket, json, time, numpy as np

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
HOME_POS      = np.array([0.55, 0.0, 1.15])
MAX_DELTA     = 0.05
NO_HAND_TIMEOUT = 2.0
GRIPPER_DOWN_QUAT = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

CUP_R    = 0.03
CUP_H    = 0.08
SPACING  = 0.07   # center-to-center gap between cups in same row
PYR_X    = 0.55   # how far forward the pyramid sits on the table

# ── Pre-placed cups (kinematic — they don't fall) ────────────────────────────
# Row 1: 3 cups, centered at y=0
# y positions: -0.07, 0.00, +0.07
ROW1_Z = TABLE_H + CUP_H / 2.0
ROW1 = [
    (PYR_X, -SPACING, ROW1_Z),
    (PYR_X,  0.0,     ROW1_Z),
    (PYR_X, +SPACING, ROW1_Z),
]

# Row 2: 2 cups, centered between row 1 cups
# y positions: -0.035, +0.035
ROW2_Z = ROW1_Z + CUP_H
ROW2 = [
    (PYR_X, -SPACING / 2.0, ROW2_Z),
    (PYR_X, +SPACING / 2.0, ROW2_Z),
]

# Where the final cup should go (apex)
APEX_Z = ROW2_Z + CUP_H
APEX   = (PYR_X, 0.0, APEX_Z)

# Loose cup — white, sitting to the side for you to pick up
LOOSE_POS = (0.35, -0.2, TABLE_H + CUP_H / 2.0)


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


def spawn_static_cup(prim_path, pos, color):
    """Spawn a kinematic (non-moving) cup."""
    cfg = sim_utils.CylinderCfg(
        radius=CUP_R,
        height=CUP_H,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    cfg.func(prim_path, cfg, translation=pos)


def build_scene():
    # Room
    sim_utils.UsdFileCfg(usd_path=ROOM_USD).func(
        "/World/Room", sim_utils.UsdFileCfg(usd_path=ROOM_USD))

    # Table
    cfg_t = sim_utils.CuboidCfg(
        size=(0.9, 0.6, TABLE_H),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
    )
    cfg_t.func("/World/Table", cfg_t, translation=(0.5, 0.0, TABLE_H / 2.0))

    # Row 1 — 3 red cups
    for i, pos in enumerate(ROW1):
        spawn_static_cup(f"/World/Row1Cup{i}", pos, color=(0.85, 0.15, 0.15))

    # Row 2 — 2 orange cups
    for i, pos in enumerate(ROW2):
        spawn_static_cup(f"/World/Row2Cup{i}", pos, color=(0.95, 0.55, 0.05))

    # Loose cup — white, physics enabled (you pick this one)
    loose_cup = RigidObject(RigidObjectCfg(
        prim_path="/World/LooseCup",
        spawn=sim_utils.CylinderCfg(
            radius=CUP_R,
            height=CUP_H,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=LOOSE_POS),
    ))

    # Franka
    franka = Articulation(ArticulationCfg(
        prim_path="/World/Franka",
        spawn=sim_utils.UsdFileCfg(usd_path=FRANKA_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_H),
            joint_pos={
                "panda_joint1": 0.0,   "panda_joint2": 0.181,
                "panda_joint3": 0.0,   "panda_joint4": -1.860,
                "panda_joint5": 0.0,   "panda_joint6": 2.087,
                "panda_joint7": 0.691,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"], stiffness=800, damping=40),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"], stiffness=800, damping=40),
        },
    ))
    return franka, loose_cup


def main():
    threading.Thread(target=udp_listener, daemon=True).start()

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    sim.set_camera_view(eye=[0.0, -1.3, 1.5], target=[0.5, 0.0, 1.0])

    franka, loose_cup = build_scene()
    sim.reset()

    # Force home pose
    franka.write_joint_state_to_sim(
        franka.data.default_joint_pos.clone(),
        franka.data.default_joint_vel.clone()
    )
    franka.reset()
    sim.step()
    franka.update(sim.get_physics_dt())

    ik_controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        ),
        num_envs=1, device="cuda:0"
    )
    ik_controller.reset()

    hand_idx          = franka.find_bodies("panda_hand")[0][0]
    jacobi_idx        = hand_idx - 1
    gripper_down_quat = GRIPPER_DOWN_QUAT.to("cuda:0")

    print("=== PYRAMID SCENE READY ===")
    print(f"Loose white cup at: {LOOSE_POS}")
    print(f"Place it at apex:   {APEX}")
    print("Right hand = XY position | Left hand = Z height + gripper")

    smooth_target = HOME_POS.copy()
    step = 0

    while simulation_app.is_running():
        with state.lock:
            raw_target   = state.target_pos.copy()
            gripper_norm = state.gripper_norm
            hand_age     = time.time() - state.last_hand_time

        if hand_age > NO_HAND_TIMEOUT:
            raw_target   = HOME_POS.copy()
            gripper_norm = 1.0

        delta  = raw_target - smooth_target
        d_norm = np.linalg.norm(delta)
        if d_norm > MAX_DELTA:
            delta = delta * (MAX_DELTA / d_norm)
        smooth_target = smooth_target + delta

        smooth_target[0] = np.clip(smooth_target[0], 0.2,  0.75)
        smooth_target[1] = np.clip(smooth_target[1], -0.4, 0.4)
        smooth_target[2] = np.clip(smooth_target[2], TABLE_H + 0.05, 1.45)

        jacobian_w = franka.root_physx_view.get_jacobians()[:, jacobi_idx, :, ARM_JOINT_IDS]
        root_rot   = matrix_from_quat(quat_inv(franka.data.root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(root_rot, jacobian_b[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(root_rot, jacobian_b[:, 3:, :])

        eef_pos_b, eef_quat_b = subtract_frame_transforms(
            franka.data.root_pos_w, franka.data.root_quat_w,
            franka.data.body_pos_w[:, hand_idx],
            franka.data.body_quat_w[:, hand_idx],
        )

        tgt_w = torch.tensor(smooth_target, dtype=torch.float32, device="cuda:0").unsqueeze(0)
        tgt_b, _ = subtract_frame_transforms(
            franka.data.root_pos_w, franka.data.root_quat_w, tgt_w)

        ik_controller.set_command(torch.cat([tgt_b, gripper_down_quat], dim=-1))
        joint_pos_des = ik_controller.compute(
            eef_pos_b, eef_quat_b, jacobian_b,
            franka.data.joint_pos[:, ARM_JOINT_IDS],
        )

        franka.set_joint_position_target(joint_pos_des, joint_ids=ARM_JOINT_IDS)
        franka.set_joint_position_target(
            torch.tensor([[gripper_norm * 0.04, gripper_norm * 0.04]], device="cuda:0"),
            joint_ids=[7, 8]
        )

        franka.write_data_to_sim()
        sim.step()
        franka.update(sim.get_physics_dt())
        loose_cup.update(sim.get_physics_dt())
        step += 1

        if step % 200 == 0:
            cup_z = loose_cup.data.root_pos_w[0, 2].item()
            print(f"Step {step} | Cup Z: {cup_z:.3f} | Apex Z: {APEX[2]:.3f} | Gripper: {gripper_norm:.2f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
