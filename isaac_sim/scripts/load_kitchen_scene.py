"""
load_kitchen_scene.py
Isaac Sim kitchen scene using NVIDIA Simple Room environment.
Franka Panda + cup + target zone placed inside the room.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Hand Teleop Kitchen Scene")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg

FRANKA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
)
ROOM_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/Environments/Simple_Room/simple_room.usd"
)

TABLE_H  = 0.8
ROBOT_Z  = TABLE_H


def build_scene():
    # Simple Room (walls, floor, ceiling, lighting all included) 
    cfg_room = sim_utils.UsdFileCfg(usd_path=ROOM_USD)
    cfg_room.func("/World/Room", cfg_room, translation=(0.0, 0.0, 0.0))

    # Table
    cfg_table = sim_utils.CuboidCfg(
        size=(0.9, 0.6, TABLE_H),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
    )
    cfg_table.func("/World/Table", cfg_table, translation=(0.5, 0.0, TABLE_H / 2.0))

    # Cup
    cup_cfg = RigidObjectCfg(
        prim_path="/World/Cup",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.08,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.1, TABLE_H + 0.04),
        ),
    )
    cup = RigidObject(cup_cfg)

    # Target zone (red marker)
    cfg_target = sim_utils.CylinderCfg(
        radius=0.07,
        height=0.005,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
    )
    cfg_target.func(
        "/World/TargetZone", cfg_target,
        translation=(0.45, -0.2, TABLE_H + 0.003)
    )

    # Franka
    franka_cfg = ArticulationCfg(
        prim_path="/World/Franka",
        spawn=sim_utils.UsdFileCfg(usd_path=FRANKA_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, ROBOT_Z),
            joint_pos={
                "panda_joint1":  0.0,
                "panda_joint2": -0.785,
                "panda_joint3":  0.0,
                "panda_joint4": -2.356,
                "panda_joint5":  0.0,
                "panda_joint6":  1.571,
                "panda_joint7":  0.785,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                stiffness=800, damping=40,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                stiffness=800, damping=40,
            ),
        },
    )
    franka = Articulation(franka_cfg)

    return franka, cup


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )
    sim.set_camera_view(eye=[-0.3, 0.0, 1.4], target=[0.6, 0.0, 0.85])

    franka, cup = build_scene()

    sim.reset()
    print("Kitchen scene loaded.")
    print(f"Franka joints : {franka.joint_names}")
    print(f"Cup position  : {cup.data.root_pos_w[0, :3]}")

    step = 0
    while simulation_app.is_running():
        sim.step()
        franka.update(sim.get_physics_dt())
        cup.update(sim.get_physics_dt())
        step += 1
        if step % 200 == 0:
            hand_idx = franka.find_bodies("panda_hand")[0][0]
            eef = franka.data.body_pos_w[0, hand_idx]
            print(f"Step {step} | EEF: {eef.cpu().numpy().round(3)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
