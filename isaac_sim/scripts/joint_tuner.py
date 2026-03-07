"""
joint_tuner.py
Keyboard control of individual Franka joints.
Press 1-7 to select joint, +/- to increase/decrease.
Press P to print current joint angles.
Press Q to quit.
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import carb
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

FRANKA_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
)

joint_angles = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
selected_joint = 0
step_size = 0.05

def on_keyboard_event(event, *args, **kwargs):
    global selected_joint, joint_angles, step_size
    if event.type == carb.input.KeyboardEventType.KEY_PRESS or \
       event.type == carb.input.KeyboardEventType.KEY_REPEAT:
        key = event.input
        if key == carb.input.KeyboardInput.KEY_1: selected_joint = 0
        elif key == carb.input.KeyboardInput.KEY_2: selected_joint = 1
        elif key == carb.input.KeyboardInput.KEY_3: selected_joint = 2
        elif key == carb.input.KeyboardInput.KEY_4: selected_joint = 3
        elif key == carb.input.KeyboardInput.KEY_5: selected_joint = 4
        elif key == carb.input.KeyboardInput.KEY_6: selected_joint = 5
        elif key == carb.input.KeyboardInput.KEY_7: selected_joint = 6
        elif key == carb.input.KeyboardInput.EQUAL:
            joint_angles[selected_joint] += step_size
            print(f"Joint {selected_joint+1}: {joint_angles[selected_joint]:.3f}")
        elif key == carb.input.KeyboardInput.MINUS:
            joint_angles[selected_joint] -= step_size
            print(f"Joint {selected_joint+1}: {joint_angles[selected_joint]:.3f}")
        elif key == carb.input.KeyboardInput.P:
            print("\nCurrent joint angles:")
            for i, a in enumerate(joint_angles):
                print(f'  "panda_joint{i+1}": {a:.3f},')
        elif key == carb.input.KeyboardInput.Q:
            simulation_app.close()
    return True

def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.4, 0.0, 0.9])

    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/Ground", cfg_ground)

    franka = Articulation(ArticulationCfg(
        prim_path="/World/Franka",
        spawn=sim_utils.UsdFileCfg(usd_path=FRANKA_USD),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={"panda_joint1": 0.0, "panda_joint2": -0.569, "panda_joint3": 0.0, "panda_joint4": -2.810, "panda_joint5": 0.0, "panda_joint6": 3.037, "panda_joint7": 0.741}),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["panda_joint[1-7]"], stiffness=800, damping=40),
            "hand": ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"], stiffness=800, damping=40),
        },
    ))
    sim.reset()

    # Register keyboard
    appwindow = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("Joint Tuner Ready:")
    print("  Press 1-7 to select joint")
    print("  Press = to increase, - to decrease (step: 0.05 rad)")
    print("  Press P to print all angles")
    print("  Press Q to quit")

    while simulation_app.is_running():
        q = torch.tensor([joint_angles], dtype=torch.float32, device="cuda:0")
        franka.set_joint_position_target(q[:, :7], joint_ids=list(range(7)))
        franka.write_data_to_sim()
        sim.step()
        franka.update(sim.get_physics_dt())

if __name__ == "__main__":
    import omni.appwindow
    main()
    simulation_app.close()
