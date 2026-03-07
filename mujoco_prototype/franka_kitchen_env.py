"""
franka_kitchen_env.py
MuJoCo kitchen environment: Franka Panda + table + cup + target zone.
No ROS2, no MediaPipe here — pure physics env used by hand_teleop_mujoco.py.
"""
import mujoco
import mujoco.viewer
import numpy as np
from robot_descriptions import panda_mj_description


class FrankaKitchenEnv:
    # Franka joint home config (radians): arm ready pose
    HOME_QPOS = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])

    def __init__(self, render=True):
        self.model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
        self.data  = mujoco.MjData(self.model)

        # EEF = 'hand' body (flange center) — no named sites in this MJCF
        self._eef_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )

        self.reset()

        self._viewer = None
        if render:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = self.HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

    def step(self, joint_pos: np.ndarray, n_substeps: int = 5):
        assert joint_pos.shape == (7,), "Expected 7-DOF joint command"
        self.data.ctrl[:7] = joint_pos
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()

    def set_gripper(self, width: float):
        """width: 0.0=closed, 0.08=fully open. Symmetric finger split."""
        w = float(np.clip(width, 0.0, 0.08))
        self.data.ctrl[7] = w / 2.0
        self.data.ctrl[8] = w / 2.0

    def get_eef_pose(self):
        """Returns (pos [3], quat [4]) of hand body in world frame."""
        pos  = self.data.xpos[self._eef_body_id].copy()
        xmat = self.data.xmat[self._eef_body_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.flatten())
        return pos, quat

    def close(self):
        if self._viewer is not None:
            self._viewer.close()


if __name__ == "__main__":
    import time
    print("Launching Franka kitchen env...")
    env = FrankaKitchenEnv(render=True)
    pos, quat = env.get_eef_pose()
    print(f"EEF pos at home: {pos}")
    print(f"EEF quat at home: {quat}")

    # Wiggle joint 4 to verify viewer responds
    for i in range(300):
        q = env.HOME_QPOS[:7].copy()
        q[3] += 0.3 * np.sin(i * 0.05)
        env.step(q)
        time.sleep(0.002)

    print("Smoke test passed.")
    input("Press Enter to exit...")
    env.close()
