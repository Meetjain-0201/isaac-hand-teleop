"""
franka_kitchen_env.py
MuJoCo kitchen environment: Franka Panda.
Gripper ctrl range is 0-255 (as per actuator8 in menagerie MJCF).
set_gripper() accepts 0.0 (closed) to 1.0 (fully open).
"""
import mujoco
import mujoco.viewer
import numpy as np
from robot_descriptions import panda_mj_description


class FrankaKitchenEnv:
    HOME_QPOS = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])

    def __init__(self, render=True):
        self.model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
        self.data  = mujoco.MjData(self.model)

        self._eef_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )

        # 0.0=closed, 1.0=open — scaled to 0-255 internally
        self._gripper_norm = 1.0
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
        self.data.ctrl[7]  = self._gripper_norm * 255.0
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()

    def set_gripper(self, norm: float):
        """norm: 0.0=closed, 1.0=fully open."""
        self._gripper_norm = float(np.clip(norm, 0.0, 1.0))

    def get_eef_pose(self):
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
    env = FrankaKitchenEnv(render=True)
    print(f"EEF pos at home: {env.get_eef_pose()[0]}")
    for i in range(600):
        env.set_gripper(1.0 if (i // 50) % 2 == 0 else 0.0)
        env.step(env.HOME_QPOS[:7].copy())
        if i % 50 == 0:
            print(f"ctrl[7]={env.data.ctrl[7]:.1f}  qpos[7]={env.data.qpos[7]:.4f}")
        time.sleep(0.002)
    env.close()
