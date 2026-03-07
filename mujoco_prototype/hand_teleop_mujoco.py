"""
hand_teleop_mujoco.py
Real-time hand teleoperation of Franka Panda in MuJoCo.
Pipeline: webcam -> MediaPipe HandLandmarker -> wrist pose -> differential IK -> MuJoCo

Position tracking : hand_landmarks (normalized) for workspace mapping.
Gripper detection : hand_world_landmarks (metric 3D) — distance-invariant pinch.
"""

import cv2
import numpy as np
import mujoco
import time
import urllib.request
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from franka_kitchen_env import FrankaKitchenEnv

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"

# ── Workspace mapping ────────────────────────────────────────────────────────
WS_X_RANGE = (0.25, 0.65)
WS_Y_RANGE = (-0.30, 0.30)
WS_Z_RANGE = (0.20, 0.70)

# ── Smoothing ────────────────────────────────────────────────────────────────
EMA_ALPHA = 0.30

# ── Gripper thresholds in real meters (world landmarks) ──────────────────────
PINCH_CLOSE_M = 0.03   # 3cm = fully closed
PINCH_OPEN_M  = 0.07   # 7cm = fully open

# ── Landmark indices ─────────────────────────────────────────────────────────
WRIST     = 0
THUMB_TIP = 4
INDEX_TIP = 8


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def compute_jacobian(model, data, body_id):
    J_pos = np.zeros((3, model.nv))
    J_rot = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, J_pos, J_rot, body_id)
    return np.vstack([J_pos, J_rot])


def ik_step(model, data, body_id, target_pos, target_quat,
            n_iter=10, step_size=0.5, damping=1e-4):
    q = data.qpos[:7].copy()
    for _ in range(n_iter):
        cur_pos  = data.xpos[body_id].copy()
        cur_xmat = data.xmat[body_id].reshape(3, 3)
        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, cur_xmat.flatten())

        pos_err  = target_pos - cur_pos
        neg_cur  = np.zeros(4)
        quat_err = np.zeros(4)
        mujoco.mju_negQuat(neg_cur, cur_quat)
        mujoco.mju_mulQuat(quat_err, target_quat, neg_cur)
        angle   = 2.0 * np.arctan2(np.linalg.norm(quat_err[1:]), abs(quat_err[0]))
        axis    = quat_err[1:] / (np.linalg.norm(quat_err[1:]) + 1e-9)
        rot_err = axis * angle

        err = np.concatenate([pos_err, rot_err])
        J   = compute_jacobian(model, data, body_id)[:, :7]
        JJT = J @ J.T + damping * np.eye(6)
        dq  = J.T @ np.linalg.solve(JJT, err)
        q   = q + step_size * dq
        q   = np.clip(q, model.jnt_range[:7, 0], model.jnt_range[:7, 1])
        data.qpos[:7] = q
        mujoco.mj_fwdPosition(model, data)
    return q


def get_gripper_norm(world_lm):
    """
    Returns 0.0 (closed) to 1.0 (open) based on real 3D thumb-index distance.
    Uses world landmarks so distance is metric and camera-distance invariant.
    """
    t = world_lm[THUMB_TIP]
    i = world_lm[INDEX_TIP]
    dist = np.linalg.norm([t.x - i.x, t.y - i.y, t.z - i.z])
    if dist < PINCH_CLOSE_M:
        return 0.0
    elif dist > PINCH_OPEN_M:
        return 1.0
    else:
        return (dist - PINCH_CLOSE_M) / (PINCH_OPEN_M - PINCH_CLOSE_M)


def get_eef_target(norm_lm, prev_pos):
    wrist = norm_lm[WRIST]
    rx = np.interp(wrist.z + 0.5, [0.0, 1.0], WS_X_RANGE)
    ry = np.interp(1.0 - wrist.x,  [0.0, 1.0], WS_Y_RANGE)
    rz = np.interp(1.0 - wrist.y,  [0.0, 1.0], WS_Z_RANGE)
    raw = np.array([rx, ry, rz])
    smoothed = raw if prev_pos is None else (
        EMA_ALPHA * raw + (1 - EMA_ALPHA) * prev_pos
    )
    return smoothed, np.array([1.0, 0.0, 0.0, 0.0])


def main():
    ensure_model()

    env     = FrankaKitchenEnv(render=True)
    model   = env.model
    data    = env.data
    body_id = env._eef_body_id

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    detector = mp_vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    smooth_pos   = None
    gripper_norm = 1.0
    t_start      = time.time()

    print("Hand teleoperation running. Show right hand to webcam.")
    print("Pinch thumb+index to close gripper. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int((time.time() - t_start) * 1000)
            result = detector.detect_for_video(mp_img, ts_ms)

            if result.hand_landmarks and result.hand_world_landmarks:
                norm_lm  = result.hand_landmarks[0]
                world_lm = result.hand_world_landmarks[0]

                smooth_pos, target_quat = get_eef_target(norm_lm, smooth_pos)
                gripper_norm            = get_gripper_norm(world_lm)

                q_target = ik_step(model, data, body_id, smooth_pos, target_quat)
                env.set_gripper(gripper_norm)
                env.step(q_target)

                # ── Overlay ──────────────────────────────────────────────────
                cx = int((1.0 - norm_lm[WRIST].x) * frame.shape[1])
                cy = int((1.0 - norm_lm[WRIST].y) * frame.shape[0])
                cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

                tx = int((1.0 - norm_lm[THUMB_TIP].x) * frame.shape[1])
                ty = int((1.0 - norm_lm[THUMB_TIP].y) * frame.shape[0])
                ix = int((1.0 - norm_lm[INDEX_TIP].x) * frame.shape[1])
                iy = int((1.0 - norm_lm[INDEX_TIP].y) * frame.shape[0])
                cv2.circle(frame, (tx, ty), 6, (0, 165, 255), -1)
                cv2.circle(frame, (ix, iy), 6, (0, 165, 255), -1)
                cv2.line(frame, (tx, ty), (ix, iy), (0, 165, 255), 2)

                state = "CLOSED" if gripper_norm < 0.1 else "OPEN"
                color = (0, 255, 0) if state == "OPEN" else (0, 0, 255)
                cv2.putText(frame, f"Gripper: {state}  norm={gripper_norm:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame,
                            f"EEF: {smooth_pos[0]:.2f} {smooth_pos[1]:.2f} {smooth_pos[2]:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "No hand detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        env.close()


if __name__ == "__main__":
    main()
