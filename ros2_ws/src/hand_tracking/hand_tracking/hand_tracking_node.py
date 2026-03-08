#!/usr/bin/env python3
"""
hand_tracking_node.py
Two-hand teleoperation:
  RIGHT hand → X/Y position on table
  LEFT hand  → Z (height) + gripper

Publishes ROS2 topics + sends UDP to Isaac Sim.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

import cv2
import numpy as np
import time
import urllib.request
import os
import socket
import json

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = os.path.expanduser("~/hand_landmarker.task")

# Right hand controls X/Y workspace
WS_X_RANGE = (0.25, 0.65)   # hand Y in image -> robot X (forward/back)
WS_Y_RANGE = (-0.30, 0.30)  # hand X in image -> robot Y (left/right)

# Left hand controls Z
WS_Z_RANGE = (0.86, 1.25)   # left hand Y in image -> robot Z (height)

EMA_ALPHA     = 0.30
PINCH_CLOSE_M = 0.03
PINCH_OPEN_M  = 0.07

UDP_IP   = "127.0.0.1"
UDP_PORT = 5005

WRIST     = 0
THUMB_TIP = 4
INDEX_TIP = 8


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


class HandTrackingNode(Node):
    def __init__(self):
        super().__init__('hand_tracking_node')

        self.pub_pose    = self.create_publisher(PoseStamped, '/hand/eef_pose', 10)
        self.pub_gripper = self.create_publisher(Float32, '/hand/gripper_cmd', 10)

        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.smooth_pos   = None
        self.smooth_z     = 1.05   # default Z
        self.t_start      = time.time()

        ensure_model()
        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,                    # detect both hands
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(opts)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info(
            "Two-hand tracking: RIGHT=XY position, LEFT=Z+gripper")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms  = int((time.time() - self.t_start) * 1000)
        result = self.detector.detect_for_video(mp_img, ts_ms)

        right_norm_lm  = None
        left_world_lm  = None
        left_norm_lm   = None
        gripper_norm   = 1.0

        # Separate hands by handedness label
        if result.hand_landmarks and result.handedness:
            for i, handedness in enumerate(result.handedness):
                label = handedness[0].display_name  # "Left" or "Right"
                if label == "Right" and i < len(result.hand_landmarks):
                    right_norm_lm = result.hand_landmarks[i]
                elif label == "Left" and i < len(result.hand_landmarks):
                    left_norm_lm  = result.hand_landmarks[i]
                    if i < len(result.hand_world_landmarks):
                        left_world_lm = result.hand_world_landmarks[i]

        # RIGHT HAND -> X/Y
        if right_norm_lm:
            wrist = right_norm_lm[WRIST]
            rx = float(np.interp(1.0 - wrist.y, [0.0, 1.0], WS_X_RANGE))
            ry = float(np.interp(wrist.x,        [0.0, 1.0], WS_Y_RANGE))
            raw = np.array([rx, ry])
            if self.smooth_pos is None:
                self.smooth_pos = raw
            else:
                self.smooth_pos = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self.smooth_pos

            # Draw right hand wrist
            h, w = frame.shape[:2]
            cx = int((1.0 - right_norm_lm[WRIST].x) * w)
            cy = int((1.0 - right_norm_lm[WRIST].y) * h)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(frame, "R", (cx+12, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            if self.smooth_pos is None:
                self.smooth_pos = np.array([0.45, 0.0])

        # LEFT HAND -> Z + gripper 
        if left_norm_lm:
            wrist_l = left_norm_lm[WRIST]
            # Left hand height in image -> Z
            rz = float(np.interp(1.0 - wrist_l.y, [0.0, 1.0], WS_Z_RANGE))
            self.smooth_z = EMA_ALPHA * rz + (1 - EMA_ALPHA) * self.smooth_z

            # Gripper from left hand world landmarks
            if left_world_lm:
                t = left_world_lm[THUMB_TIP]
                idx = left_world_lm[INDEX_TIP]
                dist = float(np.linalg.norm(
                    [t.x - idx.x, t.y - idx.y, t.z - idx.z]))
                if dist < PINCH_CLOSE_M:
                    gripper_norm = 0.0
                elif dist > PINCH_OPEN_M:
                    gripper_norm = 1.0
                else:
                    gripper_norm = (dist - PINCH_CLOSE_M) / (PINCH_OPEN_M - PINCH_CLOSE_M)

            # Draw left hand
            h, w = frame.shape[:2]
            cx = int((1.0 - left_norm_lm[WRIST].x) * w)
            cy = int((1.0 - left_norm_lm[WRIST].y) * h)
            cv2.circle(frame, (cx, cy), 10, (255, 165, 0), -1)
            cv2.putText(frame, "L", (cx+12, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            # Draw thumb-index line
            tx = int((1.0 - left_norm_lm[THUMB_TIP].x) * w)
            ty = int((1.0 - left_norm_lm[THUMB_TIP].y) * h)
            ix = int((1.0 - left_norm_lm[INDEX_TIP].x) * w)
            iy = int((1.0 - left_norm_lm[INDEX_TIP].y) * h)
            cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1)
            cv2.circle(frame, (ix, iy), 6, (0, 0, 255), -1)
            cv2.line(frame, (tx, ty), (ix, iy), (0, 0, 255), 2)

        # Build final target  
        final_pos = np.array([
            self.smooth_pos[0],
            self.smooth_pos[1],
            self.smooth_z
        ])

        # ROS2 publish 
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "robot_base"
        pose_msg.pose.position.x = float(final_pos[0])
        pose_msg.pose.position.y = float(final_pos[1])
        pose_msg.pose.position.z = float(final_pos[2])
        pose_msg.pose.orientation.w = 1.0
        self.pub_pose.publish(pose_msg)

        gmsg = Float32()
        gmsg.data = float(gripper_norm)
        self.pub_gripper.publish(gmsg)

        # UDP to Isaac Sim
        packet = json.dumps({
            "x": float(final_pos[0]),
            "y": float(final_pos[1]),
            "z": float(final_pos[2]),
            "g": float(gripper_norm),
        }).encode()
        self.udp_sock.sendto(packet, (UDP_IP, UDP_PORT))

        # Overlay
        state = "CLOSED" if gripper_norm < 0.1 else "OPEN"
        color = (0, 255, 0) if state == "OPEN" else (0, 0, 255)
        cv2.putText(frame, f"Gripper: {state}  [{gripper_norm:.2f}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame,
                    f"EEF: {final_pos[0]:.2f} {final_pos[1]:.2f} {final_pos[2]:.2f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "GREEN=Right(XY)  ORANGE=Left(Z+grip)",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Hand Tracking", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        self.udp_sock.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = HandTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
