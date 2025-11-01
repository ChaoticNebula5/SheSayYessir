# src/detector.py
import cv2
import mediapipe as mp
from collections import deque
from src.utils import mouth_ratio, distance


class Detector:

    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose

        self.face_mesh = self.mp_face.FaceMesh(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # motion memory for tracking changes
        self.prev_left_y = None
        self.prev_right_y = None
        self.wrist_history = deque(maxlen=6)  # stores recent wrist (yL, yR)

    # ------------------------------------------------------------------

    def analyze(self, frame):
        """Run MediaPipe on a frame and return results."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb)
        pose_results = self.pose.process(rgb)
        return face_results, pose_results

    # ------------------------------------------------------------------

    def detect_emote(self, face_results, pose_results):
        """
        Decide which emote to show.
        Returns one of: king_laughing, jawline, goblin_crying, six_seven, neutral
        """
        emote = "neutral"

        if not face_results.multi_face_landmarks or not pose_results.pose_landmarks:
            return emote

        face = face_results.multi_face_landmarks[0].landmark
        pose = pose_results.pose_landmarks.landmark

        # ---------- KING LAUGHING ----------
        ratio = mouth_ratio(face)
        if ratio > 0.25:
            return "king_laughing"

        # ---------- JAWLINE FLEX ----------
        # Face turned right + left wrist near jaw (mogging ;))
        face_turn_right = face[234].x < face[454].x - 0.02
        left_wrist = pose[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        nose = pose[self.mp_pose.PoseLandmark.NOSE.value]
        hand_near_jaw = (
            abs(left_wrist.y - nose.y) < 0.12 and abs(left_wrist.x - nose.x) < 0.12
        )
        if face_turn_right and hand_near_jaw:
            return "jawline"

        # ---------- GOBLIN CRYING ----------
        # One wrist near eyes +  revving motion
        left_eye = face[33]
        right_eye = face[263]
        right_wrist = pose[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = pose[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = pose[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]

        left_eye_dist = distance(left_wrist, left_eye)
        right_eye_dist = distance(right_wrist, right_eye)
        hand_near_eyes = left_eye_dist < 0.12 or right_eye_dist < 0.12

        # record current wrist Y positions
        self.wrist_history.append((left_wrist.y, right_wrist.y))
        revving_motion = self.is_revving()

        if hand_near_eyes and revving_motion:
            return "goblin_crying"

        # ---------- SIX SEVEN ----------
        lw_y = left_wrist.y
        rw_y = right_wrist.y
        hands_alt = False
        if self.prev_left_y is not None and self.prev_right_y is not None:
            # detect opposite vertical motion
            left_up = lw_y < self.prev_left_y - 0.02
            right_up = rw_y < self.prev_right_y - 0.02
            if left_up ^ right_up:  # XOR → one up & one down
                hands_alt = True

        self.prev_left_y, self.prev_right_y = lw_y, rw_y

        if hands_alt:
            return "six_seven"

        return emote


    def is_revving(self):
        """
        Detects short wrist motion for Goblin Crying
        """
        if len(self.wrist_history) < 5:
            return False

        left_y = [p[0] for p in self.wrist_history]
        right_y = [p[1] for p in self.wrist_history]

        def count_flips(seq):
            diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
            return sum(
                1 for i in range(len(diffs) - 1)
                if diffs[i] * diffs[i + 1] < 0
            )

        left_flips = count_flips(left_y)
        right_flips = count_flips(right_y)

        return left_flips >= 2 or right_flips >= 2
