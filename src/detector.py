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

        self.prev_left_y = None
        self.prev_right_y = None
        self.wrist_history = deque(maxlen=6)
        self.alt_counter = 0
        self.six_timer = 0
        
        self.prev_jaw_hand_x = None
        self.jaw_swipe_frames = 0


    def analyze(self, frame):
        """Run MediaPipe on a frame and return results."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb)
        pose_results = self.pose.process(rgb)
        return face_results, pose_results


    def detect_emote(self, face_results, pose_results):
        """
        Decide which emote to show.
        Returns one of king_laughing, jawline, goblin_crying, six_seven, neutral
        """
        emote = "neutral"

        if not face_results.multi_face_landmarks or not pose_results.pose_landmarks:
            return emote

        face = face_results.multi_face_landmarks[0].landmark
        pose = pose_results.pose_landmarks.landmark

        # ---------- COMMON LANDMARKS ----------
        left_wrist = pose[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = pose[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = pose[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        jaw_ref = face[152]
        left_eye = face[33]
        right_eye = face[263]
        nose_tip = face[1]

        #face turn angle
        face_angle = abs(face[234].x - face[454].x)

        # ---------- JAWLINE FLEX ----------
        # debug values: face angle 0.17-0.19 when turned, 0.24+ when forward
        face_turned_right = face_angle < 0.20  #0.17-0.19 range
        
        # Tighter to avoid stealing six-seven (which reaches 0.68)
        left_hand_near_face = distance(left_wrist, jaw_ref) < 0.40
        right_hand_near_face = distance(right_wrist, jaw_ref) < 0.60  # was 0.70, tightened
        
        if face_turned_right and (left_hand_near_face or right_hand_near_face):
            return "jawline"

        # ---------- GOBLIN CRYING ----------
        #Eye dist 0.35-0.39, Face ~0.2
        left_near_eyes = distance(left_wrist, left_eye) < 0.42  #0.35-0.39 range
        right_near_eyes = distance(right_wrist, right_eye) < 0.42
        
        # Record wrist positions
        self.wrist_history.append((left_wrist.y, right_wrist.y))
        has_movement = self.is_revving()
        
        if (left_near_eyes or right_near_eyes) and has_movement:
            return "goblin_crying"

        # ---------- KING LAUGHING ----------
        # Mouth ~0.25, Face ~0.2
        ratio = mouth_ratio(face)
        if ratio > 0.23:  
            return "king_laughing"

        # ---------- SIX SEVEN ----------
        lw_y = left_wrist.y
        rw_y = right_wrist.y

        hands_visible = (
            left_wrist.visibility > 0.5 and 
            right_wrist.visibility > 0.5
        )

        if hands_visible and self.prev_left_y is not None and self.prev_right_y is not None:
            # More sensitive motion detection (was 0.01, now 0.005)
            left_up = lw_y < self.prev_left_y - 0.005 
            left_down = lw_y > self.prev_left_y + 0.005
            right_up = rw_y < self.prev_right_y - 0.005
            right_down = rw_y > self.prev_right_y + 0.005
            
            alternating = (left_up and right_down) or (left_down and right_up)
            
            both_same_direction = (left_up and right_up) or (left_down and right_down)

            if alternating and not both_same_direction:
                self.alt_counter += 4  
            else:
                self.alt_counter = max(0, self.alt_counter - 1)

            if self.alt_counter > 3:  
                self.six_timer = 20
                self.alt_counter = 0
        else:
            # Reset counter if hands not visible
            self.alt_counter = 0

        self.prev_left_y, self.prev_right_y = lw_y, rw_y

        if self.six_timer > 0:
            self.six_timer -= 1
            return "six_seven"

        return emote


    def is_revving(self):
        """
        Detects rapid wrist motion for Goblin Crying
        """
        if len(self.wrist_history) < 4:
            return False

        left_y = [p[0] for p in self.wrist_history]
        right_y = [p[1] for p in self.wrist_history]

        def has_motion(seq):
            """Check for up-down motion"""
            diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
            flips = sum(
                1 for i in range(len(diffs) - 1)
                if diffs[i] * diffs[i + 1] < 0
            )
            has_movement = any(abs(d) > 0.008 for d in diffs)
            return flips >= 1 and has_movement

        return has_motion(left_y) or has_motion(right_y)

    def get_debug_info(self, face_results, pose_results):
        """
        Returns debug info for visualization
        """
        if not face_results.multi_face_landmarks or not pose_results.pose_landmarks:
            return {}

        face = face_results.multi_face_landmarks[0].landmark
        pose = pose_results.pose_landmarks.landmark

        left_wrist = pose[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = pose[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = pose[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        jaw_ref = face[152]
        left_eye = face[33]
        right_eye = face[263]
        nose_tip = face[1]

        left_eye_dist = distance(left_wrist, left_eye)
        right_eye_dist = distance(right_wrist, right_eye)
        face_angle = abs(face[234].x - face[454].x)
        mouth_r = mouth_ratio(face)
        
        return {
            "mouth_ratio": mouth_r,
            "face_angle": face_angle,
            "left_eye_dist": left_eye_dist,
            "right_eye_dist": right_eye_dist,
            "alt_counter": self.alt_counter,
            "six_timer": self.six_timer,
            "jaw_swipe_frames": self.jaw_swipe_frames,
            "revving": self.is_revving(),
            "left_wrist_vis": left_wrist.visibility,
            "right_wrist_vis": right_wrist.visibility,
        }
