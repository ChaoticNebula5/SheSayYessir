import cv2
import time
from src.capture import Camera
from src.detector import Detector
from src.emote_player import EmotePlayer

camera = Camera()
detector = Detector()
emotes = EmotePlayer({
    "neutral": "emotes/neutral.gif",
    "king_laughing": "emotes/king_laughing.gif",
    "jawline": "emotes/jawline.gif",
    "goblin_crying": "emotes/goblin_crying.gif",
    "six_seven": "emotes/six_seven.gif",
})

current_emote = "neutral"
last_switch_time = 0.0
switch_delay = 0.8  
debug_mode = True  

print("Starting....")
print("Press 'q' to quit, 'd' to toggle debug overlay")

while True:
    frame = camera.get_frame()
    if frame is None:
        print("Could not read from camera")
        break

    face_results, pose_results = detector.analyze(frame)
    detected = detector.detect_emote(face_results, pose_results)
    debug_info = detector.get_debug_info(face_results, pose_results) if debug_mode else {}

    now = time.time()
    if detected != current_emote and (now - last_switch_time) > switch_delay:
        current_emote = detected
        emotes.set_emote(current_emote)
        last_switch_time = now
        print(f"Emote changed to: {current_emote}")

    emote_frame = emotes.next_frame()

    if emote_frame is None:
        emote_frame = frame.copy()  

    emote_frame = cv2.resize(emote_frame, (frame.shape[1], frame.shape[0]))

    combined = cv2.hconcat([frame, emote_frame])

    # Main emote label
    cv2.putText(combined, f"{current_emote}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Debug overlay
    if debug_mode and debug_info:
        y_offset = 80
        cv2.putText(combined, "=== DEBUG ===", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 25
        
        # King Laughing metrics
        cv2.putText(combined, f"Mouth: {debug_info.get('mouth_ratio', 0):.3f} (>0.18=laugh)", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(combined, f"Face Angle: {debug_info.get('face_angle', 0):.3f} (<0.08=forward)", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        # Goblin Crying metrics
        cv2.putText(combined, f"L Eye Dist: {debug_info.get('left_eye_dist', 0):.3f} (<0.18=cry)", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(combined, f"R Eye Dist: {debug_info.get('right_eye_dist', 0):.3f} (<0.18=cry)", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(combined, f"Revving: {debug_info.get('revving', False)}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        # Six-Seven metrics
        cv2.putText(combined, f"Alt Counter: {debug_info.get('alt_counter', 0)} (>3=trigger)", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(combined, f"Six Timer: {debug_info.get('six_timer', 0)}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(combined, f"L/R Wrist Vis: {debug_info.get('left_wrist_vis', 0):.2f} / {debug_info.get('right_wrist_vis', 0):.2f}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        
        # Jawline metrics
        cv2.putText(combined, f"Jaw Swipe: {debug_info.get('jaw_swipe_frames', 0)} (>2=trigger)", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Clash Royale Emote Reactor", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

camera.release()
cv2.destroyAllWindows()
