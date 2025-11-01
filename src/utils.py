import math

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def mouth_ratio(face_landmarks):
    
    top_lip = face_landmarks[13]
    bottom_lip = face_landmarks[14]
    left_mouth = face_landmarks[61]
    right_mouth = face_landmarks[291]
    
    vertical_gap = distance(top_lip, bottom_lip)
    mouth_width = distance(left_mouth, right_mouth)
    
    if mouth_width == 0:
        return 0.0
    
    return vertical_gap / mouth_width
