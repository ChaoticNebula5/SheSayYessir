import imageio.v2 as imageio
import cv2

class EmotePlayer:
    
    def __init__(self, emote_paths: dict[str, str]):
        
        self.frames: dict[str, list] = {}
        
        for name, path in emote_paths.items():
            frames = imageio.mimread(path)
            self.frames[name] = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
        
        self.current = list(emote_paths.keys())[0]
        self.index = 0
        
    def set_emote(self, name: str):
        """Switch GIFS"""
        if name != self.current and name in self.frames:
            self.current = name
            self.index = 0
        
    def next_frame(self):
        """Loop the gif"""
        frames = self.frames[self.current]
        frame = frames[self.index%len(frames)]
        self.index+=1
        return cv2.resize(frame, (720, 450))