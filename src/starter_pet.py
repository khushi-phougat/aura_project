"""
AURA - Week 1 Starter
Webcam + MediaPipe hand tracking + a pet sprite pinned to your wrist.

This is MILESTONE 1 of your synopsis: 'Gesture summon works' starts here.
"""
import sys
import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
import yaml
from pathlib import Path

# ---------- load config (reproducibility: nothing hard-coded) ----------
config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
try:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
except Exception as e:
    print(f"Error loading config at {config_path}: {e}")
    sys.exit(1)

pet_path = Path(__file__).parent.parent / "assets" / "pet.png"
PET_IMG = cv2.imread(str(pet_path), cv2.IMREAD_UNCHANGED)  # keeps transparency (alpha channel)
if PET_IMG is None:
    print(f"Warning: Could not load pet image from {pet_path}")

# ---------- MediaPipe hand tracker wrapper ----------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # pinky / palm
]

def draw_custom_landmarks(image, landmarks):
    h, w = image.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(pts) and end_idx < len(pts):
            cv2.line(image, pts[start_idx], pts[end_idx], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(image, pt, 4, (0, 0, 255), -1)

class HandTracker:
    def __init__(self, model_path):
        self.use_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        if self.use_solutions:
            self.mp_hands = mp.solutions.hands
            self.detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            self.mp_draw = mp.solutions.drawing_utils
        else:
            model_file = Path(model_path)
            if not model_file.exists():
                print(f"Downloading hand landmarker model to {model_file}...")
                model_file.parent.mkdir(parents=True, exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, str(model_file))

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            base_options = python.BaseOptions(model_asset_path=str(model_file))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.5,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, rgb_image):
        if self.use_solutions:
            res = self.detector.process(rgb_image)
            if res.multi_hand_landmarks:
                return res.multi_hand_landmarks[0].landmark
            return None
        else:
            rgb_cont = np.ascontiguousarray(rgb_image)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_cont)
            res = self.detector.detect(mp_image)
            if res.hand_landmarks:
                return res.hand_landmarks[0]
            return None

    def draw_landmarks(self, frame, landmarks):
        if self.use_solutions:
            class DummyHand:
                def __init__(self, lm): self.landmark = lm
            self.mp_draw.draw_landmarks(frame, DummyHand(landmarks), self.mp_hands.HAND_CONNECTIONS)
        else:
            draw_custom_landmarks(frame, landmarks)

    def close(self):
        if hasattr(self.detector, "close"):
            self.detector.close()

# ---------- helpers ----------
def overlay_png(bg, png, x, y, scale_h):
    """Paste png (with alpha) onto bg so png's height = scale_h * bg height.
    (x, y) is the CENTER of where the pet goes. Returns modified bg."""
    if png is None or bg is None or bg.size == 0:
        return bg

    bg_h, bg_w = bg.shape[:2]
    h = int(bg_h * scale_h)
    if h <= 0 or png.shape[0] == 0 or png.shape[1] == 0:
        return bg

    w = int(png.shape[1] * h / png.shape[0])
    if w <= 0:
        return bg

    png_r = cv2.resize(png, (w, h), interpolation=cv2.INTER_AREA)

    # clip if pet goes off-screen (so it never crashes at the edges)
    x1, y1 = int(x - w / 2), int(y - h * 0.9)   # 0.9: anchor near bottom = "sits on hand"
    x2, y2 = x1 + w, y1 + h
    cx1, cy1 = max(x1, 0), max(y1, 0)
    cx2, cy2 = min(x2, bg_w), min(y2, bg_h)
    if cx2 <= cx1 or cy2 <= cy1:
        return bg

    sprite = png_r[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
    
    # handle both 3-channel (no alpha) and 4-channel (with alpha) images gracefully
    if sprite.shape[2] == 4:
        alpha = sprite[:, :, 3:4] / 255.0
        roi = bg[cy1:cy2, cx1:cx2]
        bg[cy1:cy2, cx1:cx2] = (sprite[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)
    else:
        bg[cy1:cy2, cx1:cx2] = sprite[:, :, :3]
        
    return bg

def count_curled_fingers(lm):
    """Rough fist detector: a finger is 'curled' if its TIP is below its PIP.
    Returns how many of the 4 fingers are curled (thumb ignored).
    Landmark y is normalized 0..1, and y grows DOWNWARD, so tip.y > pip.y = curled."""
    if not lm or len(lm) < 21:
        return 0
    tips   = [8, 12, 16, 20]   # index, middle, ring, pinky tips
    pips   = [6, 10, 14, 18]   # the joint one step down the finger
    curled = 0
    for t, p in zip(tips, pips):
        if lm[t].y > lm[p].y:
            curled += 1
    return curled

def init_camera(preferred_idx, width=1280, height=720):
    """Attempt to open camera by index using DirectShow on Windows, setting resolution before reading."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    indices_to_try = [preferred_idx] + [i for i in range(4) if i != preferred_idx]

    for idx in indices_to_try:
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                    # Verify camera reads a valid frame
                    for _ in range(5):
                        ok, frame = cap.read()
                        if ok and frame is not None and isinstance(frame, np.ndarray) and frame.size > 0 and frame.ndim == 3:
                            print(f"Successfully opened camera on index {idx} (backend {backend})")
                            return cap
                        time.sleep(0.05)
                    cap.release()
            except Exception:
                pass
    return None

def main():
    cap = init_camera(cfg.get("camera_index", 0), cfg.get("width", 1280), cfg.get("height", 720))
    if cap is None:
        print("Error: Could not open any camera frame. Check camera connection or camera_index in config.yaml")
        sys.exit(1)

    model_task_path = Path(__file__).parent.parent / "models" / "hand_landmarker.task"
    tracker = HandTracker(model_task_path)

    # smoothed pet position (start in the middle)
    pet_x = cfg.get("width", 1280) // 2
    pet_y = cfg.get("height", 720) // 2
    smooth = cfg.get("smoothing", 0.35)

    print("Running! Show your hand. Press 'q' or close window to quit.")
    window_name = "AURA - starter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    failed_reads = 0
    try:
        while True:
            t0 = time.time()
            try:
                ok, frame = cap.read()
            except cv2.error:
                failed_reads += 1
                if failed_reads > 50:
                    print("Camera failed continuously.")
                    break
                time.sleep(0.02)
                continue

            if not ok or frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3:
                failed_reads += 1
                if failed_reads > 50:
                    print("Camera frame failed or empty.")
                    break
                time.sleep(0.02)
                continue

            failed_reads = 0

            # perception at reduced width (per synopsis design decision)
            proc_w = cfg.get("process_width", 640)
            small_h = max(1, int(frame.shape[0] * proc_w / frame.shape[1]))
            small = cv2.resize(frame, (proc_w, small_h))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            lm = tracker.process(rgb)

            gesture = "no hand"
            show_pet = False

            if lm:
                if cfg.get("show_landmarks", True):
                    tracker.draw_landmarks(frame, lm)

                wrist = lm[0]                          # landmark 0 = wrist
                # landmarks are normalized (0..1), so multiply by full-frame size
                tx, ty = wrist.x * frame.shape[1], wrist.y * frame.shape[0]

                # THE SMOOTHNESS TRICK: move only a fraction toward the target each frame
                pet_x += (tx - pet_x) * (1 - smooth)
                pet_y += (ty - pet_y) * (1 - smooth)

                curled = count_curled_fingers(lm)
                if curled >= 3:
                    gesture = "FIST"
                    show_pet = False
                else:
                    gesture = "OPEN PALM"
                    show_pet = True

            # Only display pet on wrist when gesture is OPEN PALM (summoned)
            if show_pet and PET_IMG is not None:
                frame = overlay_png(frame, PET_IMG, pet_x, pet_y, cfg.get("pet_scale", 0.22))

            cv2.putText(frame, f"Gesture: {gesture}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            fps = 1.0 / max(time.time() - t0, 1e-6)
            cv2.putText(frame, f"FPS: {fps:.0f}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)
            
            # Clean exit on 'q' or window close button click
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()