"""
AURA - Week 1 Starter
Webcam + MediaPipe hand tracking + a pet sprite pinned to your wrist.

This is MILESTONE 1 of your synopsis: 'Gesture summon works' starts here.
Read every line. Every comment is there for you.
"""
import time
import cv2
import mediapipe as mp
import numpy as np
import yaml
from pathlib import Path

# ---------- load config (reproducibility: nothing hard-coded) ----------
cfg = yaml.safe_load(open(Path(__file__).parent.parent / "configs" / "config.yaml"))

PET_IMG = cv2.imread(str(Path(__file__).parent.parent / "assets" / "pet.png"),
                     cv2.IMREAD_UNCHANGED)  # keeps transparency (alpha channel)

# ---------- MediaPipe hand tracker ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,   # video: use tracking (faster)
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

# ---------- helpers ----------
def overlay_png(bg, png, x, y, scale_h):
    """Paste png (with alpha) onto bg so png's height = scale_h * bg height.
    (x, y) is the CENTER of where the pet goes. Returns modified bg."""
    h = int(bg.shape[0] * scale_h)
    w = int(png.shape[1] * h / png.shape[0])
    png_r = cv2.resize(png, (w, h), interpolation=cv2.INTER_AREA)

    # clip if pet goes off-screen (so it never crashes at the edges)
    x1, y1 = int(x - w / 2), int(y - h * 0.9)   # 0.9: anchor near bottom = "sits on hand"
    x2, y2 = x1 + w, y1 + h
    cx1, cy1 = max(x1, 0), max(y1, 0)
    cx2, cy2 = min(x2, bg.shape[1]), min(y2, bg.shape[0])
    if cx2 <= cx1 or cy2 <= cy1:
        return bg

    sprite = png_r[cy1 - y1:cy2 - y1, cx1 - x1:cx2 - x1]
    alpha = sprite[:, :, 3:4] / 255.0
    roi = bg[cy1:cy2, cx1:cx2]
    bg[cy1:cy2, cx1:cx2] = (sprite[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)
    return bg

def count_curled_fingers(lm):
    """Rough fist detector: a finger is 'curled' if its TIP is below its PIP.
    Returns how many of the 4 fingers are curled (thumb ignored).
    Landmark y is normalized 0..1, and y grows DOWNWARD, so tip.y > pip.y = curled."""
    tips   = [8, 12, 16, 20]   # index, middle, ring, pinky tips
    pips   = [6, 10, 14, 18]   # the joint one step down the finger
    curled = 0
    for t, p in zip(tips, pips):
        if lm[t].y > lm[p].y:
            curled += 1
    return curled

# ---------- main loop ----------
cap = cv2.VideoCapture(cfg["camera_index"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])

# smoothed pet position (start in the middle)
pet_x, pet_y = cfg["width"] // 2, cfg["height"] // 2
smooth = cfg["smoothing"]

print("Running! Show your hand. Press 'q' to quit.")

while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        print("Camera frame failed. Check camera_index in config.yaml")
        break

    # perception at reduced width (per synopsis design decision)
    small = cv2.resize(frame, (cfg["process_width"],
                               int(frame.shape[0] * cfg["process_width"] / frame.shape[1])))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    gesture = "no hand"

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]     # first detected hand
        lm = hand.landmark

        if cfg["show_landmarks"]:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        wrist = lm[0]                          # landmark 0 = wrist
        # landmarks are normalized (0..1), so multiply by full-frame size
        tx, ty = wrist.x * frame.shape[1], wrist.y * frame.shape[0]

        # THE SMOOTHNESS TRICK: move only a fraction toward the target each frame
        pet_x += (tx - pet_x) * (1 - smooth)
        pet_y += (ty - pet_y) * (1 - smooth)

        curled = count_curled_fingers(lm)
        gesture = "FIST" if curled >= 3 else "OPEN PALM"

    frame = overlay_png(frame, PET_IMG, pet_x, pet_y, cfg["pet_scale"])
    cv2.putText(frame, gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    fps = 1.0 / max(time.time() - t0, 1e-6)
    cv2.putText(frame, f"FPS: {fps:.0f}", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("AURA - starter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
