# AURA - Starter Kit (Week 1 Milestone)

Run:
    pip install -r requirements.txt
    python src/starter_pet.py

Controls:
    q  - quit

What you should see:
    - your webcam feed
    - a hand skeleton when your hand is visible
    - a pet creature sitting on your wrist, gliding smoothly
    - FPS in the corner

Week-1 exercise (do this yourself, no help):
    The script currently prints OPEN PALM / FIST based on curled fingers.
    Modify it so the pet only appears when you show an OPEN PALM,
    and disappears when you make a FIST. (Hint: an `if` around the pet draw.)

Troubleshooting:
    - Black window / no camera  -> change camera_index in configs/config.yaml to 1
    - Slow FPS                 -> already running at reduced resolution; close other apps
    - Webcam light on but black -> another app (Zoom/Teams) is holding the camera
