import os
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

DATA_DIR = Path("data")
SAMPLES_DIR = DATA_DIR / "sample_images"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def create_street_scene(save_path: Path):
    # 800x600 canvas with sky, road, buildings
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    # Sky
    img[0:300, :] = [235, 206, 135]  # Light sky blue in BGR
    # Buildings background
    cv2.rectangle(img, (50, 80), (250, 300), (120, 110, 100), -1)
    cv2.rectangle(img, (280, 50), (520, 300), (90, 80, 80), -1)
    cv2.rectangle(img, (550, 100), (750, 300), (130, 120, 110), -1)
    # Windows
    for y in range(100, 280, 40):
        for x in range(70, 230, 40):
            cv2.rectangle(img, (x, y), (x+25, y+25), (255, 255, 200), -1)
    # Road
    img[300:600, :] = [50, 50, 50]
    # Road lane markings
    for x in range(20, 800, 80):
        cv2.rectangle(img, (x, 440), (x+45, 450), (255, 255, 255), -1)

    # Convert to PIL for photo-like compositing or draw standard shapes
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Car body (Red)
    draw.rectangle([120, 360, 360, 480], fill=(220, 30, 30))
    draw.polygon([(160, 360), (200, 300), (300, 300), (330, 360)], fill=(180, 20, 20)) # cabin
    draw.ellipse([150, 460, 200, 510], fill=(20, 20, 20)) # wheel 1
    draw.ellipse([280, 460, 330, 510], fill=(20, 20, 20)) # wheel 2

    # Car body (Blue)
    draw.rectangle([460, 350, 700, 470], fill=(30, 80, 210))
    draw.polygon([(500, 350), (540, 290), (640, 290), (670, 350)], fill=(20, 60, 180)) # cabin
    draw.ellipse([490, 450, 540, 500], fill=(20, 20, 20)) # wheel 1
    draw.ellipse([620, 450, 670, 500], fill=(20, 20, 20)) # wheel 2

    # Person 1
    draw.ellipse([390, 280, 420, 310], fill=(240, 200, 170)) # head
    draw.rectangle([395, 310, 415, 390], fill=(40, 160, 80)) # torso
    draw.line([(400, 390), (395, 470)], fill=(30, 30, 30), width=6) # leg 1
    draw.line([(410, 390), (415, 470)], fill=(30, 30, 30), width=6) # leg 2

    # Person 2
    draw.ellipse([40, 290, 70, 320], fill=(230, 190, 160)) # head
    draw.rectangle([45, 320, 65, 400], fill=(220, 120, 40)) # torso
    draw.line([(50, 400), (45, 470)], fill=(20, 20, 60), width=6) # leg 1
    draw.line([(60, 400), (65, 470)], fill=(20, 20, 60), width=6) # leg 2

    pil_img.save(save_path, "JPEG", quality=95)


def create_office_desk_scene(save_path: Path):
    # 800x600 office desktop
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    # Wall
    img[0:220, :] = [240, 240, 245]
    # Desk surface (Wood finish in BGR)
    img[220:600, :] = [140, 180, 210]

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Laptop
    draw.rectangle([260, 240, 540, 420], fill=(50, 50, 55)) # screen
    draw.rectangle([275, 255, 525, 405], fill=(30, 120, 200)) # display glow
    draw.polygon([(220, 470), (260, 420), (540, 420), (580, 470)], fill=(120, 120, 125)) # keyboard base

    # Coffee Cup
    draw.ellipse([140, 340, 200, 400], fill=(245, 245, 245)) # cup rim
    draw.ellipse([150, 350, 190, 390], fill=(80, 50, 30)) # coffee

    # Bottle
    draw.rectangle([640, 260, 700, 460], fill=(30, 180, 190))
    draw.rectangle([655, 230, 685, 260], fill=(200, 200, 200))

    # Cell phone
    draw.rectangle([130, 440, 210, 540], fill=(20, 20, 20))
    draw.rectangle([135, 445, 205, 535], fill=(70, 70, 80))

    pil_img.save(save_path, "JPEG", quality=95)


def create_living_room_scene(save_path: Path):
    # 800x600 living room
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[0:300, :] = [210, 220, 225] # wall
    img[300:600, :] = [180, 160, 140] # rug

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Couch / Sofa
    draw.rectangle([100, 250, 500, 440], fill=(70, 100, 140)) # backrest
    draw.rectangle([80, 320, 120, 430], fill=(60, 90, 130)) # left arm
    draw.rectangle([480, 320, 520, 430], fill=(60, 90, 130)) # right arm
    draw.rectangle([120, 350, 480, 460], fill=(90, 120, 160)) # cushions

    # Dog / Cat on rug
    draw.ellipse([570, 400, 660, 480], fill=(160, 100, 50)) # body
    draw.ellipse([640, 370, 690, 420], fill=(160, 100, 50)) # head
    draw.polygon([(650, 370), (660, 340), (670, 370)], fill=(130, 80, 40)) # ear 1
    draw.polygon([(675, 370), (685, 340), (695, 370)], fill=(130, 80, 40)) # ear 2

    # Potted Plant (vase/potted plant)
    draw.rectangle([680, 220, 740, 320], fill=(180, 90, 50))
    draw.ellipse([650, 120, 770, 230], fill=(40, 140, 50))

    pil_img.save(save_path, "JPEG", quality=95)


def generate_all_samples():
    img1 = SAMPLES_DIR / "sample_street_traffic.jpg"
    img2 = SAMPLES_DIR / "sample_office_desk.jpg"
    img3 = SAMPLES_DIR / "sample_living_room.jpg"

    create_street_scene(img1)
    create_office_desk_scene(img2)
    create_living_room_scene(img3)

    print(f"Generated sample benchmark images at: {SAMPLES_DIR}")

    # Ground truth annotations mapping for evaluation
    gt = {
        "sample_street_traffic.jpg": [
            {"label": "car", "bbox": [120, 300, 360, 510]},
            {"label": "car", "bbox": [460, 290, 700, 500]},
            {"label": "person", "bbox": [390, 280, 420, 470]},
            {"label": "person", "bbox": [40, 290, 70, 470]},
        ],
        "sample_office_desk.jpg": [
            {"label": "laptop", "bbox": [220, 240, 580, 470]},
            {"label": "cup", "bbox": [140, 340, 200, 400]},
            {"label": "bottle", "bbox": [640, 230, 700, 460]},
            {"label": "cell phone", "bbox": [130, 440, 210, 540]},
        ],
        "sample_living_room.jpg": [
            {"label": "couch", "bbox": [80, 250, 520, 460]},
            {"label": "dog", "bbox": [570, 340, 695, 480]},
            {"label": "potted plant", "bbox": [650, 120, 770, 320]},
        ],
    }

    gt_file = DATA_DIR / "ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)
    print(f"Generated ground truth annotations at: {gt_file}")


if __name__ == "__main__":
    generate_all_samples()
