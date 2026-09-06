"""
stage4_extract_landmarks.py  (updated for MediaPipe 1.0.x Tasks API)

Reads the real ISL-CSLRT dataset (via data/dataset_index.json), runs the
MediaPipe HandLandmarker on each word's frame images, and saves the
extracted landmark numbers to disk as .npy files — one file per word.

WHERE TO PUT THIS FILE:
    cv-module/models/stage4_extract_landmarks.py

BEFORE RUNNING — one-time setup:
    Download this file and place it at cv-module/models/hand_landmarker.task
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

HOW TO RUN (from inside cv-module/, with venv activated):
    python models/stage4_extract_landmarks.py

WHAT IT PRODUCES:
    cv-module/models/landmarks_data/<WORD>.npy   (one file per word)
    cv-module/models/landmarks_data/summary.json (how many images worked/failed)
"""

import json
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------
# CONFIG — adjust these paths only if your folders differ
# ---------------------------------------------------------------
DATASET_INDEX_PATH = "../data/dataset_index.json"      # relative to cv-module/
DATASET_ROOT = "../data/ISL_CSLRT_Corpus"              # where the actual images live
OUTPUT_DIR = "models/landmarks_data"
MODEL_PATH = "models/hand_landmarker.task"             # downloaded model file

# Stage 5: now processing every word in the dataset, not just a small sample.
# Set to None to auto-process ALL words found in dataset_index.json.
WORDS_TO_PROCESS_FIRST = None


def build_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=mp_vision.RunningMode.IMAGE,  # we're processing separate photos, not video
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks_from_image(detector, image_path):
    """
    Runs the HandLandmarker on one image, returns a flat list of 63 numbers
    (21 landmark points x 3 coordinates) for the FIRST detected hand,
    or None if no hand was found / file couldn't be read.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None  # file missing or unreadable

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None  # no hand detected in this image

    first_hand = result.hand_landmarks[0]  # list of 21 NormalizedLandmark
    coords = []
    for point in first_hand:
        coords.extend([point.x, point.y, point.z])
    return coords  # length 63 list


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(DATASET_INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)

    word_entries = {w["gloss"]: w for w in index["word_level_signs"]}

    words_to_process = WORDS_TO_PROCESS_FIRST
    if words_to_process is None:
        words_to_process = list(word_entries.keys())

    print(f"Processing {len(words_to_process)} words total...\n")

    detector = build_detector()
    summary = {}

    for i, word in enumerate(words_to_process, start=1):
        if word not in word_entries:
            print(f"[skip] '{word}' not found in dataset_index.json")
            continue

        frame_paths = word_entries[word]["frame_paths"]
        all_landmarks = []
        failed = 0

        for rel_path in frame_paths:
            full_path = os.path.join(os.path.dirname(DATASET_ROOT), rel_path)
            coords = extract_landmarks_from_image(detector, full_path)

            if coords is None:
                failed += 1
                continue
            all_landmarks.append(coords)

        if all_landmarks:
            array = np.array(all_landmarks)  # shape: (num_success, 63)
            np.save(os.path.join(OUTPUT_DIR, f"{word}.npy"), array)

        summary[word] = {
            "total_images": len(frame_paths),
            "successful": len(all_landmarks),
            "failed_no_hand_or_missing": failed,
        }
        print(f"[{i}/{len(words_to_process)}] {word}: {len(all_landmarks)}/{len(frame_paths)} images processed successfully")

    detector.close()

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Summary saved to", os.path.join(OUTPUT_DIR, "summary.json"))


if __name__ == "__main__":
    main()