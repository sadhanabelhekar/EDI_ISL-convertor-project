"""
stage8_extract_sequences.py

Stage 8a - extracts landmark SEQUENCES from the sentence-level dataset.
Uses BOTH hands: each frame = 126 numbers (left hand 63 + right hand 63).
Each take is reduced to MAX_SEQ_LEN frames spread evenly over the whole video.

HOW TO RUN (from inside cv-module/, venv activated):
    python models/stage8_extract_sequences.py

WHAT IT PRODUCES:
    cv-module/models/sequence_data/<sent_id>_<take>.npy   shape: (20, 126)
    cv-module/models/sequence_data/labels.json
"""

import json
import os
import re
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DATASET_INDEX_PATH = "../data/dataset_index.json"
DATASET_ROOT = "../data/ISL_CSLRT_Corpus"
OUTPUT_DIR = "models/sequence_data"
MODEL_PATH = "models/hand_landmarker.task"

MAX_SEQ_LEN = 20
NUM_FEATURES = 126  # 2 hands x 21 landmarks x 3 coords


def build_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks(detector, image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return [0.0] * NUM_FEATURES
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return [0.0] * NUM_FEATURES

    # slot 0 = "Left" hand, slot 1 = "Right" hand
    slots = [None, None]
    for hand, handed in zip(result.hand_landmarks, result.handedness):
        wanted = 0 if handed[0].category_name == "Left" else 1
        if slots[wanted] is not None:
            wanted = 1 - wanted
        if slots[wanted] is None:
            slots[wanted] = hand

    coords = []
    for hand in slots:
        if hand is None:
            coords.extend([0.0] * 63)
        else:
            wrist = hand[0]
            for p in hand:
                coords.extend([p.x - wrist.x, p.y - wrist.y, p.z - wrist.z])
    return coords


def group_frames_by_take(frame_paths):
    takes = {}
    for path in frame_paths:
        parts = path.replace("\\", "/").split("/")
        take_id = parts[-2]
        filename = parts[-1]
        match = re.search(r"(\d+)\.\w+$", filename)
        frame_num = int(match.group(1)) if match else 0
        takes.setdefault(take_id, []).append((frame_num, path))

    for take_id in takes:
        takes[take_id].sort(key=lambda x: x[0])
        takes[take_id] = [p for _, p in takes[take_id]]

    return takes


def pad_or_truncate(sequence, max_len):
    seq = np.array(sequence)
    if len(seq) > max_len:
        idx = np.linspace(0, len(seq) - 1, max_len).round().astype(int)
        return seq[idx]
    if len(seq) == max_len:
        return seq
    pad_rows = max_len - len(seq)
    return np.vstack([seq, np.zeros((pad_rows, NUM_FEATURES))])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(DATASET_INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)

    detector = build_detector()
    labels = {}

    sentences = index["sentence_level_data"]
    print(f"Processing {len(sentences)} sentences...\n")

    for i, sent in enumerate(sentences, start=1):
        sent_id = sent["id"]
        sentence_text = sent["sentence"]
        frame_paths = sent["frame_paths"]

        if not frame_paths:
            print(f"[{i}/{len(sentences)}] {sent_id} '{sentence_text}': no frames, skipping")
            continue

        takes = group_frames_by_take(frame_paths)
        saved_takes = 0

        for take_id, ordered_paths in takes.items():
            sequence = []
            for rel_path in ordered_paths:
                full_path = os.path.join(os.path.dirname(DATASET_ROOT), rel_path)
                sequence.append(extract_landmarks(detector, full_path))

            padded = pad_or_truncate(sequence, MAX_SEQ_LEN)
            out_name = f"{sent_id}_{take_id}.npy"
            np.save(os.path.join(OUTPUT_DIR, out_name), padded)
            labels[out_name] = sentence_text
            saved_takes += 1

        print(f"[{i}/{len(sentences)}] {sent_id} '{sentence_text}': {saved_takes} takes saved")

    detector.close()

    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(labels)} sequence files saved. Labels saved to labels.json")


if __name__ == "__main__":
    main()