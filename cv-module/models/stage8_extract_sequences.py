"""
stage8_extract_sequences.py

Stage 8a — extracts landmark SEQUENCES (not single frames) from your
sentence-level dataset, preserving frame order within each video take.
This is what captures motion, unlike Stage 4's single-frame extraction.

WHERE TO PUT THIS FILE:
    cv-module/models/stage8_extract_sequences.py

BEFORE RUNNING:
    Same hand_landmarker.task file from Stage 4 must be in cv-module/models/

HOW TO RUN (from inside cv-module/, with venv activated):
    python models/stage8_extract_sequences.py

WHAT IT PRODUCES:
    cv-module/models/sequence_data/<sent_id>_<take>.npy   (one per video take)
        shape: (MAX_SEQ_LEN, 63)  -- zero-padded/truncated
    cv-module/models/sequence_data/labels.json
        maps each saved file to its sentence text (the classification label)
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

MAX_SEQ_LEN = 20  # frames per sequence; shorter takes are zero-padded, longer ones truncated


def build_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks(detector, image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return [0.0] * 63  # no hand found -- zero vector, not a dropped frame,
                            # so sequence timing/order stays intact
    first_hand = result.hand_landmarks[0]
    coords = []
    for point in first_hand:
        coords.extend([point.x, point.y, point.z])
    return coords


def group_frames_by_take(frame_paths):
    """
    Frame paths look like:
      ISL_CSLRT_Corpus/Frames_Sentence_Level/are you free today/1/are you free today 01.jpg
    The folder right before the filename (here "1") is the take number.
    The trailing number in the filename is the frame order within that take.
    """
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
    if len(seq) >= max_len:
        return seq[:max_len]
    pad_rows = max_len - len(seq)
    padding = np.zeros((pad_rows, 63))
    return np.vstack([seq, padding])


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
                coords = extract_landmarks(detector, full_path)
                sequence.append(coords)

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