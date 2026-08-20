"""
build_index.py

Regenerates the FULL dataset path index (word/sentence -> actual file paths)
from the original ISL-CSLRT source files. This file is intentionally NOT
committed to git — run this locally whenever you need the full index.

Usage:
    python build_index.py

Requires the 4 source files to sit in the same folder as this script
(or edit SOURCE_DIR below), and the actual ISL_CSLRT_Corpus dataset folder
to exist locally so the paths resolve.

Output:
    dataset_index.json  (gitignored — large, machine-specific paths)
"""

import pandas as pd
import json
import os

SOURCE_DIR = "."  # folder containing the 4 xlsx/csv files
OUTPUT_FILE = "dataset_index.json"


def norm(p):
    return str(p).strip().replace("\\", "/")


def main():
    word_df = pd.read_excel(os.path.join(SOURCE_DIR, "ISL_CSLRT_Corpus_word_details.xlsx"))
    gloss_df = pd.read_csv(os.path.join(SOURCE_DIR, "ISL_Corpus_sign_glosses.csv"))
    video_df = pd.read_excel(os.path.join(SOURCE_DIR, "ISL_CSLRT_Corpus_details.xlsx"))
    frame_df = pd.read_excel(os.path.join(SOURCE_DIR, "ISL_CSLRT_Corpus_frame_details.xlsx"))

    # Word-level: group frame paths per word
    word_groups = word_df.groupby("Word")["Frames path"].apply(lambda x: [norm(p) for p in x]).to_dict()
    word_counts_sorted = word_df["Word"].value_counts()

    word_level_signs = []
    for i, (word, count) in enumerate(word_counts_sorted.items(), start=1):
        word_level_signs.append({
            "id": f"word_{i:03d}",
            "gloss": word.strip(),
            "frame_samples": int(count),
            "frame_paths": word_groups[word],
            "level": "word",
        })

    # Sentence-level: group video + frame paths per sentence
    video_df["_key"] = video_df["Sentences"].str.strip()
    frame_df["_key"] = frame_df["Sentence"].str.strip()
    video_groups = video_df.groupby("_key")["File location"].apply(lambda x: [norm(p) for p in x]).to_dict()
    frame_groups = frame_df.groupby("_key")["Frames path"].apply(lambda x: [norm(p) for p in x]).to_dict()

    sentence_level_data = []
    for i, row in gloss_df.iterrows():
        sentence = str(row["Sentence"]).strip()
        gloss_seq = str(row["SIGN GLOSSES"]).strip().split()
        v_paths = video_groups.get(sentence, [])
        f_paths = frame_groups.get(sentence, [])
        sentence_level_data.append({
            "id": f"sent_{i+1:03d}",
            "sentence": sentence,
            "gloss_sequence": gloss_seq,
            "video_samples": len(v_paths),
            "video_paths": v_paths,
            "frame_samples": len(f_paths),
            "frame_paths": f_paths,
            "level": "sentence",
        })

    index = {
        "version": "1.0",
        "dataset": "ISL-CSLRT Corpus",
        "dataset_root": "ISL_CSLRT_Corpus/",
        "word_level_signs": word_level_signs,
        "sentence_level_data": sentence_level_data,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Wrote {OUTPUT_FILE}")
    print(f"  words: {len(word_level_signs)}, sentences: {len(sentence_level_data)}")


if __name__ == "__main__":
    main()