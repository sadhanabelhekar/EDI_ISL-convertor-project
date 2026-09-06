# Stage 6 — Baseline Classifier Results

**Date:** Week 4-5 (CV module)
**Owner:** Person 1 (CV / Sign Recognition Lead)

## Setup
- Dataset: ISL-CSLRT Corpus, word-level static frames
- Landmark extraction: MediaPipe HandLandmarker (Tasks API), 21 points x 3 coords = 63 features per image
- Classifier: Random Forest (100 trees, `class_weight="balanced"`)
- Words included: 23 (all words with >= 8 successfully extracted samples)
- Words excluded: 91 (fewer than 8 samples — logged, not discarded; candidates for future data augmentation)
- Train/test split: 80% / 20%, stratified by word

## Headline Result
**51.6% accuracy** on unseen test data, across 23-way classification.

For context: random guessing across 23 words would score ~4.3%. This result is
**~12x better than chance**, confirming the landmark features carry real,
learnable signal even with a simple baseline model and limited samples per word.

## Per-word findings

**Perfect recognition (100% precision & recall):**
CRYING, SOME ONE, THANK

**Zero recognition (model never predicted correctly):**
COLLEGE_SCHOOL, DO, GO, HOW, REALLY, ROOM, TELL

**Key insight — why some signs succeed and others fail:**
Signs with a visually distinctive, static hand shape (e.g. CRYING, THANK) are
reliably recognized even from single still frames. Signs involving motion or a
pointing gesture (notably "YOU") perform worse than their sample count would
predict — recall dropped from 95% (5-word test) to 43% (23-word test) despite
having the most training data (104 samples). This suggests the 104 "YOU" frames
capture multiple different moments of a moving gesture (start/mid/end of a
point), effectively adding label noise rather than clean repeated examples.

**Implication:** static single-frame recognition has a real ceiling for
motion-based signs. This directly motivates Stage 8 (sequence/video-based
recognition), which is necessary for the project's core goal regardless of
how much the static baseline is tuned further.

## Class imbalance
Sample counts ranged from 104 (YOU) down to 1 (BEAUTIFUL). `class_weight="balanced"`
was applied to reduce (not eliminate) bias toward high-sample words. Words below
the 8-sample threshold were excluded from this run rather than forced into
training, since fewer than 8 examples is not enough for a model to learn a
reliable pattern.

## Threshold Sensitivity Test (MIN_SAMPLES_PER_WORD)

To check whether stricter data-quality filtering would meaningfully improve
results, the classifier was re-trained with a higher minimum sample threshold.

| Threshold | Words included | Accuracy | Macro avg recall |
|---|---|---|---|
| 8  | 23 | 51.6% | 0.43 |
| 12 | 16 | 54.2% | 0.39 |

**Finding:** raising the threshold gave a marginally higher overall accuracy,
but macro avg recall (which treats all words equally, regardless of sample
count) slightly *decreased*, and specific words that scored well at threshold 8
(e.g. CRYING, WHAT) dropped to 0% at threshold 12. This shows that removing
words shifts the model's confusion patterns rather than uniformly improving
performance — the bottleneck is not "too many competing words," it is that
even the best-represented words lack enough *varied* examples, compounded by
the motion-based sign limitation noted below.

**Decision:** kept `MIN_SAMPLES_PER_WORD = 8` (23 words) as the reference
result going forward, since it covers a larger, more representative vocabulary
without a meaningful accuracy trade-off, and the threshold-12 experiment gave
no consistent evidence that stricter filtering is the right lever for
improvement.

**Conclusion of static-model tuning:** two independent runs (thresholds 8 and
12) show the same underlying weakness — motion-based/pointing signs (e.g. YOU)
are inherently hard for single-frame recognition, regardless of vocabulary
size or sample filtering. Further tuning of the static classifier has
diminishing returns; the project moves to Stage 8 (sequence/video-based
recognition) next.

## Next Steps
1. **Stage 8:** extract landmark **sequences** from full sentence videos (not
   single frames) using MediaPipe's video mode, preserving frame order so
   motion is captured — directly addressing the YOU/motion limitation above.
2. Train an LSTM-based model on these sequences instead of a Random Forest,
   since sequence order matters for motion-based signs.
3. Consider data augmentation (mirroring, slight rotation) for excluded
   low-sample words in a later iteration, as a separate improvement track.