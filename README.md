# 온디바이스 포토 큐레이션

> A fully on-device computer-vision CLI that surfaces photos by *feeling and context* —
> laughter, seasons, trips, themes — instead of scrolling a timeline.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-torch-EE4C2C?logo=pytorch&logoColor=white)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-412991?logo=openai&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-RetinaFace-00A98F)
![On-Device](https://img.shields.io/badge/On--Device-No%20Uploads-success)
![Privacy](https://img.shields.io/badge/Privacy-Read--Only%20Originals-blue)

## Overview

Tens of thousands of photos, and the only way to find *that one moment* is scrolling by date.
This is a personal project I built to fix that: a command-line pipeline that indexes a photo
library across **6 analysis dimensions** and lets me pull memories by **4 query axes** —
by emotion, by season, by trip, by theme, or any combination of them.

Everything runs **100% on-device**. No server uploads, no account required, and originals
are opened **read-only**. It runs **two ML models side by side** in roughly **704 lines of
Python**, and it's designed to survive interruption: an incremental cache checkpoints every
50 images, so a stopped run picks up where it left off.

## How It Works

The library is analyzed across six dimensions, then queried along four axes:

- **Laughter ranking** — Facial-expression analysis scores each photo by `happy%`, with a
  bonus for group shots so shared laughter rises to the top.
- **Mask quality gate** — CLIP zero-shot estimates the probability that a face is masked,
  then filters those photos out *before* trusting the expression model — defending against a
  known failure mode (see Technical Highlights).
- **Seasons** — Derived from the EXIF capture month, so spring/summer/autumn/winter views
  come straight from when the shutter actually fired.
- **Trips** — GPS coordinates are binned into a grid to estimate a "home range," photos
  outside a Haversine radius are flagged as travel, then merged by date into trips.
- **Themes** — CLIP zero-shot classifies each image into 8 categories, keeping only
  predictions above a softmax confidence threshold.
- **Multi-dimensional queries** — Any of the above can be combined into a single query
  (e.g. laughter + season + theme) to narrow down to a specific kind of memory.

## Tech Stack

- **Language:** Python 3.12
- **Facial expression:** DeepFace, with RetinaFace for face detection
- **Multimodal model:** CLIP (`openai/clip-vit-base-patch32`) via `transformers` + `torch`
- **Imaging:** Pillow + `pillow-heif` for HEIC support
- **Numerics & UX:** numpy, tqdm
- **Runtime:** fully on-device — no uploads, no account, read-only access to originals

## Technical Highlights

**Cross-model failure defense.** Facial-expression (FER) models are prone to misreading a
masked face as a smile. Instead of accepting that, I gate the expression model with CLIP — a
*different, multimodal* model — to estimate mask probability and reject those photos first.
It's a design that starts from knowing a model's limits and covers the blind spot with a
heterogeneous second opinion.

**One model, two zero-shot tasks.** A single CLIP model is reused zero-shot for *both* theme
classification and mask detection — no extra training, no second checkpoint to ship, just
well-chosen prompts driving two jobs.

**Real-world robustness.** Getting this to run on an actual library meant solving the
unglamorous parts: HEIC decoding, non-ASCII file paths, and pinning library
versions to keep the two-model stack reproducible. The incremental cache (every 50 images)
makes long runs interruption-tolerant.

**Ethical, private by design.** The system never claims to read someone's inner state. It
frames results as *"lots of laughter here,"* not *"this person was happy"* — a deliberate
choice to avoid asserting emotion. Combined with on-device processing, photos never
leave the machine.

## My Role

Solo, end-to-end personal project. I designed the pipeline architecture, chose and combined
the two-model stack, and built the cross-model gating strategy that defends the expression
model against its masked-face failure mode. I designed the 6 analysis dimensions and 4 query
axes, tuned the confidence thresholds and travel-detection heuristics, and did the
library-level troubleshooting (HEIC, path encoding, version pinning) plus the incremental
caching that makes it robust on real, messy photo collections.
