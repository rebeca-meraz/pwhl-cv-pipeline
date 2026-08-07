# Why Jersey Numbers Are Hard to Read in Hockey: An End-to-End Computer Vision Pipeline for PWHL Broadcast Video

A from-scratch computer vision pipeline — people detection, filtering, and jersey number recognition — built and evaluated on PWHL broadcast footage. Findings are consistent with the challenges that motivate specialized systems like the one described in Vats, Walters, Fani, Clausi, and Zelek's *"Player Tracking and Identification in Ice Hockey"*.

## Motivation

Computer vision applied to sports has the potential to be a genuinely powerful tool, not just for reading a jersey number in isolation. Player tracking, in particular, opens the door to a much bigger chain of questions: tracking connects to fatigue, fatigue connects to performance, and performance connects to the kind of coaching decisions I got curious about in my NBA/NHL fixture congestion project — how much rest actually matters, and what a coaching staff could do about it. This project is a first, deliberately small step toward that chain: understanding whether the basic building blocks (detecting players, reading their numbers) even work reliably in a league where, to my knowledge, this has not yet been built.

## Main Contributions

- Built a complete pipeline from raw broadcast video to labeled, evaluated data — video acquisition, frame extraction, people detection (YOLO), automated quality filtering, manual review, and jersey number recognition (OCR).
- Diagnosed *why* detection and filtering succeeded or failed at each stage, rather than treating each step as a black box — comparing YOLO model sizes on real occlusion cases, and calibrating quality thresholds by visually inspecting samples instead of guessing.
- Found that OCR failure in this dataset reflects the same underlying challenge documented in "Player Tracking and Identification in Ice Hockey" (Vats, Walters, Fani, Clausi, Zelek) — independent, quantified confirmation of that problem in a league it had, to my knowledge, never been tested on.
- Identified the most strongly supported explanation for OCR failures after investigating multiple plausible causes: not blur alone, and not font style, but a compounding combination of residual image quality and player orientation relative to the camera.

## Methodology

| Stage | What happened |
|---|---|
| Video acquisition | Downloaded PWHL broadcast clips via `yt-dlp`; re-acquired in 1080p after discovering YouTube's format restrictions |
| Frame extraction | `ffmpeg`, sampled at 0.5–1 fps across ~8 minutes of footage |
| Detection | Pre-trained YOLOv8 (nano, then medium) — medium model reduced occlusion errors |
| Filtering | Sharpness (Laplacian variance ≥ 80) and crop size (≥ 140px height), both thresholds calibrated by visual inspection across the distribution |
| Manual review | ~1,800 detections reviewed by hand via interactive widgets; 288 crops retained as usable |
| Labeling | All 288 crops hand-labeled with ground-truth jersey numbers |
| Recognition | Evaluated against EasyOCR, a general-purpose pre-trained OCR model |

## Repo Structure

```
├── pwhl_cv_pipeline.ipynb
└── README.md
```

## Limitations

- **9.7% exact-match OCR accuracy**, with no detection at all in 80.9% of cases — evaluated against a single general-purpose OCR engine (EasyOCR); a second engine was not tested for comparison.
- **Small, unevenly distributed label set** (288 examples across 26 jersey numbers, ranging from 1 to 44 examples per number) — enough to evaluate an existing model, not to train a custom one.
- **Single game, single broadcast source** — findings may not generalize to other camera setups or leagues.
- **The orientation finding (79/80 non-frontal in a follow-up n=80 sample) is drawn from an already-biased subset** — crops the author had already hand-selected as "number visible" — so it describes a pattern within this dataset, not a general claim about hockey broadcasts.
- No team classification, player role separation (skater vs. referee), or multi-frame tracking was built — this project covers detection, filtering, and single-frame number reading only, a narrower scope than Vats et al.'s full pipeline.

## Personal Reflection

This project taught me as much about process as about computer vision itself. I assumed the biggest bottleneck would be the modeling stages — tuning YOLO, choosing an OCR engine — but manual review and labeling ended up taking far more time and iteration than either. Going through ~1,800 detections by hand made it clear how much of a small-dataset CV pipeline is actually human judgment, not model architecture.

The OCR results themselves also corrected an assumption I walked in with. I expected image quality to be the deciding factor, so the crops that were sharp and fully frontal — the “easy” cases — were reassuring at first: they read correctly, and I figured the harder cases would just need better resolution. What actually broke the model was angle, independent of quality: a crisp, well-lit crop with the player turned even slightly off-frontal failed just as often as a blurry one. That distinction — quality and orientation as separate failure modes, not one problem — is exactly the kind of groundwork that has to hold up before player tracking can say anything useful about fatigue, workload, or the coaching decisions built on top of it.

## Tools

Python, YOLOv8 (Ultralytics), EasyOCR, OpenCV, pandas, Matplotlib, ipywidgets — built in Jupyter, with video acquired via `yt-dlp` and `ffmpeg`.
