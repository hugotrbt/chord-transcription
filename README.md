## 🎸 Automatic Guitar Chord Transcription

This project implements and compares two approaches for automatic guitar chord transcription from audio:

A classical signal-processing baseline, using HPSS, chroma extraction, template matching, and smoothing.

A deep-learning model (in development), based on CNN/CRNN architectures trained on spectrogram features.

The system is evaluated on the Isophonics Beatles dataset, using frame-level accuracy and chord timelines for qualitative inspection.

## 🚀 Installation

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Make sure you have ffmpeg installed if using librosa on audio files.

## 📂 Project Structure

chord-transcription/
│
├── report/
│ └── intermediate/ # Intermediate report (PDF)
│
├── data/
│ ├── audio/ # Audio files (e.g., Let It Be, I'll Follow the Sun)
│ ├── annotations/ # .lab ground-truth labels from Isophonics
│ └── features/
│
├── src/
│ ├── classical/ # Classical baseline implementation
│ │ ├── preprocessing.py
│ │ ├── chord_templates.py
│ │ ├── template_matching.py
│ │ ├── smoothing.py
│ │ └── evaluation.py
│ │
│ ├── utils/ # Helpers (I/O, plotting, etc.)
│ └── deep/ # Deep learning model (upcoming)
│
└── scripts/
├── run_preprocessing.py
├── run_classical.py # Main script for classical pipeline
└── run_deep.py # Will be added later

## 🎼 Running the Classical Baseline

Run the full classical chord-recognition pipeline:

python scripts/run_classical.py

This script performs:

audio loading and HPSS

chroma CQT extraction

template matching

smoothing and segmentation

evaluation against ground truth

Output includes:

predicted chord timeline

framewise accuracy

visual figures (waveform, chroma, classical timeline)

## 🎧 Dataset

This project uses the Isophonics Beatles annotations, containing:

.wav audio files

.lab chord ground-truth files

Download from:
https://isophonics.net/content/reference-annotations-beatles

Place them in:

data/audio/
data/annotations/

## 🧠 Deep Learning (Upcoming)

The deep-learning system will follow a CNN/CRNN architecture trained on spectrograms derived from the same dataset.
A new script run_deep.py and training utilities will be added during the next phase of the project.

## 📊 Results (Classical Baseline)

Song Accuracy
Let It Be 69.97%
I'll Follow the Sun 20.52%

These results reflect the well-known behavior of template-matching systems: high performance on simple diatonic progressions, lower performance on harmonically complex material.

## 📌 Next Steps

Implement deep-learning chord recognition model (CNN/CRNN).

Evaluate and compare with the classical baseline.

Extend the report with quantitative and qualitative analyses.
