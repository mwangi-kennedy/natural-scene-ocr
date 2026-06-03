#  Optic Nerve — Image-to-Text Recognition System

> *Teaching machines to read the world.*

A production-quality, end-to-end pipeline that detects and extracts text from natural scene images
using deep learning OCR, systematic image pre-processing, confidence filtering, and
Non-Maximum Suppression.

---

##  Overview

This project builds a complete **Image-to-Text Recognition System**, structured around
the analogy of the human optic nerve: raw visual input enters the pipeline, and structured,
readable text comes out the other end.

The system is split into two parts:

| Part | Focus | Key Technique |
|------|-------|---------------|
| Part 1 | Image conditioning + OCR | Adaptive thresholding · EasyOCR (CRAFT + CRNN) |
| Part 2 | Detection refinement | Confidence filtering · Bounding boxes · NMS |

---

##  Pipeline Stages
Image Input
│
▼
[1] Grayscale Conversion
│
▼
[2] Gaussian Blur (noise reduction)
│
▼
[3] Adaptive Thresholding → Binary Image
│
▼
[4] Morphological Dilation
│
▼
[5] EasyOCR Inference (CRAFT + CRNN)
│
▼
[6] Confidence Filter  (≥ 0.60)
│
▼
[7] Non-Maximum Suppression  (IoU ≤ 0.30)
│
▼
Annotated Output + Extracted Text

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/mwangi-kennedy/optic-nerve.git
cd optic-nerve

# Install dependencies
pip install easyocr opencv-python-headless matplotlib pillow requests numpy
```

Or open directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

##  Usage

```python
import easyocr
from pipeline import image_to_text_pipeline, visualise_results

reader = easyocr.Reader(['en'], gpu=False)

results, image = image_to_text_pipeline(
    image_source="https://your-image-url.com/image.jpg",
    reader=reader,
    confidence_threshold=0.60,
    nms_iou_threshold=0.30,
    is_url=True
)

extracted_text = visualise_results(image, results)
```

---

##  Project Structure
optic-nerve/
│
├── optic_nerve.ipynb   # Main Colab notebook (concepts + code)
├── README.md                    # This file
├── docs/
│   └── project4_documentation.pdf  # Full technical documentation
└── assets/
└── pipeline_diagram.png

---

##  Key Concepts Covered

- **Adaptive Thresholding** — local pixel-level binarization for uneven lighting
- **Transfer Learning** — leveraging EasyOCR's pre-trained CRAFT and CRNN models
- **Confidence Scores** — Softmax-derived uncertainty estimates for each detection
- **Bounding Box Anatomy** — corner coordinates, width, height, area
- **Intersection over Union (IoU)** — spatial overlap metric for duplicate detection
- **Non-Maximum Suppression** — greedy algorithm for selecting the best box per region
- **Precision–Recall Trade-off** — tunable threshold sensitivity analysis

---

## Results

The pipeline produces annotated output images with bounding boxes drawn around detected
text regions, colour-coded by detection, with confidence scores displayed inline.
A sensitivity analysis chart reveals how the confidence threshold affects the
precision–recall balance across the detection spectrum.

---

##  Documentation

Full technical documentation is available in [`docs/project4_documentation.pdf`](docs/project4_documentation.pdf),
covering:
- System architecture (9-stage pipeline diagram)
- Mathematical foundations (IoU, Softmax, adaptive thresholding)
- Algorithm walkthroughs (NMS step-by-step)
- Performance analysis and observations
- Challenges and solutions
- Academic references

---

##  Technology Stack

| Library | Purpose |
|---------|---------|
| `easyocr` | Deep learning OCR (CRAFT + CRNN) |
| `opencv-python` | Image pre-processing |
| `numpy` | Numerical array operations |
| `matplotlib` | Visualisation |
| `pillow` | Image loading and conversion |
| `requests` | Fetching images from URLs |

---

##  License

MIT License — see [LICENSE](LICENSE) for details.

---
