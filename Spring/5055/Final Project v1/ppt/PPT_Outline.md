# PPT Outline — Real-Time Partial Sketch Recognition

---

## Slide 1 — Title Page

**Real-Time Partial Sketch Recognition with Deep Learning**
*1D-ResNet + BiGRU + Attention Pipeline with Interactive Web Frontend*

- Author / Course: DDM 5055 — Final Project
- Date: Spring 2026
- Tagline: *"Recognizing drawings before they are finished"*

---

## Slide 2 — Table of Contents

1. Project Background & Motivation
2. Technical Approach — Data, Architecture & Training Strategy
3. Training Results & Zero-Shot Generalization
4. Interactive Frontend — From Model to Live Demo
5. Limitations & Future Work

---

## Slide 3 — Project Background & Motivation

### Why Quick Draw?
- Google Quick Draw: 50M+ hand-drawn sketches across 345 categories, collected from a global online game
- Each sketch is a sequence of pen strokes — naturally suited for **sequence modeling**
- Unlike image-based sketch recognition (CNN on rendered bitmap), this project works directly on **stroke sequences**

### What Makes This Different?
| Conventional Sketch Recognition | This Project |
|---|---|
| Classify finished drawings | Recognize **partially-completed** sketches |
| Train on full sketches only | Train on ≥40% complete, **evaluate on 10%-100%** |
| Static offline evaluation | **Real-time interactive frontend** |
| Report single accuracy number | **Accuracy curve across completion ratios** |

### Core Challenge — Zero-Shot Partial Recognition
The model is trained **only** on sketches that are ≥40% complete (Heavyweight Specialist strategy), yet must correctly identify drawings at 10%, 20%, and 30% completion — a **zero-shot / out-of-distribution** evaluation measuring robustness under severe domain shift.

---

## Slide 4 — Technical Approach

### Data Pipeline
- **19 categories** selected across 4 groups:
  - Shapes: circle, sun, donut, wheel, clock
  - Objects: umbrella, lollipop, hammer, toothbrush, spoon
  - Fruit: apple, pear, banana, strawberry
  - Animals: cat, dog, bear, mouse, rabbit
- **Confusable pairs**: circle ⇔ donut ⇔ wheel (round shapes); bear ⇔ cat ⇔ mouse (animal silhouettes); lollipop ⇔ spoon (stick + round head)
- **5,000 samples per category**, each truncated at **9 completion ratios** (10%–100%), producing **378,088 total samples**
- Coordinate preprocessing: absolute `[x, y]` → delta `[dx, dy, pen_lift]` with unified padding to 199 steps

### Model Architecture
```
Input (199 × 3)
  ↓
[1D-ResNet Feature Extractor]  — 2-3 residual blocks with stride-2 downsampling (4× sequence reduction)
  ↓
[Bidirectional GRU]             — 2 layers, hidden=512, dropout=0.3, captures temporal context
  ↓
[Masked Multi-Head Attention]   — 8 heads, dynamic key_padding_mask ignores [0,0,0] padding
  ↓
[Deep Classification Head]      — Linear → GELU → Dropout → Linear (19 classes)
```

**Why this architecture?**
- 1D convolutions extract local geometric primitives (curves, corners, straight lines)
- Residual connections enable deeper feature extraction without vanishing gradients
- Bidirectional GRU captures stroke order and direction — critical for distinguishing circle vs. donut (similar shape, different stroke patterns)
- Masked attention ensures padding zeros are strictly ignored, handling variable-length partial sketches

### Training Strategy
- **Heavyweight Specialist**: Train exclusively on ratio ≥ 0.40 to force specialization on nearly-complete drawings
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts (T₀=5, T_mult=2) — aggressively escapes local minima
- **Regularization**: Label smoothing (0.1), gradient clipping (1.0), dropout (0.3)
- **Early stopping**: patience=10 on validation macro-F1
- **Training time**: ~70 min on RTX 4070 (46 epochs, batch_size=512)

---

## Slide 5 — Training Results

### Key Metrics (Best Model — Epoch 35)
| Metric | Train | Validation |
|--------|-------|-----------|
| Loss | 0.604 | 0.796 |
| Top-1 Accuracy | **99.88%** | **93.25%** |
| Top-3 Accuracy | 99.96% | 98.07% |
| Macro-F1 | 99.87% | **93.27%** |
| Parameters | — | **12.6M** |
| CPU Inference | — | **~3ms** |

### Zero-Shot Generalization *(5_accuracy_vs_truncation.png)*
| Completion | Top-1 Accuracy | Note |
|-----------|---------------|------|
| 10% | 8.7% | Zero-shot (vs. random = 5.3%) |
| 20% | 24.6% | Zero-shot |
| 30% | **55.9%** | Zero-shot — already >50% |
| 40% | **76.7%** | Training boundary |
| 100% | **98.9%** | Fully drawn |

The model demonstrates strong zero-shot generalization: at 30% completion (never seen during training), accuracy jumps to 55.9% — more than 10× random chance. At the training boundary (40%), accuracy reaches 76.7%, showing the Heavyweight Specialist strategy successfully transfers to earlier completion stages.

### Per-Class Performance *(5_per_class_accuracy.png)*
- **16 of 19 classes** achieve ≥97% accuracy at full completion
- Most challenging: circle (91.1%) — confusable with donut, wheel, sun
- No single class dominates predictions — prediction frequency is balanced (4.5%–6.1% per class)
- Four category groups (shapes, objects, fruit, animals) show comparable accuracy, confirming no group-level bias

### Training Convergence *(5_training_curves.png)*
- CosineAnnealingWarmRestarts (T₀=5) visible as periodic loss spikes followed by rapid recovery
- No significant overfitting gap — validation tracks training closely through epoch 46
- Best model saved at epoch 35 (val macro-F1 = 93.3%)

### Per-Class Performance *(5_per_class_accuracy.png)*
- **16 of 19 classes** achieve ≥97% accuracy at full completion
- Most challenging: circle (91.1%) — confusable with donut, wheel, sun
- No single class dominates predictions — prediction frequency is balanced (4.5%–6.1% per class)

### Training Convergence *(5_training_curves.png)*
- Smooth convergence with CosineAnnealingWarmRestarts
- No significant overfitting gap — validation tracks training closely
- Model: 12.6M parameters, inference < 5ms on CPU

---

## Slide 6 — Interactive Frontend

### Technology Stack
- **Backend**: FastAPI (Python) — loads PyTorch model, serves REST API
- **Frontend**: Single-page HTML + vanilla JS + CSS (zero build tools)
- **Model**: 12.6M parameters, CPU inference ~3ms per prediction

### Features
- **19-category selection** with grouped display (Shapes / Objects / Fruit / Animals)
- **Real-time live prediction** — Top-3 probabilities update every 100ms as user draws
- **Completion analysis** — after clicking "Finish", shows prediction at 10%, 20%, …, 100% of drawing progress with **green (correct) / red (incorrect)** color coding
- **Dark/light theme** toggle + **Chinese/English** language switching
- **Undo / Clear** drawing controls with full redraw support

### Engineering Challenges Solved

**1. Coordinate Scale Mismatch**
Canvas CSS pixels (e.g., 800×600) ≠ Quick Draw training space (256×256). Raw CSS deltas could reach 400+, far outside the training distribution (median |dx| ≈ 13).
→ *Solution: Normalize coordinates to 256×256 model space before computing deltas.*

**2. Smooth Mouse vs. Jittery Finger**
Browser mouse events at 60fps produce unnaturally smooth, dense point sequences. The model, trained on irregular touchscreen finger strokes, fails to recognize geometric perfection.
→ *Solution: RDP (Ramer-Douglas-Peucker) simplification with ε=2.0 removes redundant co-linear points while preserving shape-critical corners. Circle recognition improved from strawberry 56% → circle 60%.*

**3. Variable Canvas Position & Size**
Users draw at different positions and scales on the canvas, producing inconsistent absolute coordinates.
→ *Solution: Adaptive bounding-box scaling — compute the drawing's bounding box, then scale the larger dimension to 255 units. The drawing always fills the model's expected input space regardless of canvas size or drawing position.*

**4. Cross-Stroke Delta Continuity**
When the pen lifts and starts a new stroke, the spatial displacement between strokes carries meaningful information.
→ *Solution: Always compute deltas from the previous point's position, even across stroke boundaries. This matches the training data format and preserves inter-stroke spatial relationships.*

---

## Slide 7 — Limitations & Future Work

### 1. Domain Gap Between Human Drawing Styles
The model was trained exclusively on Quick Draw data (touchscreen, rapid sketching game). Users drawing on a desktop with a mouse produce smoother, slower strokes with different rhythm characteristics. Even with RDP preprocessing, synthetic geometric shapes often fail to be recognized.
- **Future work**: Fine-tune the model with self-collected data from the frontend (user drawings). Implement data augmentation that simulates various input devices (mouse, trackpad, stylus, finger). Use domain adaptation techniques to bridge the Quick Draw ↔ real-world gap.

### 2. No Stroke-Level Interpretability
The model outputs a single probability distribution over 19 classes, but provides no explanation of *which strokes* contributed most to the prediction. A user who draws a lollipop but gets "bear" cannot understand why.
- **Future work**: Implement attention visualization — highlight which segments of the stroke sequence received the highest attention weights. This would provide real-time visual feedback (e.g., color-coded strokes) showing which parts of the drawing drive the model's decision.

### 3. Fixed Sequence Length and Single-Scale Processing
All drawings are padded/truncated to exactly 199 timesteps. The 1D-ResNet stem uses fixed stride-2 downsampling, producing a single-scale representation. Very short drawings (< 20 points) lose most information after 4× downsampling. Very long drawings (> 199 points) are truncated, losing tail information.
- **Future work**: Adopt multi-scale feature extraction (e.g., feature pyramids) to handle diverse drawing lengths. Replace fixed padding with dynamic sequence handling (e.g., Transformer with positional encoding). Explore replacing the ResNet+GRU backbone with a pure Transformer architecture pretrained on larger stroke-sequence corpora.
