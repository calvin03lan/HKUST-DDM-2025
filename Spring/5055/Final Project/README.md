# Quick Draw Sketch Recognition

## Training Objective

Train a deep learning model to classify hand-drawn sketches into 19 categories using the Google Quick Draw dataset. The model takes a sequence of pen strokes (dx, dy, pen_lift) as input and predicts the category of the drawing. A key capability is **partial sketch recognition** — the model should correctly identify a drawing before it is finished, at 10%, 20%, 50%, and 100% completion.

### Categories (19)

| Group | Categories |
|-------|-----------|
| Shapes | circle, sun, donut, wheel, clock |
| Objects | umbrella, lollipop, hammer, toothbrush, spoon |
| Fruit | apple, pear, banana, strawberry |
| Animals | cat, dog, bear, mouse, rabbit |

## Project Structure

```
Final Project/
├── raw_data/                  # Raw .ndjson files downloaded from Google Cloud
├── data_processing.py         # Data preprocessing pipeline
├── quickdraw_processed.npz    # Processed and split dataset
├── download_raw.sh            # Script to download raw data
├── selected_categories.txt    # List of categories to download
└── venv/                      # Python 3.12 virtual environment
```

## Data Processing Pipeline

### 1. Raw Data Format

Each `.ndjson` file contains one JSON object per line. Each object has a `drawing` field — a list of strokes, where each stroke is `[[x1,x2,...], [y1,y2,...]]`.

### 2. Parsing (`parse_ndjson`)

- Read each line from the `.ndjson` file
- Extract stroke coordinates and flatten them into a list of `[x, y]` points
- Filter out drawings with fewer than 10 points or more than 200 points
- Cap at 5,000 samples per category

### 3. Delta Conversion (`points_to_delta`)

Convert absolute coordinates to relative offsets:

| Input | Output |
|-------|--------|
| `[[x0,y0], [x1,y1], [x2,y2], ...]` | `[[0,0,1], [dx1,dy1,0], [dx2,dy2,0], ...]` |

- **dx, dy**: Difference between consecutive points (stroke direction)
- **pen_lift**: 1 for the first point of a stroke, 0 otherwise

### 4. Truncation (`truncate_sequence`)

To enable partial-sketch recognition, each drawing is truncated to 4 completion levels:

| Ratio | Meaning |
|-------|---------|
| 10% | First 10% of strokes (early recognition) |
| 20% | First 20% of strokes |
| 50% | First 50% of strokes |
| 100% | Full drawing |

### 5. Padding (`pad_sequence`)

All sequences are padded/truncated to a unified length (199) with `[0, 0, 0]` so they can be batched for training.

### 6. Train / Val / Test Split

Stratified split by label to maintain class balance:

| Split | Proportion | Samples |
|-------|-----------|---------|
| Train | 70% | 264,661 |
| Val | 15% | 56,713 |
| Test | 15% | 56,714 |

### 7. Output

The processed dataset is saved as `quickdraw_processed.npz` with arrays:
- `X_train`, `X_val`, `X_test` — shape `(N, 199, 3)` float32
- `y_train`, `y_val`, `y_test` — shape `(N,)` int32 (class index)
- `ratios_train`, `ratios_val`, `ratios_test` — truncation ratio per sample

## Data Summary

| Metric | Value |
|--------|-------|
| Total samples | 378,088 |
| Categories | 19 |
| Sequence length | 199 |
| Features per step | 3 (dx, dy, pen_lift) |
| Truncation levels | 4 (10%, 20%, 50%, 100%) |

## Usage

```bash
# Activate environment
source venv/bin/activate

# Process the raw data
python data_processing.py

# The output is quickdraw_processed.npz
```
