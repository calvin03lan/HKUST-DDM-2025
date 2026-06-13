import json
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
import os

# ==================== Configuration ====================
DATA_DIR = "./raw_data/"
CATEGORIES = [
    "circle", "sun", "donut", "wheel", "clock",
    "umbrella", "lollipop", "hammer", "toothbrush", "spoon",
    "apple", "pear", "banana", "strawberry",
    "cat", "dog", "bear", "mouse", "rabbit"
]

MAX_STROKE_POINTS = 200
MIN_STROKE_POINTS = 10
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TRUNC_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0]

MAX_SAMPLES_PER_CAT = 5000


# ==================== 1. Parse NDJSON files ====================
def parse_ndjson(file_path, max_samples=None):
    """
    Parse a Quick Draw simplified .ndjson file.
    Returns a list of drawings, each as [[x,y], [x,y], ...].
    """
    drawings = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                drawing = data['drawing']
                points = []
                for stroke in drawing:
                    xs, ys = stroke[0], stroke[1]
                    for x, y in zip(xs, ys):
                        points.append([x, y])
                if MIN_STROKE_POINTS <= len(points) <= MAX_STROKE_POINTS:
                    drawings.append(points)
            except:
                continue
    return drawings


# ==================== 2. Convert to delta (dx, dy, pen_lift) ====================
def points_to_delta(points):
    """
    Convert absolute coordinate sequence to relative offsets.
    Input:  [[x0,y0], [x1,y1], ...]
    Output: [[dx0, dy0, pen0], [dx1, dy1, pen1], ...]
            pen_lift: 1 = new stroke start, 0 = continuation
    """
    if len(points) < 2:
        return []

    result = []
    result.append([0, 0, 1])

    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        result.append([dx, dy, 0])

    return result


# ==================== 3. Truncate sequence to a given ratio ====================
def truncate_sequence(seq, ratio):
    """
    Truncate sequence, keeping only the first (ratio * len) points.
    """
    if ratio >= 1.0:
        return seq
    n = max(1, int(len(seq) * ratio))
    return seq[:n]


# ==================== 4. Padding ====================
def pad_sequence(seq, max_len, pad_value=0):
    """
    Pad / truncate a sequence to a fixed length.
    """
    if len(seq) >= max_len:
        return np.array(seq[:max_len])
    else:
        pad_len = max_len - len(seq)
        pad = np.full((pad_len, 3), pad_value)
        return np.vstack([seq, pad])


# ==================== 5. Main processing pipeline ====================
def main():
    print("Processing Quick Draw data...")

    all_data = []
    label_to_id = {cat: idx for idx, cat in enumerate(CATEGORIES)}

    for cat in CATEGORIES:
        file_path = os.path.join(DATA_DIR, f"{cat}.ndjson")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping.")
            continue

        print(f"Processing: {cat}...")
        drawings = parse_ndjson(file_path, MAX_SAMPLES_PER_CAT)
        print(f"  {len(drawings)} valid drawings loaded")

        for points in drawings:
            delta_seq = points_to_delta(points)
            if len(delta_seq) < 2:
                continue

            original_len = len(delta_seq)

            for ratio in TRUNC_RATIOS:
                truncated_seq = truncate_sequence(delta_seq, ratio)
                all_data.append({
                    'features': truncated_seq,
                    'label': label_to_id[cat],
                    'ratio': ratio,
                    'original_len': original_len,
                    'category': cat
                })

    print(f"\nTotal samples generated: {len(all_data)}")

    # ==================== 6. Determine unified padding length ====================
    max_seq_len = 0
    for item in all_data:
        max_seq_len = max(max_seq_len, len(item['features']))
    max_seq_len = min(max_seq_len, MAX_STROKE_POINTS)
    print(f"Unified sequence length: {max_seq_len}")

    # ==================== 7. Pad and convert to arrays ====================
    X = []
    y = []
    ratios = []
    for item in all_data:
        padded = pad_sequence(item['features'], max_seq_len)
        X.append(padded)
        y.append(item['label'])
        ratios.append(item['ratio'])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    ratios = np.array(ratios, dtype=np.float32)

    # ==================== 8. Train / Val / Test split ====================
    X_train, X_temp, y_train, y_temp, r_train, r_temp = train_test_split(
        X, y, ratios, test_size=(1 - TRAIN_RATIO), random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, r_val, r_test = train_test_split(
        X_temp, y_temp, r_temp, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO),
        random_state=42, stratify=y_temp
    )

    print(f"\nDataset split complete:")
    print(f"  Train: {X_train.shape}, label distribution: {np.bincount(y_train)}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # ==================== 9. Save to .npz ====================
    np.savez_compressed(
        'quickdraw_processed.npz',
        X_train=X_train, y_train=y_train, ratios_train=r_train,
        X_val=X_val, y_val=y_val, ratios_val=r_val,
        X_test=X_test, y_test=y_test, ratios_test=r_test,
        max_seq_len=max_seq_len,
        categories=np.array(CATEGORIES),
        trunc_ratios=np.array(TRUNC_RATIOS)
    )

    print("\nData saved to quickdraw_processed.npz")

    # ==================== 10. Summary statistics ====================
    print("\n=== Data Summary ===")
    for ratio in TRUNC_RATIOS:
        count = np.sum(ratios == ratio)
        print(f"Truncation ratio {int(ratio*100)}%: {count} samples")

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    main()
