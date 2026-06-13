import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from train_quickdraw import ConvGRUClassifier

CATEGORIES = [
    "circle", "sun", "donut", "wheel", "clock",
    "umbrella", "lollipop", "hammer", "toothbrush", "spoon",
    "apple", "pear", "banana", "strawberry",
    "cat", "dog", "bear", "mouse", "rabbit",
]

CATEGORY_GROUPS = {
    "shapes": ["circle", "sun", "donut", "wheel", "clock"],
    "objects": ["umbrella", "lollipop", "hammer", "toothbrush", "spoon"],
    "fruit": ["apple", "pear", "banana", "strawberry"],
    "animals": ["cat", "dog", "bear", "mouse", "rabbit"],
}

MAX_SEQ_LEN = 199
TRUNC_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0]

device = torch.device("cpu")
model = ConvGRUClassifier(hidden_size=512, num_classes=19, dropout=0.3, gru_layers=2)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location="cpu", weights_only=True))
model.to(device)
model.eval()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    strokes: list


def strokes_to_tensor(strokes: list) -> tuple[torch.Tensor, torch.Tensor]:
    arr = np.array(strokes, dtype=np.float32)
    actual_len = min(arr.shape[0], MAX_SEQ_LEN)
    padded = np.zeros((MAX_SEQ_LEN, 3), dtype=np.float32)
    padded[:actual_len] = arr[:actual_len]
    x = torch.from_numpy(padded).unsqueeze(0).to(device)
    length = torch.tensor([actual_len], dtype=torch.long).to(device)
    return x, length


def run_inference(x: torch.Tensor, length: torch.Tensor) -> list[dict]:
    with torch.no_grad():
        logits = model(x, length)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    top3_values, top3_indices = torch.topk(probs, k=3)
    return [
        {
            "category": CATEGORIES[idx.item()],
            "index": idx.item(),
            "probability": round(val.item(), 4),
        }
        for idx, val in zip(top3_indices, top3_values)
    ]


@app.post("/api/predict")
def predict(req: PredictRequest):
    if len(req.strokes) < 2:
        return {"top3": [], "stroke_count": len(req.strokes)}
    x, length = strokes_to_tensor(req.strokes)
    top3 = run_inference(x, length)
    return {"top3": top3, "stroke_count": len(req.strokes)}


@app.post("/api/analyze")
def analyze(req: PredictRequest):
    strokes = req.strokes
    total = len(strokes)
    results = []
    for ratio in TRUNC_RATIOS:
        n = max(1, int(total * ratio))
        truncated = strokes[:n]
        x, length = strokes_to_tensor(truncated)
        top3 = run_inference(x, length)
        top1 = top3[0] if top3 else None
        results.append({
            "ratio": ratio,
            "ratio_label": f"{int(ratio * 100)}%",
            "points_used": n,
            "top1": top1,
            "top3": top3,
        })
    return {"results": results, "total_strokes": total}


@app.get("/api/categories")
def get_categories():
    return {
        "categories": CATEGORIES,
        "groups": CATEGORY_GROUPS,
    }


@app.get("/")
def root():
    with open("frontend/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
