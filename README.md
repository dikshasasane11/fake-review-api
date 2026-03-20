# Fake Review Detection — FastAPI

REST API that classifies hotel reviews as **FAKE** or **GENUINE** using
Logistic Regression + TF-IDF (5,000 bigrams) + 9 handcrafted text signals.

---

## Project Structure

```
fake_review_api/
├── main.py               ← FastAPI app & routes
├── models.py             ← ML model (training + prediction)
├── schemas.py            ← Pydantic request/response schemas
├── requirements.txt      ← Python dependencies
└── tripadvisor_hotel_reviews.csv   ← dataset (place here)
```

---

## Setup & Run (VS Code)

### 1. Install dependencies
Open the VS Code terminal (`Ctrl+`` `) and run:
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
Copy `tripadvisor_hotel_reviews.csv` into the `fake_review_api/` folder.

### 3. Start the server
```bash
uvicorn main:app --reload
```

The API will be live at **http://127.0.0.1:8000**

`--reload` means the server restarts automatically when you edit any file.

---

## API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/`               | Health check                       |
| GET    | `/docs`           | Swagger UI (interactive docs)      |
| GET    | `/redoc`          | ReDoc documentation                |
| POST   | `/predict`        | Classify a single review           |
| POST   | `/predict/batch`  | Classify up to 50 reviews at once  |
| GET    | `/model/info`     | Model metadata & top features      |

---

## Example Requests

### Single prediction
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "AMAZING!!! BEST HOTEL EVER!!! PERFECT!!!", "rating": 5}'
```

**Response:**
```json
{
  "label": "FAKE",
  "is_fake": true,
  "confidence": 0.97,
  "signals": {
    "word_count": 7,
    "exclamation_count": 4,
    "unique_word_ratio": 0.86,
    ...
  },
  "latency_ms": 12.4
}
```

### Batch prediction
```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      {"text": "AMAZING!!! BEST EVER!!!", "rating": 5},
      {"text": "Stayed 3 nights. Clean room, friendly staff. Minor noise issue.", "rating": 4}
    ]
  }'
```

---

## Using Swagger UI (easiest)

1. Start the server
2. Open **http://127.0.0.1:8000/docs** in your browser
3. Click any endpoint → **Try it out** → fill in the fields → **Execute**

No curl or Postman needed!

---

## Model Performance

| Model               | Accuracy | Precision | Recall  | F1     |
|---------------------|----------|-----------|---------|--------|
| Logistic Regression | 100.00%  | 100.00%   | 100.00% | 100.00%|
| Random Forest       | 99.98%   | 100.00%   | 98.88%  | 99.44% |
| Linear SVM          | 99.95%   | 100.00%   | 97.75%  | 98.86% |
| Naive Bayes         | 98.83%   | 97.67%    | 47.19%  | 63.64% |
