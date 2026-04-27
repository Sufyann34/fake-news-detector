# 🔍 Fake News Detector

An end-to-end Machine Learning system that detects whether a news article is **fake or real**.

##  Live Demo
- **API:** https://fake-news-detector-381q.onrender.com
- **API Docs:** https://fake-news-detector-381q.onrender.com/docs

##  Accuracy
- **99.3%** on 8,980 test articles

##  How It Works
1. Text is cleaned (lowercased, URLs removed, stopwords filtered)
2. Converted to numbers using **TF-IDF** (50,000 features + bigrams)
3. Classified by a **Logistic Regression** model
4. Result returned via **FastAPI** REST API

## 📁 Project Structure
| File | Purpose |
|------|---------|
| `data.py` | Downloads & merges Kaggle dataset |
| `train.py` | Cleans data, trains & saves model |
| `api.py` | FastAPI REST API |
| `app.py` | Streamlit UI |
| `requirements.txt` | Dependencies |
| `start.sh` | Render deployment script |

## Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Start the API
```bash
uvicorn api:app --reload
```

### 4. Launch the UI
```bash
streamlit run app.py
```

## 🔌 API Usage
```bash
curl -X POST https://fake-news-detector-381q.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news headline here"}'
```

### Response
```json
{
  "label": "fake",
  "confidence": 0.94,
  "latency_ms": 0.84
}
```

##  Dataset
- **Source:** Kaggle — Fake and Real News Dataset
- **Size:** 44,898 articles
- **Labels:** Fake (1) / Real (0)

##  Limitations
- Dataset is mostly political news from 2016–2017
- Business/tech headlines may misclassify due to training data bias
- Real-world fake news detection remains an open research problem

##  Built With
- Python 3.13
- scikit-learn
- FastAPI
- Streamlit
- Deployed on Render
