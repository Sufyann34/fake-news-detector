#!/bin/bash
python -c "import nltk; nltk.download('stopwords')"
uvicorn api:app --host 0.0.0.0 --port $PORT