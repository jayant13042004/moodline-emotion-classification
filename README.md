# Moodline: Emotion Classification

A six-class text emotion classifier (sadness, joy, love, anger, fear, surprise) built with TensorFlow and Keras and served through FastAPI.

## Model
- Trained on **16,000 training examples**
- Compared Simple RNN, LSTM, GRU, and BiGRU architectures
- Final BiGRU: 300-d embedding → two BiGRU layers (128 and 64 units) → dropout 0.5 → six-class softmax

## Results
| Model | Accuracy |
|---|---|
| Simple RNN / LSTM / GRU | [fill from notebook] |
| **BiGRU (final)** | **92.10%** (loss 0.2258, as reported in the notebook) |

Note: the test split was also used as validation during training, so treat this figure as a notebook-reported accuracy, not an independent held-out score.

## API
FastAPI endpoint returns the predicted emotion, confidence, and class-wise probabilities.
```bash
uvicorn [app_file]:app --reload      # docs at http://127.0.0.1:8000/docs
```
Example response:
```json
{ "emotion": "joy", "confidence": [x], "probabilities": { "joy": [x], "sadness": [x], "...": [x] } }
```

## Run locally
```bash
git clone https://github.com/jayant13042004/moodline-emotion-classification.git
cd moodline-emotion-classification
pip install -r requirements.txt
```

## Next steps
- Re-evaluate with a separate validation set and held-out test set
- Try a transformer baseline for comparison

## Author
Jayant Jagtap · [LinkedIn](https://www.linkedin.com/in/jayantjagtap/)
