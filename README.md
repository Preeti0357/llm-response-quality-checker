# LLM Response Preference Predictor

A beginner-friendly baseline that learns from the Kaggle **LLM Classification – Human Preference Prediction** dataset. It compares two responses to the same prompt and predicts whether human raters preferred Response A, Response B, or considered them a tie.

## 1. Get the data
Download `train.csv` from:
https://www.kaggle.com/competitions/llm-classification-finetuning/data

Place it here: `data/train.csv`

The training script expects these columns:
- `prompt`
- `response_a`
- `response_b`
- `winner_model_a`
- `winner_model_b`
- `winner_tie`

## 2. Install dependencies
Open a terminal in this project folder and run:

```bash
python -m pip install -r requirements.txt
```

## 3. Train
```bash
python train.py
```

This creates `models/preference_model.joblib` and prints held-out evaluation metrics.

If your computer runs out of memory, train on a smaller random sample (for example, 20,000 rows):
- Windows CMD: `set MAX_ROWS=20000`
- PowerShell: `$env:MAX_ROWS=20000`
- Then run `python train.py`

Set `MAX_ROWS=0` or remove the variable to use the full dataset.

## 4. Evaluate
```bash
python evaluate.py
```

## 5. Launch the app
```bash
python -m streamlit run app.py
```

## Limitations
This is a text-feature baseline. It does not truly verify facts, detect every hallucination, or understand all aspects of quality. Kaggle labels represent human preference between two responses, not an absolute `Good`/`Bad` label for one response. Treat predictions as experimental.
