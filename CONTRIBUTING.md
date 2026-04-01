# Contributing to Intent Classifier

## Development Setup

1. Clone and install dependencies:

```bash
git clone https://github.com/GleckusZeroFive/intent-classifier.git
cd intent-classifier
pip install transformers torch datasets scikit-learn pandas
```

2. The training notebook is `train.ipynb` — run it in Jupyter or VS Code.

## Project Structure

- `train.ipynb` — model training notebook (fine-tuning ruBERT-tiny2)
- `generate_dataset.py` — synthetic dataset generation
- `seeds.py` — seed examples for dataset generation
- `test_model.py` — model evaluation script
- `dataset.csv` — training dataset
- `new_training_data.jsonl` — additional training examples
- `PROBLEM.md` / `SOLUTION.md` / `ANALYSIS.md` — documentation of the approach

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Test the model: `python test_model.py`
5. Commit with a clear message
6. Push and open a Pull Request

## Key Areas for Contribution

- **Dataset quality** — adding edge cases to `dataset.csv` or `seeds.py`
- **New intents** — extending the classifier beyond rag/chat/followup
- **Evaluation** — adding metrics, cross-validation, confusion matrix analysis
- **Integration examples** — showing how to use the model in different frameworks

## Code Style

- Python 3.10+
- HuggingFace Transformers API
- Keep the model lightweight — the whole point is sub-5ms inference

## Reporting Issues

Open an issue with:
- Example input that was misclassified
- Expected vs actual label
- Model version / commit hash
