# Project Review & Audit

## Summary

Project: fine-tuned ruBERT-tiny2 intent classifier for RAG chatbot.
Status: **Working prototype**, ONNX model exported and tested locally.
Overall: solid first iteration, production-ready after minor fixes.

---

## What Works Well

### Architecture
- Correct choice of model (ruBERT-tiny2) — compact, Russian-specific, proven
- ONNX export for production — no PyTorch dependency at runtime
- Clean separation: seeds → generator → dataset → training → export

### Code Quality
- `seeds.py` — well-structured, clear categories, 279 hand-crafted examples
- `generate_dataset.py` — smart template system with pre-declined phrases (avoids most grammar issues)
- `test_model.py` — includes edge cases and speed benchmark
- `train.ipynb` — complete Colab pipeline, runs in <10 minutes

### Documentation
- `PROBLEM.md` — thorough problem analysis with diagrams, costs, failure points
- `SOLUTION.md` — explains every choice with analogies (accessible to non-ML readers)
- `README.md` — complete project overview with usage examples

### Results
- 90% overall F1-score on test set
- 3.7ms inference on CPU
- Model size: ~111MB (ONNX, non-quantized)

---

## Issues Found

### Critical (must fix before production)

**1. Declension bugs in dataset generator**
The `_DOC_NAMES` template naively appends "е" for prepositional case, breaking words ending in "-ция"/"-ия":
```
"спецификация" → "спецификацияе" (wrong, should be "спецификации")
"инструкция"   → "инструкцияе"  (wrong, should be "инструкции")
```
Affects ~15-20 training examples. Not fatal (model still learns the pattern), but lowers quality and looks bad in portfolio review.

**Fix:** Add proper declension rules for words ending in "-ция", "-ия", "-ие" in `generate_dataset.py`, or use pre-declined forms like the `_TOPICS_PHRASES` approach.

### Important (should fix)

**2. Quantization not applied**
README mentions INT8 quantization (~15MB), but the actual exported model is full-precision ONNX:
- `model.onnx` — 172KB (graph definition only)
- `model.onnx.data` — 116MB (weights, full float32)
- Total: ~111MB vs expected ~30MB

The quantization step in Colab failed (likely due to `QUANTIZED_DIR` path issue — the workaround cell redirected to non-quantized model). Model works correctly, just larger than optimal.

**Fix:** Re-run quantization in Colab, or add a local quantization script.

**3. No `.gitignore`**
Missing `.gitignore` means:
- `__pycache__/` will be committed
- Model files (111MB) will bloat the repo — need Git LFS or exclude from repo
- `.ipynb_checkpoints/` may appear

**4. Model too large for GitHub (111MB)**
GitHub has a 100MB file size limit. `model.onnx.data` is 116MB.
Options:
- a) Git LFS (recommended)
- b) Host model on HuggingFace Hub, link from README
- c) Quantize to get under 100MB

### Minor (nice to have)

**5. `test_model.py` hardcodes `./model` path**
Works locally but would break if run from a different directory. Consider `Path(__file__).parent / "model"`.

**6. Train notebook has no token_type_ids**
The model export doesn't include `token_type_ids` input. ruBERT-tiny2 supports them but the current export ignores them. Functionally fine for single-sentence classification, but adding them could marginally improve accuracy.

**7. Dataset balance**
rag:1309, chat:914, followup:654 — followup is underrepresented (23% vs 45% for rag). This correlates with followup being the weakest class (73% recall). More followup examples would help.

---

## Metrics Assessment

| Metric | Value | Verdict |
|--------|-------|---------|
| Overall F1 | 0.90 | Good for first iteration |
| RAG F1 | 0.96 | Excellent |
| Chat F1 | 0.88 | Good |
| Followup F1 | 0.79 | Acceptable, room for improvement |
| Inference speed | 3.7ms | Excellent |
| Model size | 111MB | Too large, quantization needed |

### Accuracy vs API comparison
The current LLM API classifier likely achieves ~95%+ accuracy (large model, task-specific prompt). Our model at 90% is a trade-off: slightly lower accuracy for massive cost/speed gains. For production, the confidence threshold approach (fallback to API for low-confidence predictions) would close this gap.

---

## Security & Privacy

- No sensitive data in dataset (all synthetic examples)
- No API keys or credentials in code
- Model runs locally — no data leaves the server
- No user data was used for training

---

## Verdict

**Ready for portfolio and GitHub.** Fix the .gitignore and model hosting before pushing. Declension bugs are cosmetic — they don't significantly affect model quality but should be fixed for a clean portfolio presentation.

**Ready for production after:**
1. Quantization (reduce from 111MB to ~30MB)
2. Confidence threshold integration (fallback to API for uncertain predictions)
3. More followup training data
