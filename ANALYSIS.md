# Pros, Cons & Honest Analysis

## Pros

### 1. Real cost savings
Not a theoretical exercise — this replaces actual API calls in a production bot. At scale (1000 users), saves ~$450/month. Even at 10 users, it removes a dependency that costs time and attention.

### 2. Speed improvement is dramatic
3.7ms vs 300-2000ms. The user literally cannot perceive the classification step anymore. It's faster than a single network packet roundtrip.

### 3. Zero runtime cost
After one-time training (~10 min on free GPU), the model runs forever at $0. No tokens, no API keys, no billing surprises.

### 4. No external dependencies for classification
Bot works fully offline for the classification step. No more "Cerebras is down, falling back to Claude, which classifies differently." Deterministic, consistent, always available.

### 5. Privacy
User messages stay on the server. No third-party sees the text just to determine its category. Important for B2B customers with sensitive documents.

### 6. Portable knowledge
The same approach works for any chatbot, any language, any set of intents. The pipeline (seeds → dataset → fine-tune → ONNX → deploy) is reusable.

### 7. Portfolio value
Demonstrates real ML engineering: problem identification → data creation → training → optimization → deployment. Not a Kaggle competition — a production solution.

---

## Cons

### 1. Accuracy is lower than LLM
90% F1 vs estimated 95%+ from a 120B-parameter model with a carefully crafted prompt. The 5% gap means roughly 1 in 20 messages gets misclassified. In practice:
- A "followup" message classified as "chat" → bot responds without context (annoying but not critical)
- A "rag" message classified as "chat" → bot doesn't search documents (user has to rephrase)

**Mitigation:** Confidence threshold — if model is <70% confident, fall back to API. Best of both worlds.

### 2. Followup class is weak
73% recall — 27% of followup messages get misclassified. This is partly because followup is inherently ambiguous ("а сроки?" could be a new question or a clarification) and partly because we have fewer training examples (654 vs 1309 for RAG).

**Mitigation:** More followup training data, or treat low-confidence followup as RAG (safer default).

### 3. Model is not quantized
111MB instead of target ~30MB. Quantization step failed during training. The model works fine but takes more disk space and memory than necessary.

**Mitigation:** Re-run quantization locally or in Colab. One-time fix, ~5 minutes.

### 4. Dataset is synthetic
All 2,877 examples are generated from templates and augmentation, not from real user messages. The model may underperform on message patterns it hasn't seen in training. Real user data would dramatically improve quality.

**Mitigation:** After deploying, log real messages with model predictions, manually correct wrong ones, retrain periodically. This is standard active learning.

### 5. Only 3 classes
The current model only distinguishes rag/chat/followup. If new intents are needed (e.g., "complaint", "order_status", "feedback"), the model must be retrained. An LLM can handle new categories with just a prompt change.

**Mitigation:** Adding new classes requires new seed data + retraining (~30 min total). Not zero-effort, but not hard either.

### 6. No context awareness
The model classifies each message independently. It doesn't see conversation history. The LLM classifier sees previous Q&A pairs, which helps with ambiguous messages.

**Mitigation:** For v2, concatenate last bot response with current message as input. Requires dataset restructuring.

### 7. Declension bugs in dataset
The template generator has naive Russian declension for some document names ("спецификацияе" instead of "спецификации"). Affects ~15-20 examples. Doesn't significantly impact model quality but needs fixing for a clean dataset.

---

## Trade-off Summary

| Dimension | LLM API | Local Classifier | Winner |
|-----------|---------|-------------------|--------|
| Accuracy | ~95% | ~90% | API |
| Speed | 300-2000ms | 3.7ms | **Local** |
| Cost | ~$450/mo at scale | $0 | **Local** |
| Reliability | External dependency | Offline | **Local** |
| Privacy | Data sent externally | Data stays local | **Local** |
| Flexibility | New intents via prompt | Retrain needed | API |
| Context awareness | Sees history | Single message | API |
| Consistency | Varies by provider | Deterministic | **Local** |

**Score: Local 5 — API 3.** The local classifier wins on the metrics that matter most for production (cost, speed, reliability, privacy). The API wins on flexibility and accuracy, but these gaps can be closed with confidence thresholds and better data.

---

## Honest Assessment

This is a **good v1 that solves the stated problem.** It's not perfect — 90% accuracy with a synthetic dataset leaves room for improvement. But it:

1. Eliminates a real cost center ($450/mo at 1000 users)
2. Removes a real reliability risk (API dependencies)
3. Improves user experience (3.7ms vs seconds)
4. Creates a foundation for iterative improvement (active learning)

For a portfolio project, it demonstrates the right thinking: identify a problem → quantify the cost → build a targeted solution → measure results → document trade-offs. That's what engineering teams want to see.

**What would make it great (v2):**
- Train on real user messages (active learning)
- Add confidence threshold with API fallback
- Quantize to ~30MB
- Add conversation context as input feature
- Expand to 5-7 intent classes
