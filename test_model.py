"""Test the trained ONNX intent classifier locally."""

import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = "./model"
LABELS = ["rag", "chat", "followup"]

# Load model and tokenizer
print("Loading model...")
session = ort.InferenceSession(f"{MODEL_DIR}/model.onnx")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("Model loaded!\n")


def classify(text: str) -> tuple[str, float]:
    """Classify text. Returns (label, confidence)."""
    inputs = tokenizer(
        text, return_tensors="np",
        padding="max_length", truncation=True, max_length=128,
    )
    outputs = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    })
    logits = outputs[0][0]
    probs = np.exp(logits) / np.exp(logits).sum()  # softmax
    pred_id = np.argmax(probs)
    return LABELS[pred_id], float(probs[pred_id])


# ── Test messages ──
test_messages = [
    # RAG — should be "rag"
    "какие условия возврата товара?",
    "найди в документе информацию о сроках",
    "сколько стоит годовая подписка?",
    "какой штраф за просрочку?",
    "какие документы нужны для оформления?",
    "есть ли скидки для постоянных клиентов?",

    # Chat — should be "chat"
    "привет!",
    "что ты умеешь?",
    "спасибо за помощь",
    "как загрузить документ?",
    "ты бот?",
    "пока",

    # Followup — should be "followup"
    "расскажи подробнее",
    "а почему именно так?",
    "а можно пример?",
    "объясни проще",
    "а если нет?",
    "продолжай",

    # Edge cases
    "а?",
    "хм",
    "ну расскажи про доставку подробнее",
    "а если клиент отказывается?",
    "ок",
    "123",
]

print(f"{'Message':<45} {'Label':>10} {'Conf':>7}")
print("=" * 65)

for msg in test_messages:
    label, conf = classify(msg)
    print(f"{msg:<45} {label:>10} {conf:>6.1%}")

# ── Speed benchmark ──
print("\n" + "=" * 65)
print("SPEED BENCHMARK (100 runs)")
print("=" * 65)

# Warmup
for _ in range(10):
    classify("какие условия доставки?")

# Measure
times = []
for _ in range(100):
    start = time.perf_counter()
    classify("какие условия доставки?")
    times.append((time.perf_counter() - start) * 1000)

print(f"Average: {np.mean(times):.1f} ms")
print(f"Median:  {np.median(times):.1f} ms")
print(f"P95:     {np.percentile(times, 95):.1f} ms")
print(f"P99:     {np.percentile(times, 99):.1f} ms")
