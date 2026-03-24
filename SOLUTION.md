# Intent Classifier: Solution Design

## Quick Summary

```
Что делаем:  Берём маленькую русскоязычную нейросеть (ruBERT-tiny2, 30MB)
             и учим её отвечать на один вопрос:
             "Это сообщение — вопрос по документам, болтовня или уточнение?"

Чем учим:    HuggingFace Transformers — стандартный фреймворк для обучения
             нейросетей, как Unity для геймдева или React для веба.

На чём учим: Датасет из ~3000 примеров сообщений с метками (rag/chat/followup),
             который сгенерируем сами.

Результат:   Модель ~15MB, работает за 2ms на CPU, без интернета, бесплатно.
```


## 1. Model — Why ruBERT-tiny2

### What is BERT?

BERT — это нейросеть, которая "понимает" текст. Не генерирует (как ChatGPT), а именно понимает — определяет смысл, тональность, категорию. Идеальный инструмент для классификации.

```
ChatGPT / Claude (генеративные):      BERT (понимающие):
─────────────────────────────          ─────────────────
"Напиши стихотворение о коте"         "Это сообщение о коте? Да/Нет"
→ генерирует текст                    → даёт ответ из заданных вариантов

Размер: 7-120 МИЛЛИАРДОВ параметров   Размер: 12-110 МИЛЛИОНОВ параметров
Скорость: секунды                     Скорость: миллисекунды
Стоимость: $$$                        Стоимость: бесплатно (локально)
```

### Why "tiny2"?

```
Модели BERT для русского языка:

ai-forever/ruBert-base        110M параметров    700MB    ~50ms
ai-forever/ruBert-large        340M параметров   1.3GB    ~150ms
cointegrated/rubert-tiny2       29M параметров    30MB    ~5ms   <-- THIS
                                ^^^                       ^^^^
                            В 4-12 раз меньше         В 10-30 раз быстрее

Для нашей задачи (выбор из 3 вариантов) "tiny" более чем достаточно.
Это как выбирать между грузовиком и велосипедом для поездки в соседний дом.
```

**rubert-tiny2 specifically because:**
- Обучен на русском языке (не мультиязычный — а именно русский)
- 312 hidden dimensions (компактный, но достаточный)
- Проверен сообществом на задачах классификации — работает
- Автор (cointegrated) — известный NLP-исследователь из России


## 2. Training Framework — Why HuggingFace Transformers

### What is HuggingFace?

HuggingFace — это "GitHub для нейросетей". Платформа где:
- Хранятся готовые модели (как npm-пакеты)
- Есть фреймворк `transformers` для обучения (как ORM для ML)
- Можно выложить свою модель (как npm publish)

```
Аналогия из веб-разработки:

npm install react          ←→   pip install transformers
import React from 'react'  ←→   from transformers import AutoModel
npm publish                 ←→   model.push_to_hub("my-model")
npmjs.com                   ←→   huggingface.co
```

### Why not just PyTorch?

```
PyTorch напрямую (raw):                 HuggingFace Transformers:
───────────────────                     ─────────────────────────
model = BertForSequenceClassification(  model = AutoModelForSequenceClassification
  config=BertConfig(                      .from_pretrained("rubert-tiny2",
    hidden_size=312,                       num_labels=3)
    num_attention_heads=12,
    ...50 параметров...                 trainer = Trainer(model, args, dataset)
  )                                     trainer.train()
)
optimizer = AdamW(model.parameters())   # Всё. Trainer сам:
scheduler = get_linear_schedule(...)    # - батчи
for epoch in range(3):                  # - оптимизатор
  for batch in dataloader:              # - learning rate schedule
    outputs = model(**batch)            # - логирование
    loss = outputs.loss                 # - чекпоинты
    loss.backward()                     # - evaluation
    optimizer.step()                    # - early stopping
    scheduler.step()
    ...ещё 50 строк...

~150 строк boilerplate                 ~20 строк
```

HuggingFace Trainer делает то же самое, но без ручного написания training loop. А для портфолио — "HuggingFace Transformers" в резюме = индустриальный стандарт.


## 3. Dataset — How We'll Create Training Data

### The Problem

У нас нет готового датасета. Нужно создать ~3000 примеров сообщений с метками.

### The Strategy

```
Step 1: SEED — Extract patterns from existing code
────────────────────────────────────────────────────
Из query.py уже есть 45+ regex-паттернов.
Это готовые правила, что такое "chat" и "rag":

  "привет" → chat
  "найди в документе X" → rag
  "расскажи подробнее" → followup

Step 2: GENERATE — Use LLM to create variations
────────────────────────────────────────────────
Просим Claude/GPT сгенерировать 1000 вариаций на каждый класс:

  Prompt: "Сгенерируй 100 вопросов, которые пользователь
           чат-бота по документам может задать, когда ему
           нужно НАЙТИ ИНФОРМАЦИЮ в загруженных файлах.
           Разнообразные формулировки, разговорный стиль."

  Result: "какие условия возврата?"
          "а что там про сроки доставки?"
          "сколько стоит подписка по документу?"
          ...

Step 3: VALIDATE — Manual review
────────────────────────────────
Пробежаться глазами, убрать мусор, поправить метки.
~30 минут работы на 3000 примеров.

Step 4: SPLIT — Train / Validation / Test
─────────────────────────────────────────
  Train:      70%  (~2100 примеров) — учимся
  Validation: 15%  (~450 примеров)  — проверяем во время обучения
  Test:       15%  (~450 примеров)  — финальная проверка
```

### Target Distribution

```
  rag:      ~1000 примеров   "какие условия?", "найди в договоре", ...
  chat:     ~1200 примеров   "привет", "как дела", "что ты умеешь", ...
  followup: ~800 примеров    "а подробнее?", "почему?", "объясни", ...
                             (меньше, т.к. класс проще)
```


## 4. Export — Why ONNX

### The Problem with PyTorch in Production

```
Training environment:          Production (the bot):
─────────────────────          ─────────────────────
pip install torch              pip install torch
                               ^^^^^^^^^^^^^^^^^^^^
                               +2 GB disk space
                               +500MB RAM
                               Just to load a 30MB model!
```

PyTorch — тяжёлый фреймворк. Нужен для обучения, но не для inference.

### What is ONNX?

ONNX (Open Neural Network Exchange) — универсальный формат для моделей. Как PDF для документов: создаёшь в Word, но читаешь в любом viewer-е.

```
Training (PyTorch):           Export:              Production (ONNX Runtime):
──────────────────            ───────              ──────────────────────────
torch: 2GB                    model.onnx           onnxruntime: 30MB
model: 30MB                   15MB (quantized)     model: 15MB
= тяжело                     ─────────────→        = легко
                              one-time convert
```

### Quantization — Making It Even Smaller

```
Full precision (float32):      Quantized (int8):
─────────────────────          ──────────────────
Each weight: 32 bits           Each weight: 8 bits
Model size: 30MB               Model size: ~15MB
Speed: ~5ms                    Speed: ~2ms
Accuracy: 99.1%                Accuracy: ~98.8%
                                          ^^^^
                               Потеря <0.5% точности
                               при 2x уменьшении размера
                               и 2x ускорении
```


## 5. Production Integration

### How It Fits Into AskBase

```
BEFORE (current):
─────────────────
app/core/classifier.py
  → builds prompt (470 chars)
  → calls provider.generate() over HTTPS
  → parses "rag"/"chat"/"followup" from response
  → 300-2000ms, costs money

AFTER (new):
────────────
app/core/classifier.py
  → loads ONNX model (once, on startup, ~100ms)
  → tokenizer.encode(message)  → 0.5ms
  → model.predict(tokens)      → 2ms
  → argmax → "rag"/"chat"/"followup"
  → Total: ~3ms, free, offline, deterministic
```

### What Changes in Code

```
DELETED:                              ADDED:
────────                              ──────
- LLM API call for classification     + ONNX model file (~15MB)
- 470-char system prompt              + tokenizer files (~500KB)
- History formatting for classifier   + 20 lines of inference code
- Fallback provider logic for clf     + Model loading on startup
- Retry/backoff for classification
- 45+ regex patterns (potentially)
- Greeting pattern lists (potentially)

Dependencies removed:                 Dependencies added:
─────────────────────                  ───────────────────
(none removed, but API calls           + onnxruntime (~30MB)
 are eliminated)                       + tokenizers (~5MB, already present
                                         via sentence-transformers)
```


## 6. Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│                    DEVELOPMENT (one-time)                │
│                                                         │
│  1. Create dataset                                      │
│     regex seeds + LLM generation + manual validation    │
│     → dataset.csv (~3000 rows)                          │
│                                                         │
│  2. Fine-tune rubert-tiny2                              │
│     HuggingFace Trainer, 3 epochs, ~10 min on GPU      │
│     → pytorch model (~30MB)                             │
│                                                         │
│  3. Evaluate                                            │
│     accuracy, precision, recall, confusion matrix       │
│     → target: >95% accuracy                             │
│                                                         │
│  4. Export to ONNX + quantize                           │
│     → model.onnx (~15MB)                                │
│                                                         │
│  5. Publish                                             │
│     → HuggingFace Hub (model + model card)              │
│     → GitHub (training code + dataset + results)        │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    PRODUCTION (ongoing)                  │
│                                                         │
│  Bot startup:                                           │
│    model = onnxruntime.InferenceSession("model.onnx")   │
│    tokenizer = AutoTokenizer("rubert-tiny2")            │
│                                                         │
│  Every message:                                         │
│    tokens = tokenizer.encode(message)     # 0.5ms       │
│    logits = model.run(tokens)             # 2ms         │
│    intent = ["rag","chat","followup"][argmax(logits)]    │
│    confidence = softmax(logits).max()     # bonus       │
│                                                         │
│  Total: ~3ms, $0, offline, deterministic                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```


## 7. Stack Summary

| Layer | Tool | Why | Size |
|-------|------|-----|------|
| Base model | `cointegrated/rubert-tiny2` | Russian, tiny, proven | 30MB |
| Training | HuggingFace Transformers + Trainer | Industry standard, portfolio value | dev only |
| Dataset | LLM-generated + regex seeds | Fast, cheap, sufficient | ~3000 rows |
| Export | ONNX + INT8 quantization | No torch in production | 15MB |
| Inference | onnxruntime | Lightweight, fast | 30MB |
| Hosting | HuggingFace Hub + GitHub | Visibility, portfolio | free |

**Total production footprint: ~45MB (model + runtime)**
**vs current: 2GB+ torch + API costs + internet dependency**
