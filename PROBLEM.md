# Intent Classifier: Problem Analysis

## 1. The Problem — Visual

### Current Architecture

```
USER MESSAGE
      |
      v
+------------------+
| _needs_rag()     |  Regex patterns (6 штук)
| regex match?     |  "найди в документ", "что написано в", *.pdf...
+--------+---------+
    NO   |   YES --> [RAG mode, free]
         v
+------------------+
| _is_quick_chat() |  Greeting patterns + 1-2 word check
| short/greeting?  |  "привет", "спасибо", "ок", "как дела"...
+--------+---------+
    NO   |   YES --> [Chat mode, free]
         v
+==================+
| LLM API CALL     |  <--- THE PROBLEM
| classify_intent()|
|                  |
| Provider chain:  |
| Cerebras ------> |  fail? --> Claude proxy
|                  |
| Input:  ~500 tok |  system prompt + history + question
| Output: 1 token  |  "rag" / "chat" / "followup"
| Latency: 300-2000ms
| Cost: ~$0.001/req|
+==================+
         |
    "rag" | "chat" | "followup"
         v
   [Process by intent]
```

### The Absurdity, Visualized

```
WHAT WE SEND TO API:                     WHAT WE GET BACK:

  470 chars system prompt                     "rag"
+ doc list (variable)                         (3 bytes)
+ 2 Q&A pairs history
+ user question
= ~500-2000 tokens input
─────────────────────────
  Full HTTPS roundtrip                   Just ONE word
  TLS handshake                          from THREE options
  Load balancer
  GPU inference on 120B model
  Response serialization
  Network return trip
```

### Cost at Scale

```
              1 user      100 users     1000 users
              (50 msg/d)  (50 msg/d)    (50 msg/d)
             ─────────── ───────────── ─────────────
Messages/day:     50        5,000        50,000
After regex
  (~70% filtered): 15       1,500        15,000
                  ^^^       ^^^^^        ^^^^^^
              API calls   API calls    API calls
              for ONE     for ONE      for ONE
              word each   word each    word each

API cost/day:  ~$0.015     ~$1.50       ~$15.00
API cost/mo:   ~$0.45      ~$45.00      ~$450.00

LOCAL MODEL:   $0.00       $0.00        $0.00
              (5ms CPU)   (5ms CPU)    (5ms CPU)
```

### Failure Points

```
                     YOUR SERVER
                         |
    [1] ──── Internet ───|──── [2] Provider API
    DNS                  |     Rate limits (429)
    Routing              |     Outages (500-503)
    Latency              |     Model deprecation
                         |     Auth errors (401)
                         |
                    [3] Fallback
                    Same problems
                    + different model
                    = different behavior

LOCAL CLASSIFIER: 0 failure points
    Model loaded in RAM → classify() → done
```


## 2. Problem Details

### 2.1 What We Have Now

**File:** `app/core/classifier.py` (126 lines)

Three intents, hardcoded:
```
Intent = Literal["rag", "chat", "followup"]
```

Classification prompt: 470+ chars system prompt explaining three categories.
Response: single word. Temperature 0.0 for determinism.

**Three-tier system in `app/bot/handlers/query.py`:**

| Tier | Method | Handles | Cost |
|------|--------|---------|------|
| 1 | `_needs_rag()` — 6 regex patterns | Explicit doc requests | Free |
| 2 | `_is_quick_chat()` — greeting patterns + word count | Greetings, 1-2 word msgs | Free |
| 3 | `classify_intent()` — LLM API call | Everything else (~30%) | $$$ |

Tier 3 is the problem. Regex catches obvious cases, but ~30% of messages are "ambiguous" and require an LLM call that returns one word from three options.

### 2.2 Why This is a Real Problem

**Cost per classification request:**
- Input tokens: ~500-2000 (system prompt + history + question)
- Output tokens: 1 (literally one word)
- Cerebras pricing: Input $0.60/M, Output $0.60/M tokens
- Claude Haiku fallback: Input $0.80/M, Output $4.00/M tokens
- Per request: ~$0.0003-0.001

Sounds cheap? At 1000 users x 50 msgs/day x 30% ambiguous = 15,000 API calls/day just to get 15,000 single words.

**Latency per classification:**
- Network roundtrip: 50-200ms
- API queue wait: 0-500ms
- Inference (even for 1 token): 100-300ms
- Total: 300-2000ms added to EVERY ambiguous message
- User sees "Thinking..." for an extra 0.3-2 seconds

**Reliability:**
- Cerebras: occasional 429s and 503s
- Fallback to Claude proxy: different model = potentially different classification
- If both fail: defaults to "chat" (safe but wrong — RAG messages get missed)
- Retry logic: up to 3 attempts x backoff (1s, 3s, 7s) = up to 11 seconds worst case

### 2.3 The Configuration Tells the Story

```python
# config.py — Intent Classifier section
classifier_enabled: bool = True
classifier_max_tokens: int = 20       # We only need 1 token,
                                      # but set 20 "just in case"
classifier_model: str | None = None   # Uses the SAME model as generation
                                      # (120B parameters for 3-class classification!)
classifier_temperature: float = 0.0   # Deterministic — good, but
                                      # different models in fallback chain
                                      # break this guarantee
```


## 3. Related Problems

### 3.1 Vendor Lock-in

The classifier prompt is written for a specific LLM behavior. Switching providers means:
- Rewriting/testing the system prompt
- Different models interpret "answer in ONE word" differently
- Some models add explanations despite instructions
- The `FallbackProvider` already uses two different models — classification consistency is not guaranteed

### 3.2 Scaling Ceiling

Every user added multiplies API costs linearly. The classifier doesn't get smarter with more users — it makes the same API call every time. No learning, no caching (messages are unique), no improvement.

### 3.3 Privacy Leak

Every ambiguous message goes to an external API server just to be classified. The actual content is irrelevant to classification — only the intent matters. But the full message text, conversation history, and document list are sent externally. For a B2B product handling corporate documents, this is a compliance risk.

### 3.4 Consistency Gap

```
PRIMARY: Cerebras gpt-oss-120B, temp=0.0
    "Какие у нас сроки?" → "rag"

FALLBACK: Claude Haiku, temp=0.0
    "Какие у нас сроки?" → "chat"    (maybe)
```

Two models, same prompt, potentially different answers. The `classifier_model: None` setting means the classifier uses whatever model is currently active — primary or fallback. User experience becomes non-deterministic based on which provider is healthy.

### 3.5 Prompt Engineering Overhead

470+ characters of carefully crafted Russian prompt, with examples and edge cases, maintained alongside code. Every new intent requires prompt rewriting and testing across multiple LLM providers. Compare to:
```python
model.predict("Какие у нас сроки?")  # → "rag" (99.2% confidence)
```

### 3.6 Cold Start Problem

Bot startup doesn't preload the classifier — first classification after restart requires a full API roundtrip including connection establishment, TLS handshake, etc. A local model loads once and stays in memory.


## 4. The Bigger Picture — Universal Problems

### 4.1 The "Cannon vs Sparrow" Anti-pattern

Using a 120B-parameter generative model to choose between 3 options is like renting a cargo plane to deliver a letter. The model can write poetry, solve math, generate code — but we're asking it: "Is this 'rag' or 'chat'?" A fine-tuned 30MB BERT handles this in 5ms on CPU.

**Industry pattern:** Teams start with LLM-for-everything because it's fast to prototype. Then it works. Then nobody replaces it because "it works." Meanwhile, the bill grows linearly with usage.

### 4.2 The External Dependency Trap

The classification function — which runs on EVERY user message — depends on:
- Internet connectivity
- DNS resolution
- Provider API availability
- Provider API rate limits
- Provider billing/auth status
- Provider model availability (models get deprecated)

Any of these failing = degraded bot experience. For a feature that could run entirely locally.

### 4.3 The Hidden Cost Illusion

"$0.001 per classification" sounds trivial. But:
- It compounds: 15,000/day = $15/day = $450/month at 1000 users
- It's recurring: every message, every day, forever
- It doesn't improve: the 10,000th classification costs the same as the 1st
- A one-time investment in training a local model = $0 ongoing cost

### 4.4 The "Good Enough" Trap

The current system works. Regex handles 70%, LLM handles 30%, defaults to "chat" on errors. It's reliable enough. But "good enough" means:
- Every new customer increases costs
- Every provider outage affects ALL users
- Every price increase by the provider hits the bottom line
- The team spends time maintaining prompts instead of building features

### 4.5 Engineering Debt as a Feature

The three-tier system (regex → quick_chat → LLM) is itself a symptom. The regex patterns and greeting lists exist because the team already recognized that calling an LLM for "привет" is wasteful. The 45+ greeting patterns and 6 RAG patterns are manual attempts to reduce API calls. A trained classifier eliminates the need for all of this.


## 5. Solution Direction

**Replace Tier 3 (LLM API call) with a local fine-tuned classifier:**

```
BEFORE:                          AFTER:
regex → quick_chat → LLM API    Local Model (covers ALL tiers)
~70%     ~10%        ~20%       100% local, 0 API calls
                     ^^^^
                  $$$, slow,     ~30MB, ~5ms, offline,
                  fragile        deterministic
```

**Target model:** ruBERT-tiny2 (~30MB) or similar small Russian-language BERT, fine-tuned on intent classification dataset specific to AskBase use cases.

**What this eliminates:**
- API cost for classification: 100%
- Classification latency: 300-2000ms → 5ms
- External dependency for classification: completely removed
- Prompt engineering maintenance: eliminated
- Consistency issues between providers: eliminated
- All 45+ regex patterns and greeting lists: potentially replaceable by the model

**What this enables:**
- Offline operation for classification
- Easy addition of new intents (retrain, not re-prompt)
- Confidence scores (model returns probabilities, not just labels)
- Batch classification for analytics
- Portfolio piece: real ML project solving a real production problem
