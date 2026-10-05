# Magic Mirror NLP Subsystem

## Overview
The NLP subsystem is the semantic engine of Magic Mirror. It listens to a user speaking naturally while they explain, describe, demonstrate, or compare concepts. Crucially, the NLP module treats speech as context, not as an explicit command interface.

Its goal is to extract the **meaning, intent, and structure** of the discourse and output this as a standardized, real-time JSON stream conforming strictly to `nlp_schema.json`. The downstream 3D/AR visualization pipeline ("Brain" & Visual Rendering) consumes this structured stream to autonomously decide when, where, and how to project diagrams, models, and animations in Augmented Reality.

## Key Capabilities
- **Local Streaming ASR:** Uses `faster-whisper` (`base.en` / `small.en`) with integrated Voice Activity Detection (VAD) and repetition penalty to transcribe natural speech with sub-second latency.
- **Local Neural Extraction:** Powered completely offline by Ollama running `qwen2.5:1.5b`, providing zero API costs, zero cloud telemetry, and 100% privacy.
- **Adaptive Token Budgeting:** Dynamically balances speed and completeness by scaling output prediction budgets (`num_predict`: 400–700 tokens) based on syntactic complexity and sentence length, preventing JSON truncation on complex entity graphs while achieving ~2s average latency.
- **Fast Processor Bypass:** Instantaneous rule-based extraction (<1ms) for basic conversation, greetings, and simple queries to preserve compute.
- **State & Context Tracking:** Manages multi-turn conversation history, resolving pronouns (`it`, `this`, `that`) to persistent entity IDs and handling real-time verbal self-corrections.
- **Strict Schema Enforcement:** Automatic JSON structural repair and schema validation ensuring 100% valid payloads against `nlp_schema.json`.

---

## Local Architecture & Migration

| Component | Legacy Architecture (Phase 1) | Current Local Architecture (Phase 2) |
|---|---|---|
| **ASR Engine** | Cloud / Heavy Whisper | `faster-whisper` (`base.en`, CPU/GPU float16/int8) |
| **LLM Reasoning** | Cloud Gemini API | Local Ollama (`qwen2.5:1.5b`, JSON mode) |
| **Average Latency** | ~3.5s - 5.0s (Network dependent) | **2.01s** (Min: 1.35s, Max: 3.13s) |
| **Schema Pass Rate** | 95% | **100.0%** (20/20 test cases) |
| **Semantic Pass Rate**| ~65% | **70.0%** (Intent: 90%, Entities: 75%) |
| **Privacy & Cost** | Requires API Key & Internet | **100% Offline, Zero Cost** |

---

## Setup & Prerequisites

1. **Install Ollama:**
   Download and install from [ollama.com](https://ollama.com/).
2. **Pull the Local Model:**
   ```bash
   ollama pull qwen2.5:1.5b
   ```
3. **Install Dependencies:**
   From the repository root or `nlp/`:
   ```bash
   pip install -r nlp/requirements.txt
   ```
   *(Ensure PyAudio has access to your system microphone).*

---

## Running the Pipeline

All execution modes are run via `nlp/sandbox/run.py`:

### 1. Text Mode (`--text`)
Runs direct text extraction through the NLP pipeline (bypasses microphone). Useful for testing specific phrases, complex relationships, and debugging.
```bash
python -m sandbox.run --text "Water is made up of two hydrogen atoms and one oxygen atom."
```
Output:
- Prints real-time extraction metrics (latency, prompt evaluation, tokens).
- Displays parsed `speaker_activity` and `semantic_graph` (entities, relationships, quantities).
- Saves full state to `nlp/runtime_outputs/`.

### 2. Microphone Mode (`--mic`)
Captures live audio from your microphone, transcribes speech with `faster-whisper`, and runs semantic extraction on complete sentences.
```bash
python -m sandbox.run --mic
```
Press `Ctrl+C` at any time to cleanly stop the audio stream and view accumulated session state.

### 3. Live Terminal Dashboard (`--live`)
Launches the full interactive live terminal UI. It continuously visualizes streaming partial ASR transcripts, final transcript utterances, real-time token/latency stats, and the active semantic entity graph in a split-pane layout.
```bash
python -m sandbox.run --live
```

---

## Benchmarking & Evaluation

The subsystem includes a comprehensive 20-case test suite (`nlp/nlp_test_cases.json`) covering complex discourse patterns: causal chains, analogies, comparisons, self-corrections, spatial relations, and multi-sentence context.

To run the automated benchmark harness:
```bash
python -m sandbox.evaluate
```

### Benchmark Summary (Final Phase 2 Result)
```text
==================== BENCHMARK SUMMARY ====================
Test Suite:           nlp_test_cases.json
Total Test Cases:     20
Schema Pass Rate:     20/20 (100.0%)
Semantic Pass Rate:   14/20 (70.0%)
  - Intent Matches:   18/20 (90.0%)
  - Entity Matches:   15/20 (75.0%)
Average Latency:      2.01s
Latency Min/Med/Max:  1.35s / 2.10s / 3.13s
===========================================================
```
Results and full JSON payloads are automatically archived to `nlp/runtime_outputs/`.

---

## Known Semantic Limitations

While schema conformity is **100%**, the remaining 30% semantic misses (6 out of 20 edge cases) stem from compact 1.5B model capacity trade-offs:
1. **Implicit Entity Omission in Comparisons (`test_006`, `test_007`):** In complex multi-entity comparisons (e.g., comparing Earth and Jupiter as gas giants, or CPUs to computer brains), the model sometimes captures the primary subject while omitting the comparative baseline noun.
2. **Subtle Intent Nuances (`test_015`, `test_018`):** When describing properties of an object ("It is very dense..."), the model occasionally classifies the utterance as `explaining` rather than `describing`.
3. **Pronominal/Grammatical Placeholder Entities (`test_018`, `test_019`):** The benchmark expects placeholder tokens like `"It"` or generic conversational parts like `"section"`, which the model filters out in favor of concrete nouns.
4. **Multi-Clause Entity Drop (`test_020`):** Long multi-sentence paragraphs can drop secondary background nouns (e.g., `"axis"`, `"seasons"`) when concentrating tokens on primary physical bodies.

*These limitations are well within acceptable parameters for AR visual grounding, where primary nouns and physical relationships drive 3D asset generation.*

---

## Downstream Interface (Handoff to Visual Rendering)

The NLP subsystem writes its canonical state to:
- `nlp/runtime_outputs/nlp_state_latest.json`

Downstream consumers (the Visual Rendering / AR engine) can poll or subscribe to this file to read:
- `entities`: Array of active entities with persistent IDs, class tags, and explicit/inferred status.
- `relationships`: Directed subject-predicate-object triples for spatial/causal graph placement.
- `quantities`: Numerical values and units bound to entities for scaling and procedural generation.
- `speaker_activity`: Discourse intent and confidence for triggering animation modes.
