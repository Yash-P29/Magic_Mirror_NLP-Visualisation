# NLP Subsystem Changelog & Milestone Report

## Phase 2 Milestone: Local ASR + Local LLM Migration Complete

### Key Highlights
1. **Migration to 100% Local Inference:**
   - **ASR Engine:** Migrated from remote/heavy Whisper to `faster-whisper` (`base.en` / `small.en`) with integrated Voice Activity Detection (VAD) and repetition penalty.
   - **Neural Extractor:** Migrated from Google Gemini API to local Ollama running `qwen2.5:1.5b` with JSON-mode output.
   - **Cost & Privacy:** Completely offline, zero external API costs, zero data egress, and predictable on-device execution.

2. **Adaptive Token Budgeting Architecture:**
   - **Problem:** Fixed token limits resulted in either high latency on simple conversational inputs or premature JSON truncation (broken syntax) on multi-clause sentences containing multiple entities and relationships.
   - **Solution:** Designed and deployed an adaptive token budget heuristic:
     - Sentences with >65 chars, >=2 clauses, or multiple sentences dynamically receive `num_predict = 700`.
     - Moderate sentences receive `num_predict = 500`.
     - Short utterances receive `num_predict = 400`.
   - **Impact:** Eliminated truncated JSON failures while keeping overall extraction latency around 2 seconds.

3. **Final Benchmark Performance (`nlp_test_cases.json`):**
   - **Total Test Cases:** 20 complex discourse scenarios (cause-effect, analogy, comparison, spatial, self-correction, multi-turn).
   - **Schema Conformity:** **100.0% (20/20)** — All outputs strictly validate against `nlp_schema.json`.
   - **Semantic Pass Rate:** **70.0% (14/20)**:
     - Intent Classification: **90.0% (18/20)**
     - Entity Extraction: **75.0% (15/20)**
   - **Latency Profile:**
     - **Average:** **2.01s**
     - **Median:** **2.10s**
     - **Min / Max:** **1.35s / 3.13s**
   - **Target Achieved:** Meets the target balance for real-time live conversational AR visualization.

4. **Runtime Pipeline & Modes:**
   - `--text "<input>"`: Direct text input testing bypassing ASR.
   - `--mic`: Continuous microphone capture with real-time speech-to-intent pipeline.
   - `--live`: Split-pane live terminal interface tracking partial audio streams, final transcriptions, and active semantic entity graph.

---

### Known Semantic Failure Analysis (Remaining 30%)
The 6 test cases that did not meet strict semantic pass criteria are categorized below:
- **test_006_comparing & test_007_analogy:** Complex relational metaphors (Earth vs Jupiter, CPU vs computer). The 1.5B model extracts the primary subject accurately but occasionally omits the comparative baseline noun.
- **test_015_coreference & test_018_ambiguous:** Subtle classification boundaries between `explaining` and `describing` for descriptive statements, and pronominal entity resolution for `"It"`.
- **test_019_casual_speech:** Model filters conversational filler (`"section"`) in favor of actionable domain entities.
- **test_020_multi_sentence_context:** Secondary contextual terms (`"axis"`, `"seasons"`) dropped in favor of core planetary bodies.

---

### Readiness Status
- **Pipeline State:** Fully operational, tested, and self-contained.
- **Handoff Ready:** The NLP subsystem is verified and ready for downstream integration with the 3D Visual Rendering phase via `nlp/runtime_outputs/nlp_state_latest.json`.
