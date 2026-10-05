import time
import json
import threading
import queue
import os
import re
import ollama

try:
    import spacy
    spacy_nlp = spacy.load("en_core_web_sm")
except ImportError:
    spacy_nlp = None

from sandbox.extraction.base_extractor import BaseExtractor
from sandbox.config import DEFAULT_OLLAMA_MODEL, SCHEMA_PATH, RUNTIME_OUTPUTS_DIR, SANDBOX_OUTPUTS_DIR

class QwenExtractor(BaseExtractor):
    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL):
        self.model_name = model_name
        self.is_running = False
        self.thread = None
        self.last_metrics = {}
        
        # Keep track of the current full state
        self.current_state = {}
        
        # Load schema definition to guide extraction
        try:
            with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
                self.schema_dict = json.load(f)
        except Exception as e:
            print(f"Warning: could not load schema from {SCHEMA_PATH}: {e}")
            self.schema_dict = {}

        self.system_instruction = """You are the NLP reasoning engine for Magic Mirror, a live Augmented Reality visualization system.
Your job is to read the CURRENT TEXT spoken by the user and the PREVIOUS CONTEXT.
You must output a structured JSON object representing the semantic graph and discourse intent strictly conforming to nlp_schema.json.

CRITICAL INSTRUCTIONS:
1. Distinguish between explicit statements (actually spoken) and inferred knowledge.
2. Resolve pronouns (it, this, that, they) using the PREVIOUS CONTEXT. If a pronoun refers to an entity, add to coreference_resolutions: [{"text_reference": "it", "resolved_entity": "<id>"}].
3. State Updates & Corrections: When the current text contains a correction signal (e.g. 'actually', 'no wait', 'I meant', 'correction:'), classify intent as 'correcting'. You MUST REPLACE the relevant value in semantic_graph.quantities and semantic_graph.entities.
4. Extract relationships as simple directed triples: subject -> predicate -> object. Never repeat identical relationships.
5. Accurately classify the speaker_activity intent (e.g. explaining, defining, describing, giving_example, demonstrating, comparing, analogy, hypothetical, cause_effect, process_sequencing, questioning, correcting, clarifying, summarizing, instructing, casual).
6. Do NOT invent information. If a statement has no concrete entities or objects, return entities as empty [].
7. Conciseness: Keep responses short and compact. Use short entity names (1-3 words max). Omit empty optional fields. Do not include markdown fences or conversational text.
8. Concise & Complete Entities: Include ALL key nouns and noun phrases as entities in entities[]. Every entity referenced in relationships[] or quantities[] MUST also appear in entities[].
9. Inferred Information: Provide 1-3 concrete spatial, physical, or contextual facts that the downstream AR/3D visual model needs.
10. Few-Shot Examples:
   - Example 1: "The core generates heat." -> {"speaker_activity": [{"intent": "explaining", "confidence": 0.95}], "semantic_graph": {"entities": [{"id": "core", "name": "core", "class": "object", "status": "new_explicit"}, {"id": "heat", "name": "heat", "class": "energy", "status": "new_explicit"}], "relationships": [{"subject": "core", "predicate": "generates", "object": "heat", "type": "causal", "confidence": 0.95}]}}
11. Output JSON exactly in this compact structure (omit any optional keys not needed):
{"speaker_activity": [{"intent": "...", "confidence": 0.95}], "semantic_graph": {"entities": [{"id": "...", "name": "...", "class": "...", "status": "new_explicit"}], "relationships": [{"subject": "...", "predicate": "...", "object": "...", "type": "causal", "confidence": 0.95}], "quantities": []}, "explicit_information": [], "inferred_information": [], "coreference_resolutions": [], "explanation_state": {"current_topic": "...", "active_concept": "...", "recent_entities": []}}
"""

    def _refine_intent(self, intent: str, text: str) -> str:
        """
        Refines broader intents (explaining, describing) to specific categories
        when clear, unambiguous lexical cues exist in the spoken text.
        Guards against false positives (e.g. 'the first atom' vs 'first, ...').
        """
        lower = text.strip().lower()
        
        # Self-correction signals
        if re.search(r'\b(actually|no wait|wait no|i meant|correction)\b', lower):
            return "correcting"
            
        # Example signals
        if re.search(r'\b(for instance|for example|consider a|take the case of)\b', lower):
            return "giving_example"
            
        # Cause / Effect signals (e.g. "Water freezes when temperature drops...")
        if re.search(r'\b(as a result|causes|leads to|due to this|which creates)\b', lower):
            return "cause_effect"
        if re.search(r'\bwhen\b', lower) and re.search(r'\b(drops|falls|rises|increases|decreases|cools|heats|boils|melts|freezes)\b', lower):
            return "cause_effect"
            
        # Process sequencing signals (strictly requiring sequence markers, not ordinal adjectives like 'the first atom')
        if re.search(r'\b(first|second|third|finally|step 1|step 2|firstly)\s*,', lower) or (lower.startswith("first ") and " then " in lower):
            return "process_sequencing"
            
        # Comparison signals
        if re.search(r'\b(unlike|in contrast to|compared to|whereas)\b', lower):
            return "comparing"
            
        # Analogy / Metaphor signals
        if re.search(r'\b(think of|analogous to)\b', lower) and re.search(r'\b(as|like)\b', lower):
            return "analogy"
            
        # Demonstrating / Spatial signals
        if re.search(r'\b(look right here|on my palm|in this hand|watch what happens)\b', lower):
            return "demonstrating"
            
        # Hypothetical signals
        if re.search(r'\b(suppose we|what if|hypothetically|if we were to)\b', lower):
            return "hypothetical"
            
        # Check polite casual/instruction requests before treating ? as a questioning intent
        is_polite_request = bool(re.search(r'\b(can you|could you|hey, can you|would you mind|please)\b', lower))
        has_action_verb = bool(re.search(r'\b(grab|bring|pass|get|hand|pick up|give me)\b', lower))
        if is_polite_request and has_action_verb:
            return "casual"
            
        # Instruction signals (imperatives, directions, manipulation commands)
        is_instructing = False
        if spacy_nlp is not None and text.strip():
            doc = spacy_nlp(text)
            if len(doc) > 0:
                first_token = doc[0]
                if first_token.tag_ == "VB" and first_token.lower_ not in ("suppose", "think", "look", "watch"):
                    has_subj = any(token.dep_ in ("nsubj", "nsubjpass") for token in doc)
                    if not has_subj:
                        is_instructing = True
                    
        instruction_patterns = r'\b(take this|place the|place it|put the|connect the|rotate the|set the|attach the)\b'
        if re.search(instruction_patterns, lower) or is_instructing:
            return "instructing"

        # Question signals
        if lower.endswith("?") or re.match(r'^(what is|why does|how does|can you|where is)\b', lower):
            return "questioning"
            
        # Defining signals (e.g. "A black hole is a region...", "X is defined as...")
        if re.match(r'^(a|an)\s+[\w\s]+\s+is\s+(a|an|the)\b', lower) or re.search(r'\b(is defined as|refers to|can be defined as)\b', lower):
            return "defining"
            
        # Clarification signals
        if re.search(r'\b(to be specific|specifically|in particular)\b', lower):
            return "clarifying"
            
        # Summary signals
        if re.search(r'\b(in short|in summary|to summarize|overall)\b', lower):
            return "summarizing"
            
        return intent

    def _enrich_inferred_information(self, result_dict: dict, current_text: str) -> list[str]:
        """
        Derives high-value spatial, AR grounding, and contextual inferences
        for downstream 3D/AR visualization models if not already generated.
        """
        raw_inferences = result_dict.get("inferred_information", [])
        inferences = [str(item).strip() for item in raw_inferences if isinstance(item, str) and item.strip()]
        lower_text = current_text.lower()
        
        entities = result_dict.get("semantic_graph", {}).get("entities", [])
        quantities = result_dict.get("semantic_graph", {}).get("quantities", [])
        
        # 1. Detect physical body parts / real-world anchors
        anchor_keywords = {
            "left hand": "speaker's physical left hand",
            "right hand": "speaker's physical right hand",
            "palm": "speaker's palm surface",
            "hand": "speaker's hand",
            "wrist": "speaker's wrist",
            "finger": "speaker's finger",
            "desk": "physical desk surface",
            "table": "physical table surface",
        }
        
        found_anchor = None
        for kw, desc in anchor_keywords.items():
            if kw in lower_text or any(kw in str(e.get("name", "")).lower() for e in entities if isinstance(e, dict)):
                found_anchor = desc
                break
                
        # Find non-anchor virtual or target entities
        anchor_words = ["hand", "palm", "wrist", "finger", "desk", "table"]
        target_entities = [
            e.get("name", e.get("id", "object")) 
            for e in entities 
            if isinstance(e, dict) and not any(bw in str(e.get("name", "")).lower() for bw in anchor_words)
        ]
        target_name = target_entities[0] if target_entities else "visual asset"
        
        if found_anchor:
            spatial_anchor_inf = f"The {found_anchor} serves as the real-world spatial coordinate anchor in the camera workspace."
            if not any("coordinate anchor" in inf.lower() or found_anchor in inf.lower() for inf in inferences):
                inferences.append(spatial_anchor_inf)
                
            render_inf = f"The visual asset ('{target_name}') should be tracked and pinned directly to the {found_anchor}."
            if not any("pinned" in inf.lower() or "tracked" in inf.lower() for inf in inferences):
                inferences.append(render_inf)
                
        # 2. Check for quantities / distribution
        for q in quantities:
            if isinstance(q, dict) and q.get("amount", 0) > 1:
                amt = q.get("amount")
                ent = q.get("entity", target_name)
                qty_inf = f"{amt} discrete instances of '{ent}' should be procedurally placed or aligned on the anchor surface."
                if not any(str(amt) in inf for inf in inferences):
                    inferences.append(qty_inf)
                    
        # 3. Deduce domain context for scientific / molecular / celestial concepts
        if any(w in lower_text for w in ["hydrogen", "oxygen", "water", "molecule", "atom"]):
            domain_inf = "The discourse references molecular composition; downstream renderer should project molecular bonds and 3D atomic structures."
            if not any("molecular" in inf.lower() for inf in inferences):
                inferences.append(domain_inf)
        elif any(w in lower_text for w in ["earth", "jupiter", "mars", "planet", "sun"]):
            celestial_inf = "Astronomical objects reference orbital scales; render planetary bodies with comparative scaling."
            if not any("planetary" in inf.lower() or "orbital" in inf.lower() for inf in inferences):
                inferences.append(celestial_inf)
                
        return inferences


    def extract(self, current_text: str, context_text: str, is_partial: bool) -> tuple[dict, float]:
        """
        Extract semantic meaning from the text using the local Ollama Qwen model.
        
        Args:
            current_text: The current sentence or transcript chunk.
            context_text: The rolling history of the conversation for coreference and state.
            is_partial: Whether this is a partial stream or a complete sentence.
            
        Returns:
            A tuple of (output_dict, latency_in_seconds) where output_dict matches nlp_schema.json.
        """
        start_time = time.time()
        
        user_message = f"""PREVIOUS CONTEXT (RAW TEXT):
{context_text if context_text else 'No previous context.'}

CURRENT TEXT TO PROCESS:
"{current_text}"
"""

        # Compute adaptive token budget: allocate up to 700 tokens for multi-clause/list sentences
        # (preventing premature JSON truncation on 3-4 entity graphs), while using 400-500 tokens for simple inputs.
        commas = current_text.count(',')
        clauses = commas + current_text.count(';') + current_text.count('...')
        sentences = current_text.count('.') + current_text.count('!') + current_text.count('?')
        
        if len(current_text) > 65 or clauses >= 2 or sentences > 1:
            adaptive_num_predict = 900
        elif len(current_text) > 50 or clauses == 1:
            adaptive_num_predict = 600
        else:
            adaptive_num_predict = 450

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_instruction},
                    {"role": "user", "content": user_message}
                ],
                format="json",
                options={
                    "temperature": 0.0,
                    "num_predict": adaptive_num_predict,
                    "repeat_penalty": 1.0
                }
            )
            latency = time.time() - start_time
            
            # Record telemetry metrics
            self.last_metrics = {
                "eval_count": response.get("eval_count", 0),
                "eval_duration_sec": (response.get("eval_duration") or 0) / 1e9,
                "prompt_eval_count": response.get("prompt_eval_count", 0),
                "prompt_eval_duration_sec": (response.get("prompt_eval_duration") or 0) / 1e9,
                "load_duration_sec": (response.get("load_duration") or 0) / 1e9,
                "total_duration_sec": (response.get("total_duration") or 0) / 1e9
            }
            
            raw_content = response.get("message", {}).get("content", "{}").strip()
            
            try:
                result_dict = json.loads(raw_content)
            except (json.JSONDecodeError, TypeError) as parse_err:
                print(f"Warning: JSON decode error in Qwen response ({parse_err}): {raw_content[:200]}")
                return {}, latency
                
            if not isinstance(result_dict, dict):
                return {}, latency

            # Guarantee all 9 required top-level schema keys:
            # 1. timestamp
            result_dict["timestamp"] = int(time.time())
            # 2. is_partial
            result_dict["is_partial"] = is_partial
            # 3. text_segment
            result_dict["text_segment"] = current_text
            
            # 4. speaker_activity
            if "speaker_activity" not in result_dict or not isinstance(result_dict["speaker_activity"], list) or len(result_dict["speaker_activity"]) == 0:
                result_dict["speaker_activity"] = [{"intent": "explaining", "confidence": 0.9}]
            else:
                norm_activities = []
                for act in result_dict["speaker_activity"]:
                    if isinstance(act, dict):
                        if "confidence" not in act:
                            act["confidence"] = 0.9
                        if "intent" not in act:
                            act["intent"] = "explaining"
                        act["intent"] = self._refine_intent(act["intent"], current_text)
                        norm_activities.append(act)
                    elif isinstance(act, str) and act.strip():
                        norm_activities.append({
                            "intent": self._refine_intent(act.strip(), current_text),
                            "confidence": 0.9
                        })
                result_dict["speaker_activity"] = norm_activities if norm_activities else [{"intent": "explaining", "confidence": 0.9}]
            
            # 5. semantic_graph
            if "semantic_graph" not in result_dict or not isinstance(result_dict["semantic_graph"], dict):
                result_dict["semantic_graph"] = {"entities": [], "relationships": [], "quantities": []}
            else:
                if "entities" not in result_dict["semantic_graph"] or not isinstance(result_dict["semantic_graph"]["entities"], list):
                    result_dict["semantic_graph"]["entities"] = []
                if "relationships" not in result_dict["semantic_graph"] or not isinstance(result_dict["semantic_graph"]["relationships"], list):
                    result_dict["semantic_graph"]["relationships"] = []
                else:
                    # Coerce any relationship type not in schema enum to 'composition'
                    _valid_rel_types = {"spatial", "temporal", "causal", "property", "action", "composition"}
                    for rel in result_dict["semantic_graph"]["relationships"]:
                        if isinstance(rel, dict) and rel.get("type") not in _valid_rel_types:
                            rel["type"] = "composition"
                if "quantities" not in result_dict["semantic_graph"] or not isinstance(result_dict["semantic_graph"]["quantities"], list):
                    result_dict["semantic_graph"]["quantities"] = []
                else:
                    # Deduplicate quantities by entity, keeping the latest/corrected entry
                    deduped_quantities = []
                    seen_entities = set()
                    for q in reversed(result_dict["semantic_graph"]["quantities"]):
                        if isinstance(q, dict) and "entity" in q:
                            ent_key = str(q["entity"]).strip().lower()
                            if ent_key and ent_key not in seen_entities:
                                seen_entities.add(ent_key)
                                deduped_quantities.append(q)
                    
                    # If this statement is a self-correction with an explicit number, ensure the corrected quantity is updated
                    is_correction = any(act.get("intent") == "correcting" for act in result_dict.get("speaker_activity", []))
                    if is_correction:
                        words_to_num = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
                        num_found = None
                        digit_match = re.search(r'\b\d+\b', current_text)
                        if digit_match:
                            num_found = int(digit_match.group())
                        else:
                            for w, val in words_to_num.items():
                                if re.search(r'\b' + w + r'\b', current_text.lower()):
                                    num_found = val
                                    break
                        if num_found is not None:
                            resolved_ent = None
                            for cr in result_dict.get("coreference_resolutions", []):
                                if isinstance(cr, dict) and cr.get("resolved_entity"):
                                    resolved_ent = str(cr["resolved_entity"]).strip().lower()
                                    break
                            
                            target_updated = False
                            for q in deduped_quantities:
                                ent_str = str(q.get("entity", "")).strip().lower()
                                if resolved_ent and (resolved_ent in ent_str or ent_str in resolved_ent):
                                    q["amount"] = num_found
                                    target_updated = True
                                    break
                            if not target_updated and deduped_quantities:
                                deduped_quantities[0]["amount"] = num_found

                    if deduped_quantities:
                        result_dict["semantic_graph"]["quantities"] = list(reversed(deduped_quantities))
                        
            # --- Auto-backfill Ghost Entities ---
            existing_ids = {
                str(e["id"]).strip().lower() 
                for e in result_dict["semantic_graph"]["entities"] 
                if isinstance(e, dict) and e.get("id")
            }
            existing_names = {
                str(e["name"]).strip().lower() 
                for e in result_dict["semantic_graph"]["entities"] 
                if isinstance(e, dict) and e.get("name")
            }
            
            referenced_entities = []
            # Scan relationships for subject and object
            for rel in result_dict["semantic_graph"].get("relationships", []):
                if isinstance(rel, dict):
                    for k in ("subject", "object"):
                        val = rel.get(k)
                        if val and isinstance(val, str) and val.strip():
                            referenced_entities.append(val.strip())
                            
            # Scan quantities for entity
            for q in result_dict["semantic_graph"].get("quantities", []):
                if isinstance(q, dict):
                    val = q.get("entity")
                    if val and isinstance(val, str) and val.strip():
                        referenced_entities.append(val.strip())
                        
            # Scan coreference_resolutions for resolved_entity
            for cr in result_dict.get("coreference_resolutions", []):
                if isinstance(cr, dict):
                    val = cr.get("resolved_entity")
                    if val and isinstance(val, str) and val.strip():
                        referenced_entities.append(val.strip())
                        
            # Insert any missing referenced entities
            for raw_name in referenced_entities:
                norm_name = raw_name.lower()
                norm_id = re.sub(r'[^a-z0-9_]+', '_', norm_name).strip('_') or "entity"
                if norm_id not in existing_ids and norm_name not in existing_names:
                    result_dict["semantic_graph"]["entities"].append({
                        "id": norm_id,
                        "name": raw_name,
                        "class": "inferred",
                        "status": "inferred_context"
                    })
                    existing_ids.add(norm_id)
                    existing_names.add(norm_name)
            
            # --- spaCy Noun Safety Net ---
            if spacy_nlp is not None and current_text.strip():
                doc = spacy_nlp(current_text)
                candidates = []
                for chunk in doc.noun_chunks:
                    candidates.append(chunk.text)
                for token in doc:
                    if token.pos_ in ("NOUN", "PROPN"):
                        candidates.append(token.text)
                
                # Filter candidates
                stop_words = {"it", "this", "that", "they", "he", "she", "we", "i", "you", "a", "an", "the", "some", "any"}
                for cand in set(candidates):
                    cand_lower = cand.strip().lower()
                    if not cand_lower or cand_lower in stop_words:
                        continue
                        
                    # Remove leading articles for comparison
                    clean_cand_lower = re.sub(r'^(the|a|an)\s+', '', cand_lower)
                    
                    # Check if candidate is already in existing_names or overlaps significantly
                    is_found = False
                    for existing_name in existing_names:
                        clean_existing = re.sub(r'^(the|a|an)\s+', '', existing_name)
                        if clean_cand_lower in clean_existing or clean_existing in clean_cand_lower:
                            is_found = True
                            break
                    
                    if not is_found:
                        norm_id = re.sub(r'[^a-z0-9_]+', '_', clean_cand_lower).strip('_') or "entity"
                        if norm_id and norm_id not in existing_ids:
                            result_dict["semantic_graph"]["entities"].append({
                                "id": norm_id,
                                "name": cand.strip(),
                                "class": "inferred_candidate",
                                "status": "inferred_context"
                            })
                            existing_ids.add(norm_id)
                            existing_names.add(cand_lower)

            # 6. explicit_information — must be array of strings
            if "explicit_information" not in result_dict or not isinstance(result_dict["explicit_information"], list):
                result_dict["explicit_information"] = [current_text] if current_text else []
            else:
                # Flatten any dicts/objects the model accidentally outputs as {"text": "..."}
                flat_explicit = []
                for item in result_dict["explicit_information"]:
                    if isinstance(item, str):
                        flat_explicit.append(item)
                    elif isinstance(item, dict):
                        flat_explicit.append(item.get("text") or item.get("value") or str(item))
                result_dict["explicit_information"] = flat_explicit
            
            # 7. inferred_information
            result_dict["inferred_information"] = self._enrich_inferred_information(result_dict, current_text)
            
            # 8. coreference_resolutions
            if "coreference_resolutions" not in result_dict or not isinstance(result_dict["coreference_resolutions"], list):
                result_dict["coreference_resolutions"] = []
            
            # 9. explanation_state
            if "explanation_state" not in result_dict or not isinstance(result_dict["explanation_state"], dict):
                result_dict["explanation_state"] = {
                    "current_topic": "general",
                    "active_concept": "statement",
                    "recent_entities": []
                }
            else:
                if "recent_entities" not in result_dict["explanation_state"] or not isinstance(result_dict["explanation_state"]["recent_entities"], list):
                    result_dict["explanation_state"]["recent_entities"] = [
                        e.get("id", "") for e in result_dict["semantic_graph"]["entities"] if isinstance(e, dict) and "id" in e
                    ]
                else:
                    # Flatten any objects the model outputs inside recent_entities — schema requires strings
                    flat_recent = []
                    for item in result_dict["explanation_state"]["recent_entities"]:
                        if isinstance(item, str):
                            flat_recent.append(item)
                        elif isinstance(item, dict):
                            flat_recent.append(item.get("id") or item.get("name") or str(item))
                    result_dict["explanation_state"]["recent_entities"] = flat_recent
                if "current_topic" not in result_dict["explanation_state"]:
                    result_dict["explanation_state"]["current_topic"] = "general"
                if "active_concept" not in result_dict["explanation_state"]:
                    result_dict["explanation_state"]["active_concept"] = "statement"

            return result_dict, latency
            
        except Exception as e:
            latency = time.time() - start_time
            print(f"Qwen Extractor Exception: {e}")
            return {}, latency

    def start_processing(self, nlp_queue: queue.Queue, ui_queue: queue.Queue):
        """Starts background deep processing loop for streaming updates."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, args=(nlp_queue, ui_queue), daemon=True)
        self.thread.start()
        print("Deep NLP Qwen model processor started.")

    def stop_processing(self):
        """Stops the background processor."""
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("Deep NLP Qwen model processor stopped.")

    def save_state(self, state_dict: dict = None):
        """Persists the latest NLP state to the runtime_outputs directory."""
        target_state = state_dict if state_dict is not None else self.current_state
        filename = "nlp_state_latest.json"
        for d in [RUNTIME_OUTPUTS_DIR, SANDBOX_OUTPUTS_DIR]:
            try:
                os.makedirs(d, exist_ok=True)
                filepath = os.path.join(d, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(target_state, f, indent=2)
            except Exception as e:
                print(f"Warning: could not save state to {d}: {e}")

    def _process_loop(self, nlp_queue: queue.Queue, ui_queue: queue.Queue):
        while self.is_running:
            try:
                task = nlp_queue.get(timeout=0.5)
                text = task.get("text", "")
                
                if not text:
                    continue
                
                updated_state, latency = self.extract(text, json.dumps(self.current_state), is_partial=False)
                
                if updated_state:
                    self.current_state = updated_state
                    self.save_state()
                    
                    ui_queue.put({
                        "type": "nlp_update",
                        "state": self.current_state,
                        "latency": latency
                    })
                
            except queue.Empty:
                continue
            except Exception as e:
                pass
