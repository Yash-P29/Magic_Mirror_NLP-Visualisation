import json
import os
import re
import argparse
import statistics
from sandbox.config import DEFAULT_OLLAMA_MODEL
from sandbox.context.state_manager import ContextManager
from sandbox.extraction.qwen_extractor import QwenExtractor
from sandbox.utils.validator import NLPValidator

ACRONYM_MAP = {
    "cpu": "central processing unit",
    "gpu": "graphics processing unit",
    "ram": "random access memory",
    "dna": "deoxyribonucleic acid",
    "rna": "ribonucleic acid",
    "ai": "artificial intelligence",
    "ar": "augmented reality",
}

COMMON_STOP_WORDS = {"of", "the", "a", "an", "in", "on", "at", "to", "for", "with", "and", "is", "it", "this", "that"}
GENERIC_CATEGORY_WORDS = {"atom", "particle", "object", "feature", "thing", "one", "part", "unit"}

EXCLUSIVE_KEYWORDS = {
    # Chemistry & Physics elements / particles / polarities
    "hydrogen", "oxygen", "proton", "electron", "neutron", 
    "anode", "cathode", "positive", "negative",
    # Astronomy bodies
    "mars", "jupiter", "earth", "venus", "mercury", "saturn", "uranus", "neptune",
    # Anatomy & Directional / Spatial pairs
    "atrium", "ventricle", "left", "right", "inner", "outer", "upper", "lower",
    "anterior", "posterior", "dorsal", "ventral"
}

def is_entity_match(expected: str, actual: str) -> bool:
    ee = expected.strip().lower()
    fe = actual.strip().lower()
    
    # 1. Exact string equality or direct substring inclusion
    if ee == fe or ee in fe or fe in ee:
        return True
        
    # 2. Known bidirectional acronym resolution
    if ACRONYM_MAP.get(ee) == fe or ACRONYM_MAP.get(fe) == ee:
        return True
        
    # 3. Token-level overlap with semantic safety guards
    ee_tokens = [w for w in re.split(r'[\s_\-]+', ee) if w and w not in COMMON_STOP_WORDS]
    fe_tokens = [w for w in re.split(r'[\s_\-]+', fe) if w and w not in COMMON_STOP_WORDS]
    if not ee_tokens or not fe_tokens:
        return False
        
    # Ensure mutually exclusive domain entities never match each other
    ee_conflicts = set(ee_tokens) & EXCLUSIVE_KEYWORDS
    fe_conflicts = set(fe_tokens) & EXCLUSIVE_KEYWORDS
    if ee_conflicts and fe_conflicts and ee_conflicts != fe_conflicts:
        return False
        
    shared_tokens = set(ee_tokens) & set(fe_tokens)
    distinguishing_shared = shared_tokens - GENERIC_CATEGORY_WORDS
    
    # Must share at least one distinguishing content word
    if distinguishing_shared:
        overlap_ratio = len(shared_tokens) / min(len(ee_tokens), len(fe_tokens))
        if overlap_ratio >= 0.5:
            return True
            
    return False

def evaluate_test_case(test: dict, extractor: QwenExtractor, context_manager: ContextManager, validator: NLPValidator) -> dict:
    text = test['input']
    context_str = context_manager.get_context_string()
    
    output, latency = extractor.extract(text, context_str, is_partial=False)
    
    if not output:
        return {
            "schema_pass": False,
            "semantic_pass": False,
            "activity_pass": False,
            "entities_pass": False,
            "latency": latency,
            "err": "No output returned",
            "details": "Model returned empty or unparseable output."
        }
        
    is_valid, err = validator.validate_output(output)
    
    # 1. Activity intent matching
    expected_primary = test.get('expected_primary_activity')
    expected_secondary = test.get('expected_secondary_activity')
    activities = [a.get('intent') for a in output.get('speaker_activity', []) if isinstance(a, dict)]
    
    activity_pass = False
    if expected_primary and expected_primary in activities:
        activity_pass = True
    elif expected_secondary and expected_secondary in activities:
        activity_pass = True
    elif not expected_primary:
        activity_pass = True
        
    # 2. Entity extraction matching
    expected_entities = test.get('expected_entities', [])
    found_entities = [
        e.get('name', '').strip().lower() 
        for e in output.get('semantic_graph', {}).get('entities', []) 
        if isinstance(e, dict) and e.get('name')
    ]
    
    entities_pass = True
    missing_entities = []
    for ee in expected_entities:
        if not any(is_entity_match(ee, fe) for fe in found_entities):
            entities_pass = False
            missing_entities.append(ee)

    # 3. Optional checks for coreference & spatial reference
    extra_details = []
    expected_coref = test.get('expected_coreference')
    if expected_coref:
        found_corefs = output.get('coreference_resolutions', [])
        if not found_corefs:
            extra_details.append("Note: Expected coreference not captured.")

    semantic_pass = activity_pass and entities_pass
    
    details = ""
    if not activity_pass:
        details += f"Activity mismatch (expected '{expected_primary}', got {activities}). "
    if not entities_pass:
        details += f"Missing entities: {missing_entities}. "
    if extra_details:
        details += " ".join(extra_details)

    # Update conversation context
    context_manager.add_interaction(text)
    
    return {
        "schema_pass": is_valid,
        "semantic_pass": semantic_pass,
        "activity_pass": activity_pass,
        "entities_pass": entities_pass,
        "latency": latency,
        "err": err,
        "details": details.strip(),
        "extracted": output
    }

def evaluate_tests(test_file: str, model_name: str = DEFAULT_OLLAMA_MODEL):
    print(f"\n========================================================")
    print(f" BENCHMARK EVALUATION: {os.path.basename(test_file)}")
    print(f" Model: {model_name} (Local Ollama)")
    print(f"========================================================")
    
    if not os.path.exists(test_file):
        print(f"Error: test file not found at {test_file}")
        return

    with open(test_file, 'r', encoding='utf-8') as f:
        tests = json.load(f)

    validator = NLPValidator()
    qwen = QwenExtractor(model_name=model_name)
    context_manager = ContextManager()

    stats = {
        "total": len(tests),
        "schema_passes": 0,
        "semantic_passes": 0,
        "activity_passes": 0,
        "entities_passes": 0,
        "total_latency": 0.0
    }
    
    results_report = []
    detailed_results = []

    for idx, test in enumerate(tests, 1):
        test_id = test.get('id', f'test_{idx:03d}')
        
        # Reset context if required by test or by default for standalone tests
        if test.get("reset_context", True):
            context_manager.reset()
            qwen.current_state = {}

        res = evaluate_test_case(test, qwen, context_manager, validator)
        
        if res["schema_pass"]:
            stats["schema_passes"] += 1
        if res["semantic_pass"]:
            stats["semantic_passes"] += 1
        if res["activity_pass"]:
            stats["activity_passes"] += 1
        if res["entities_pass"]:
            stats["entities_passes"] += 1
        stats["total_latency"] += res["latency"]

        schema_status = "[PASS]" if res["schema_pass"] else "[FAIL]"
        semantic_status = "[PASS]" if res["semantic_pass"] else "[FAIL]"
        
        log_line = f"{idx:02d}. {test_id:<32} | Schema: {schema_status} | Semantic: {semantic_status} | {res['latency']:.2f}s"
        print(log_line)
        if res["details"]:
            print(f"    -> {res['details']}")
            
        results_report.append(log_line)
        if res["details"]:
            results_report.append(f"    Details: {res['details']}")
            
        detailed_results.append({
            "idx": idx,
            "id": test_id,
            "input": test.get("input"),
            "expected_primary_activity": test.get("expected_primary_activity"),
            "expected_secondary_activity": test.get("expected_secondary_activity"),
            "expected_entities": test.get("expected_entities", []),
            "res": res
        })

    total = stats["total"]
    latencies = [r["res"]["latency"] for r in detailed_results]
    avg_latency = (stats["total_latency"] / total) if total > 0 else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    median_latency = statistics.median(latencies) if latencies else 0.0
    
    stats["avg_latency"] = avg_latency
    stats["min_latency"] = min_latency
    stats["max_latency"] = max_latency
    stats["median_latency"] = median_latency
    
    summary = [
        "\n==================== BENCHMARK SUMMARY ====================",
        f"Test File:           {os.path.basename(test_file)}",
        f"Total Test Cases:    {total}",
        f"Schema Pass Rate:    {stats['schema_passes']}/{total} ({stats['schema_passes']/total*100:.1f}%)",
        f"Semantic Pass Rate:  {stats['semantic_passes']}/{total} ({stats['semantic_passes']/total*100:.1f}%)",
        f"  - Intent Matches:  {stats['activity_passes']}/{total} ({stats['activity_passes']/total*100:.1f}%)",
        f"  - Entity Matches:  {stats['entities_passes']}/{total} ({stats['entities_passes']/total*100:.1f}%)",
        f"Average Latency:     {avg_latency:.2f}s",
        f"Latency Min/Med/Max: {min_latency:.2f}s / {median_latency:.2f}s / {max_latency:.2f}s",
        "==========================================================="
    ]
    
    for line in summary:
        print(line)
        
    out_dir = os.path.join(os.path.dirname(__file__), "..", "runtime_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"benchmark_{os.path.basename(test_file)}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results_report) + "\n" + "\n".join(summary))
    print(f"\nReport saved to: {out_file}\n")

    out_json = os.path.join(out_dir, f"benchmark_{os.path.basename(test_file)}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "results": detailed_results}, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QwenExtractor on NLP test suites")
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL, help="Model to evaluate")
    parser.add_argument("--extended", action="store_true", help="Also evaluate extended test suite")
    args = parser.parse_args()

    test_file_path = os.path.join(os.path.dirname(__file__), "..", "nlp_test_cases.json")
    evaluate_tests(test_file_path, model_name=args.model)
    
    if args.extended:
        extended_test_file_path = os.path.join(os.path.dirname(__file__), "..", "nlp_extended_test_cases.json")
        evaluate_tests(extended_test_file_path, model_name=args.model)
