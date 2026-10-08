import time
import json
from integration.runtime.multimodal_runtime import MultimodalRuntime
from integration.tests.test_integration import create_nlp_json, create_vision_scene

def run_evaluations():
    print("==================================================")
    print("Running Multimodal Integration Evaluation Suite...")
    print("==================================================\n")
    
    runtime = MultimodalRuntime()
    total_latency_ms = 0.0
    cases_run = 0

    eval_cases = [
        {
            "name": "Imagine a dog on my hand",
            "nlp": create_nlp_json("Imagine a dog sitting on my hand.", "hypothetical", entities=[{"id": "dog_1", "name": "dog", "class": "animal", "status": "new_explicit"}], rels=[{"subject": "dog_1", "predicate": "sitting_on", "object": "hand_1", "type": "spatial"}]),
            "vision": create_vision_scene(hand_available=True)
        },
        {
            "name": "Move it onto the table",
            "nlp": create_nlp_json("Move it onto the table.", "instructing", corefs=[{"text_reference": "it", "resolved_entity": "dog_1"}]),
            "vision": create_vision_scene(table_available=True)
        }
    ]

    for i, case in enumerate(eval_cases):
        print(f"Test Case {i+1}: {case['name']}")
        if case["vision"]:
            runtime.process_vision_update(case["vision"])
            
        start_time = time.perf_counter()
        plan = runtime.process_nlp_update(case["nlp"])
        end_time = time.perf_counter()
        
        latency = (end_time - start_time) * 1000
        total_latency_ms += latency
        cases_run += 1
        
        print(f"  Latency: {latency:.2f} ms")
        if plan:
            print(f"  Action: {plan.action.value}")
            print(f"  Concept: {plan.concept}")
            print(f"  Visual ID: {plan.visual_id}")
            if plan.spatial and plan.spatial.anchored:
                print(f"  Anchored to: {plan.spatial.anchor_type.value}")
        print("\n")
        
    print("==================================================")
    print(f"Average Decision Latency: {(total_latency_ms / max(1, cases_run)):.2f} ms")
    print("Evaluation Complete.")

if __name__ == "__main__":
    run_evaluations()
