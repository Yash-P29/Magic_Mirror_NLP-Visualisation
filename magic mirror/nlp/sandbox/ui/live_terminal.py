import os
import sys

class LiveTerminalUI:
    def __init__(self):
        self.transcript = ""
        self.is_partial_transcript = False
        self.nlp_state = {}
        self.nlp_latency = 0.0

    def clear(self):
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')

    def render(self):
        try:
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

        self.clear()
        print("========================================")
        print(" MAGIC MIRROR - NLP LIVE")
        print("========================================")
        
        status = "[MIC] Listening (Partial)..." if self.is_partial_transcript else "[MIC] Listening..."
        print(f"\nSTATUS:\n{status}")
        
        print(f"\nTRANSCRIPT:\n{self.transcript}")
        
        activity = self.nlp_state.get("activity", [])
        print("\nACTIVITY:")
        if activity:
            for act in activity:
                print(act)
        else:
            print("-")
            
        concepts = self.nlp_state.get("concepts", [])
        print("\nCONCEPT:")
        if concepts:
            for c in concepts:
                print(c)
        else:
            print("-")
            
        entities = self.nlp_state.get("entities", [])
        print("\nENTITIES:")
        if entities:
            for e in entities:
                print(e.get("name", "Unknown"))
        else:
            print("-")
            
        relationships = self.nlp_state.get("relationships", [])
        print("\nRELATIONSHIPS:")
        if relationships:
            for r in relationships:
                print(f"{r.get('subject')} -> {r.get('predicate')} -> {r.get('object')}")
        else:
            print("-")
            
        confidence = self.nlp_state.get("confidence", 0.0)
        print(f"\nCONFIDENCE:\n{confidence}")
        
        print(f"\nNLP LATENCY:\n{self.nlp_latency:.2f} sec")
        
        print("========================================")
        print("(Press CTRL+C to stop)")

    def process_queue(self, ui_queue):
        while not ui_queue.empty():
            item = ui_queue.get()
            if item["type"] == "transcript_update":
                self.transcript = item["text"]
                self.is_partial_transcript = item["is_partial"]
            elif item["type"] == "nlp_update":
                self.nlp_state = item["state"]
                self.nlp_latency = item["latency"]
            
        self.render()
