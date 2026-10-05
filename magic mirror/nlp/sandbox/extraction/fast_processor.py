import threading
import queue

class FastProcessor:
    """
    FastProcessor acts as a low-latency queue coordinator between ASR and the LLM extractor.
    - Emits partial transcripts to ui_queue for immediate user feedback.
    - Buffers and sends finalized text chunks to nlp_queue, which is consumed solely
      by the local QwenExtractor for deep schema-compliant semantic extraction.
    """
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.accumulated_text = ""
        self.last_sent_text = ""

    def start_processing(self, text_queue: queue.Queue, nlp_queue: queue.Queue, ui_queue: queue.Queue):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(
            target=self._process_loop,
            args=(text_queue, nlp_queue, ui_queue),
            daemon=True
        )
        self.thread.start()
        print("Fast Processor started.")

    def stop_processing(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("Fast Processor stopped.")

    def _process_loop(self, text_queue: queue.Queue, nlp_queue: queue.Queue, ui_queue: queue.Queue):
        while self.is_running:
            try:
                # Get transcript from ASR
                update = text_queue.get(timeout=0.1)
                text = update.get("text", "")
                is_partial = update.get("is_partial", True)
                
                if not text:
                    continue
                
                # If it's a partial update, notify the UI for low-latency visual feedback
                if is_partial:
                    current_display_text = self.accumulated_text + " " + text
                    ui_queue.put({
                        "type": "transcript_update",
                        "text": current_display_text.strip(),
                        "is_partial": True
                    })
                else:
                    # Finalized text chunk committed from speech pause
                    self.accumulated_text += (" " + text)
                    self.accumulated_text = self.accumulated_text.strip()
                    
                    ui_queue.put({
                        "type": "transcript_update",
                        "text": self.accumulated_text,
                        "is_partial": False
                    })
                    
                    # Forward the updated text to QwenExtractor via nlp_queue
                    if self.accumulated_text != self.last_sent_text:
                        nlp_queue.put({
                            "text": self.accumulated_text
                        })
                        self.last_sent_text = self.accumulated_text

            except queue.Empty:
                continue
            except Exception as e:
                pass
