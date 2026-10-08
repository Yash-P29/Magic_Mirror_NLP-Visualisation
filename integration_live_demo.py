"""
integration_live_demo.py
========================
Milestone: REAL NLP + REAL VISION → REAL MULTIMODAL CONTEXT → REAL VISUAL PLAN

Smallest possible live demo wrapper around the EXISTING systems.
- Does NOT modify any existing Vision or NLP code.
- Does NOT add Three.js / any renderer.
- Does NOT generate any visual content.
- ONLY wires the existing pipelines together and adds a HUD on the camera frame.

Run from the project root:
    python integration_live_demo.py

Optional flags (same as the existing Vision demo):
    --device N       camera index (default 0)
    --backend dshow|msmf|any
    --mic-device N   microphone device index (default 1)
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allows imports from both sub-projects without installation
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_VIS_SRC = _ROOT / "MagicMirror_Visualisation"
_NLP_SANDBOX = _ROOT / "magic_mirror_nlp" / "nlp" / "sandbox"
_NLP_PARENT = _ROOT / "magic_mirror_nlp" / "nlp"

for _p in [str(_ROOT), str(_VIS_SRC), str(_NLP_SANDBOX), str(_NLP_PARENT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Vision imports  (existing, unmodified)
# ---------------------------------------------------------------------------
from src.camera.capture import AsyncCameraReader, Camera, CameraError
from src.perception.scene import SceneBuilder

# ---------------------------------------------------------------------------
# NLP imports  (existing, unmodified)
# ---------------------------------------------------------------------------
from audio.microphone import MicrophoneInput
from asr.whisper_transcriber import WhisperTranscriber
from extraction.fast_processor import FastProcessor
from extraction.qwen_extractor import QwenExtractor
from config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_MIC_DEVICE_INDEX,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_COMPUTE_TYPE,
)

# ---------------------------------------------------------------------------
# Integration imports  (existing, unmodified)
# ---------------------------------------------------------------------------
from integration.runtime.multimodal_runtime import MultimodalRuntime
from integration.multimodal_fusion.schemas import VisualPlan, ActionType

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("integration_demo")

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------
_WINDOW          = "Magic Mirror — NLP + Vision Integration Demo"
_FONT            = cv2.FONT_HERSHEY_SIMPLEX
_FS_SMALL        = 0.48
_FS_LABEL        = 0.52
_FT              = 1
_LT              = cv2.LINE_AA
_LH              = 20

# Colours (BGR)
_C_SHADOW        = (10,  10,  10)
_C_TEXT          = (220, 220, 220)
_C_OPPORTUNITY   = (0, 220, 220)    # teal-yellow — matches existing demo
_C_NLP_PANEL     = (80,  220, 255)  # amber for NLP section
_C_FUSION_PANEL  = (100, 255, 100)  # green for fusion section
_C_NONE          = (120, 120, 120)  # grey for idle/NONE
_C_ACTION        = (50,  200, 255)  # bright for active action


# ---------------------------------------------------------------------------
# Shared live state
# ---------------------------------------------------------------------------
class LiveState:
    """Thread-safe container for latest NLP + multimodal data."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.transcript: str = ""
        self.is_partial: bool = False
        self.intent: str = "—"
        self.intent_conf: float = 0.0
        self.entities: list = []
        self.plan: Optional[VisualPlan] = None

    def update_transcript(self, text: str, partial: bool) -> None:
        with self._lock:
            self.transcript = text
            self.is_partial = partial

    def update_nlp(self, raw_json: dict) -> None:
        with self._lock:
            acts = raw_json.get("speaker_activity", [])
            if acts:
                self.intent = acts[0].get("intent", "—")
                self.intent_conf = acts[0].get("confidence", 0.0)
            else:
                self.intent = "—"
                self.intent_conf = 0.0
            self.entities = raw_json.get("semantic_graph", {}).get("entities", [])
            # Sync transcript if the NLP state carries the processed text
            seg = raw_json.get("text_segment", "")
            if seg:
                self.transcript = seg
                self.is_partial = raw_json.get("is_partial", False)

    def update_plan(self, plan: VisualPlan) -> None:
        with self._lock:
            self.plan = plan

    def snapshot(self):
        with self._lock:
            return (
                self.transcript,
                self.is_partial,
                self.intent,
                self.intent_conf,
                list(self.entities),
                self.plan,
            )


# ---------------------------------------------------------------------------
# Diagnostic monitor  (TEMPORARY — boundary probe only)
# ---------------------------------------------------------------------------

class _DiagMonitor:
    """
    Attaches to the existing queues via thin put/get wrappers.
    Prints one heartbeat line per second for each boundary.
    Prints transcript / NLP event lines only when the value changes.
    No existing NLP or Vision files are modified.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # A — microphone callback counters
        self.mic_chunks:  int   = 0      # audio_queue.put calls
        self.mic_samples: int   = 0      # int16 samples received
        self.mic_rms_acc: float = 0.0    # running RMS accumulator (sum of rms)
        self.mic_rms_n:   int   = 0

        # B — Whisper input counters
        self.wh_chunks_consumed: int = 0  # audio_queue.get calls by Whisper

        # C — Whisper output / text_queue
        self.wh_partials: int = 0
        self.wh_finals:   int = 0
        self._last_transcript: str = ""

        # D — nlp_queue (FastProcessor → Qwen)
        self.nlp_q_puts: int = 0

        # E — ui_queue puts (all types emitted by sub-threads)
        self.ui_puts: int = 0
        self._last_nlp_event_type: str = ""

        # Heartbeat timing
        self._last_hb: float = time.monotonic()

    # ------------------------------------------------------------------
    # Boundary A — microphone → audio_queue
    # ------------------------------------------------------------------
    def wrap_mic_put(self, original_put):
        """Returns a replacement for audio_queue.put that counts samples."""
        mon = self
        def _put(item, *args, **kwargs):
            chunk_bytes, _ = item
            int_data  = np.frombuffer(chunk_bytes, dtype=np.int16)
            float_data = int_data.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(float_data ** 2))) if len(float_data) else 0.0
            with mon._lock:
                mon.mic_chunks  += 1
                mon.mic_samples += len(int_data)
                mon.mic_rms_acc += rms
                mon.mic_rms_n   += 1
            return original_put(item, *args, **kwargs)
        return _put

    # ------------------------------------------------------------------
    # Boundary B — Whisper reading from audio_queue
    # ------------------------------------------------------------------
    def wrap_audio_q_get(self, original_get):
        """Returns a replacement for audio_queue.get that counts Whisper reads."""
        mon = self
        def _get(*args, **kwargs):
            result = original_get(*args, **kwargs)
            with mon._lock:
                mon.wh_chunks_consumed += 1
            return result
        return _get

    # ------------------------------------------------------------------
    # Boundary C — Whisper → text_queue (transcripts)
    # ------------------------------------------------------------------
    def wrap_text_q_put(self, original_put):
        """Returns a replacement for text_queue.put that logs transcripts."""
        mon = self
        def _put(item, *args, **kwargs):
            text       = item.get("text", "") if isinstance(item, dict) else ""
            is_partial = item.get("is_partial", True) if isinstance(item, dict) else True
            label = "partial" if is_partial else "final"
            with mon._lock:
                if is_partial:
                    mon.wh_partials += 1
                else:
                    mon.wh_finals += 1
                changed = (text != mon._last_transcript)
                if changed:
                    mon._last_transcript = text
            if changed:
                print("[WHISPER] %s: '%s'" % (label, text[:120]))
            return original_put(item, *args, **kwargs)
        return _put

    # ------------------------------------------------------------------
    # Boundary D — FastProcessor → nlp_queue
    # ------------------------------------------------------------------
    def wrap_nlp_q_put(self, original_put):
        """Returns a replacement for nlp_queue.put that logs Qwen inputs."""
        mon = self
        def _put(item, *args, **kwargs):
            text = item.get("text", "")[:80] if isinstance(item, dict) else str(item)[:80]
            with mon._lock:
                mon.nlp_q_puts += 1
                n = mon.nlp_q_puts
            print("[FAST→QWEN] #%d  text: '%s'" % (n, text))
            return original_put(item, *args, **kwargs)
        return _put

    # ------------------------------------------------------------------
    # Boundary E — ui_queue puts (Qwen + FastProcessor → demo loop)
    # ------------------------------------------------------------------
    def wrap_ui_q_put(self, original_put):
        """Returns a replacement for ui_queue.put that logs event types."""
        mon = self
        def _put(item, *args, **kwargs):
            msg_type = item.get("type", "unknown") if isinstance(item, dict) else "unknown"
            with mon._lock:
                mon.ui_puts += 1
                changed = (msg_type != mon._last_nlp_event_type)
                if changed:
                    mon._last_nlp_event_type = msg_type
            if changed or msg_type == "nlp_update":
                snippet = ""
                if msg_type == "nlp_update":
                    state = item.get("state", {})
                    seg = state.get("text_segment", "")[:60]
                    snippet = " | text_segment='%s'" % seg
                print("[UI_Q] type='%s'%s" % (msg_type, snippet))
            return original_put(item, *args, **kwargs)
        return _put

    # ------------------------------------------------------------------
    # Heartbeat — call from main NLP thread loop (non-blocking)
    # ------------------------------------------------------------------
    def heartbeat(self):
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_hb
            if elapsed < 1.0:
                return
            self._last_hb = now
            mic_chunks  = self.mic_chunks
            mic_samples = self.mic_samples
            mic_rms_avg = (self.mic_rms_acc / self.mic_rms_n) if self.mic_rms_n else 0.0
            wh_consumed = self.wh_chunks_consumed
            wh_part     = self.wh_partials
            wh_fin      = self.wh_finals
            nlp_puts    = self.nlp_q_puts
            ui_puts     = self.ui_puts

        print(
            "[MIC]     chunks=%d  samples=%d  avg_rms=%.5f" % (mic_chunks, mic_samples, mic_rms_avg)
        )
        print(
            "[WHISPER] consumed=%d  partials=%d  finals=%d" % (wh_consumed, wh_part, wh_fin)
        )
        print(
            "[PIPELINE] fast→qwen_puts=%d  ui_q_puts=%d" % (nlp_puts, ui_puts)
        )


# ---------------------------------------------------------------------------
# NLP pipeline thread
# ---------------------------------------------------------------------------

def _nlp_thread(
    mic_device: int,
    model_name: str,
    runtime: MultimodalRuntime,
    live_state: LiveState,
    stop_event: threading.Event,
) -> None:
    """
    Runs the existing MicrophoneInput → WhisperTranscriber → FastProcessor
    → QwenExtractor pipeline in a background thread.
    For each finalised NLP JSON, calls runtime.process_nlp_update().
    """
    text_queue: queue.Queue = queue.Queue()
    nlp_queue:  queue.Queue = queue.Queue()
    ui_queue:   queue.Queue = queue.Queue()

    print("[NLP] Initialising microphone (device %d)…" % mic_device)
    mic = MicrophoneInput(device_index=mic_device)

    print("[NLP] Loading faster-whisper (%s)…" % DEFAULT_WHISPER_MODEL)
    asr = WhisperTranscriber(
        model_size=DEFAULT_WHISPER_MODEL,
        device=DEFAULT_WHISPER_DEVICE,
        compute_type=DEFAULT_WHISPER_COMPUTE_TYPE,
        device_index=mic_device,
    )

    fast_proc = FastProcessor()

    print("[NLP] Loading QwenExtractor (%s)…" % model_name)
    deep_proc = QwenExtractor(model_name=model_name)

    # ── Attach diagnostic probes BEFORE starting threads ─────────────
    diag = _DiagMonitor()

    # Probe A: count what MicrophoneInput.callback puts into audio_queue
    mic.audio_queue.put = diag.wrap_mic_put(mic.audio_queue.put)

    mic.start_listening()

    # Probe B: count what Whisper reads from audio_queue
    mic.audio_queue.get = diag.wrap_audio_q_get(mic.audio_queue.get)

    # Probe C: count/log what Whisper puts into text_queue
    text_queue.put = diag.wrap_text_q_put(text_queue.put)

    asr.start_transcribing(mic.audio_queue, text_queue)

    # Probe D: count/log what FastProcessor puts into nlp_queue
    nlp_queue.put = diag.wrap_nlp_q_put(nlp_queue.put)

    fast_proc.start_processing(text_queue, nlp_queue, ui_queue)

    # Probe E: count/log what Qwen/FastProcessor puts into ui_queue
    ui_queue.put = diag.wrap_ui_q_put(ui_queue.put)

    deep_proc.start_processing(nlp_queue, ui_queue)

    print("[NLP] All sub-threads running — listening for speech…")
    print("[DIAG] Boundary probes active — heartbeat every 1s.")

    try:
        while not stop_event.is_set():
            # Diagnostic heartbeat (once per second, non-blocking)
            diag.heartbeat()

            try:
                msg = ui_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "transcript_update":
                live_state.update_transcript(
                    msg.get("text", ""),
                    msg.get("is_partial", True),
                )

            elif msg_type == "nlp_update":
                # QwenExtractor._process_loop emits {"type": "nlp_update", "state": {...}}
                raw = msg.get("state", {})
                if not raw:
                    continue

                live_state.update_nlp(raw)

                try:
                    plan = runtime.process_nlp_update(json.dumps(raw))
                    if plan is not None:
                        live_state.update_plan(plan)
                except Exception as exc:
                    logger.warning("Integration error: %s", exc)

    finally:
        mic.stop_listening()
        asr.stop_transcribing()
        fast_proc.stop_processing()
        deep_proc.stop_processing()
        print("[NLP] Thread shut down.")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _shadow_text(frame, text: str, x: int, y: int, color, scale=_FS_SMALL, thick=_FT):
    cv2.putText(frame, text, (x + 1, y + 1), _FONT, scale, _C_SHADOW, thick + 1, _LT)
    cv2.putText(frame, text, (x, y),          _FONT, scale, color,     thick,     _LT)


def _draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_length=15):
    """Same logic as in the existing main.py — not modifying that file."""
    x1, y1 = pt1
    x2, y2 = pt2
    lines = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for start, end in lines:
        length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        dashes = max(1, int(length / dash_length))
        dx = (end[0] - start[0]) / dashes
        dy = (end[1] - start[1]) / dashes
        for i in range(0, dashes, 2):
            p1 = (int(start[0] + i * dx), int(start[1] + i * dy))
            p2_x = start[0] + (i + 1) * dx
            p2_y = start[1] + (i + 1) * dy
            if i + 1 >= dashes:
                p2_x, p2_y = end[0], end[1]
            cv2.line(img, p1, (int(p2_x), int(p2_y)), color, thickness, _LT)


def _draw_visual_opportunity(frame, scene_state) -> None:
    """
    Draws the EXISTING yellow VisualOpportunity dashed rectangle.
    Comes directly from scene_state.visual_opportunity — the detector is
    completely unchanged.
    """
    if scene_state is None:
        return
    opp = scene_state.visual_opportunity
    if opp and opp.available:
        h, w = frame.shape[:2]
        x1 = int(opp.bbox[0] * w)
        y1 = int(opp.bbox[1] * h)
        x2 = int(opp.bbox[2] * w)
        y2 = int(opp.bbox[3] * h)
        _draw_dashed_rect(frame, (x1, y1), (x2, y2), _C_OPPORTUNITY, thickness=2)
        label = opp.type.value.upper() if opp.type else "OPPORTUNITY"
        _shadow_text(frame, label, x1, max(y1 - 6, 12), _C_OPPORTUNITY, scale=_FS_SMALL)


def _draw_integration_hud(frame, transcript, is_partial, intent, intent_conf, entities, plan) -> None:
    """
    Right-side HUD panel:
        NLP: transcript / intent / entities
        MULTIMODAL: action / concept / visual_id / anchor / confidence
    """
    h, w = frame.shape[:2]

    panel_w = 310
    panel_x = w - panel_w - 8
    panel_y = 10
    line_h  = _LH

    # Count lines
    transcript_lines = _wrap(transcript or "— (listening…)", 38)[:3]
    entity_lines     = min(len(entities), 3)
    n_lines = (
        1               # NLP header
        + len(transcript_lines)
        + 2             # intent + entities header
        + entity_lines
        + 1             # spacer
        + 1             # MULTIMODAL header
        + 6             # action/concept/visual_id/anchor/conf/reason
    )
    panel_h = n_lines * line_h + 16

    # Semi-transparent dark background
    y0 = panel_y
    y1 = min(panel_y + panel_h, h)
    x0 = panel_x
    x1 = min(panel_x + panel_w, w)
    if y1 > y0 and x1 > x0:
        frame[y0:y1, x0:x1] = (frame[y0:y1, x0:x1] * 0.32).astype(frame.dtype)

    x = panel_x + 8
    y = panel_y + line_h

    # ── NLP ──
    _shadow_text(frame, "─── NLP ───────────────────", x, y, _C_NLP_PANEL, scale=_FS_LABEL)
    y += line_h

    partial_tag = " [partial]" if is_partial else ""
    display = (transcript + partial_tag).strip() if transcript else "— (listening…)"
    for line in _wrap(display, 38)[:3]:
        _shadow_text(frame, line, x, y, _C_TEXT, scale=_FS_SMALL)
        y += line_h

    # Intent
    intent_str = "Intent: %s" % intent
    if intent_conf > 0:
        intent_str += " (%.0f%%)" % (intent_conf * 100)
    _shadow_text(frame, intent_str, x, y, _C_NLP_PANEL)
    y += line_h

    # Entities
    if entities:
        _shadow_text(frame, "Entities:", x, y, _C_NLP_PANEL, scale=_FS_SMALL)
        y += line_h
        for ent in entities[:3]:
            name = ent.get("name", ent.get("id", "?"))
            cls  = ent.get("class", "")
            stat = ent.get("status", "")
            _shadow_text(frame, "  %s [%s] %s" % (name, cls, stat), x, y, _C_TEXT, scale=_FS_SMALL)
            y += line_h
    else:
        _shadow_text(frame, "Entities: —", x, y, _C_NLP_PANEL, scale=_FS_SMALL)
        y += line_h

    y += 4  # spacer

    # ── MULTIMODAL ──
    _shadow_text(frame, "─── MULTIMODAL ─────────────", x, y, _C_FUSION_PANEL, scale=_FS_LABEL)
    y += line_h

    if plan is None:
        _shadow_text(frame, "Waiting for first plan…", x, y, _C_NONE, scale=_FS_SMALL)
        return

    action_val   = plan.action.value if plan.action else "—"
    action_color = _C_ACTION if plan.action not in (ActionType.NONE, None) else _C_NONE
    _shadow_text(frame, "Action:     %s" % action_val, x, y, action_color)
    y += line_h

    _shadow_text(frame, "Concept:    %s" % (plan.concept or "—"), x, y, _C_FUSION_PANEL)
    y += line_h

    _shadow_text(frame, "Visual ID:  %s" % (plan.visual_id or "—"), x, y, _C_FUSION_PANEL)
    y += line_h

    anchor_val = "—"
    if plan.spatial:
        anchor_val = plan.spatial.anchor_type.value if plan.spatial.anchor_type else "—"
    _shadow_text(frame, "Anchor:     %s" % anchor_val, x, y, _C_FUSION_PANEL)
    y += line_h

    conf_val = "—"
    if plan.confidence:
        conf_val = "%.0f%%" % (plan.confidence.overall * 100)
    _shadow_text(frame, "Confidence: %s" % conf_val, x, y, _C_FUSION_PANEL)
    y += line_h

    reason = (plan.reason or "")[:40]
    _shadow_text(frame, "Reason: %s" % reason, x, y, _C_NONE, scale=_FS_SMALL)


def _wrap(text: str, width: int) -> list:
    """Word-wrap text to lines of at most `width` characters."""
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        if len(cur) + len(word) + 1 > width:
            lines.append(cur.strip())
            cur = word
        else:
            cur += " " + word
    if cur:
        lines.append(cur.strip())
    return lines if lines else [""]


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(
    device: int = 0,
    backend_str: str = "dshow",
    mic_device: int = DEFAULT_MIC_DEVICE_INDEX,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    timeout: float = 0.0,
) -> int:

    backend_map = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
    backend_flag = backend_map.get(backend_str.lower(), cv2.CAP_DSHOW)

    # ── 1. Open camera ────────────────────────────────────────────
    cam = Camera(device_index=device, backend=backend_flag)
    try:
        cam.open()
    except CameraError as exc:
        print("[VISION] Camera error: %s" % exc)
        return 1

    reader = AsyncCameraReader(cam)
    reader.start()

    # ── 2. Vision perception pipeline ────────────────────────────
    pipeline: Optional[SceneBuilder] = None
    try:
        pipeline = SceneBuilder(yolo_interval_ms=100.0)
        pipeline.open()
        print("[VISION] SceneBuilder ready.")
    except Exception as exc:
        print("[VISION] Warning: perception failed to init (%s). Running without." % exc)

    # ── 3. Integration runtime ────────────────────────────────────
    live_state = LiveState()
    runtime = MultimodalRuntime(on_plan_emitted=live_state.update_plan)
    print("[INTEGRATION] MultimodalRuntime ready.")

    # ── 4. NLP pipeline in background thread ─────────────────────
    stop_event = threading.Event()
    nlp_t = threading.Thread(
        target=_nlp_thread,
        args=(mic_device, model_name, runtime, live_state, stop_event),
        daemon=True,
        name="nlp-pipeline",
    )
    nlp_t.start()

    print()
    print("=" * 60)
    print("  Magic Mirror — NLP + Vision Integration Demo")
    print("=" * 60)
    print("  Speak naturally into the microphone.")
    print("  Example: 'Imagine a dog on my hand'")
    print("  Then:    'Move it there' (while pointing)")
    print("  HUD panel (right) shows live NLP + plan data.")
    print("  Press Q / ESC to quit.")
    print("=" * 60)

    # ── 5. Display loop ───────────────────────────────────────────
    last_scene = None
    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    start_time = time.monotonic()

    try:
        while True:
            if timeout > 0.0 and (time.monotonic() - start_time) >= timeout:
                break

            frame_data = reader.read_latest(timeout=0.2)
            if frame_data is None:
                continue

            cv2.flip(frame_data.image, 1, dst=frame_data.image)

            if pipeline is not None:
                try:
                    result, *_ = pipeline.process(frame_data)
                    last_scene = result
                    if last_scene is not None:
                        runtime.process_vision_update(last_scene)
                except Exception as exc:
                    logger.warning("Perception error: %s", exc)

            # Existing yellow VisualOpportunity rectangle
            _draw_visual_opportunity(frame_data.image, last_scene)

            # Integration HUD
            transcript, is_partial, intent, intent_conf, entities, plan = live_state.snapshot()
            _draw_integration_hud(
                frame_data.image,
                transcript, is_partial,
                intent, intent_conf,
                entities, plan,
            )

            cv2.imshow(_WINDOW, frame_data.image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except cv2.error as exc:
        logger.error("OpenCV error: %s", exc)
        return 1
    finally:
        stop_event.set()
        if pipeline is not None:
            pipeline.close()
        reader.stop()
        cv2.destroyAllWindows()
        print("Demo shutdown complete.")

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Magic Mirror — NLP + Vision Integration Live Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device",     type=int,   default=0,
                   help="Camera device index")
    p.add_argument("--backend",    choices=["dshow", "msmf", "any"], default="dshow")
    p.add_argument("--mic-device", type=int,   default=DEFAULT_MIC_DEVICE_INDEX,
                   dest="mic_device", help="Microphone device index")
    p.add_argument("--model",      type=str,   default=DEFAULT_OLLAMA_MODEL,
                   help="Ollama model name for QwenExtractor")
    p.add_argument("--timeout",    type=float, default=0.0,
                   help="Auto-quit after N seconds (0=forever)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    sys.exit(run(
        device=args.device,
        backend_str=args.backend,
        mic_device=args.mic_device,
        model_name=args.model,
        timeout=args.timeout,
    ))
