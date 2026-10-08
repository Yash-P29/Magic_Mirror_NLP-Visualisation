# Magic Mirror - NLP + Visualisation Integration

This repository hosts the live integration of the Magic Mirror NLP pipeline and the Vision subsystem. It leverages local AI processing (Qwen/Gemini, Faster Whisper) and multimodal computer vision (Mediapipe, YOLO) to create a conversational, visually-aware AR interface.

## Prerequisites

- **Python 3.10+**
- An active microphone device.
- A connected webcam.
- [Ollama](https://ollama.com/) installed and running locally with the `qwen2.5:1.5b` model (or the model specified in your config). 

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yash-P29/Magic_Mirror_NLP-Visualisation.git
   cd Magic_Mirror_NLP-Visualisation
   ```

2. **Install the dependencies:**
   It is highly recommended to use a Python virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Ollama is running:**
   ```bash
   ollama serve
   ```
   *Make sure you have pulled the required model in ollama:* `ollama pull qwen2.5:1.5b`

## Running the Live Demo

To run the live multimodal integration demo (combining live camera vision with live streaming NLP):

```bash
python integration_live_demo.py
```

### Usage
- Speak naturally into the microphone. 
- **Example Flow:** Say *"Imagine a dog on my hand"*, then say *"Move it there"* (while pointing).
- A Heads-Up Display (HUD) panel will appear on the camera feed showing the live NLP and multimodal planning data.
- The pipeline will detect silence automatically, and avoid processing audio when no one is speaking.
- Press **`Q`** or **`ESC`** to exit the demo.

## Running Tests

To execute the automated unit and integration tests, run the following from the root directory:

```bash
python -m pytest integration/tests/test_integration.py -v
python -m pytest magic_mirror_nlp/nlp/sandbox/tests/test_whisper.py -v
```
