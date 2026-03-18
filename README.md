# Offline Hungarian Text-to-Speech

This project converts Hungarian text into MP3 files using locally stored text-to-speech models. The main goal is offline synthesis, with the downloaded models kept inside the `models/` folder so the project can run without fetching model files at runtime.

> Status: work in progress. The project is already usable, but the package structure, tooling, and engine integrations are still being refined.

The current implementation supports three engines:

- `mms` for fully local Hugging Face MMS Hungarian TTS
- `piper` for local Piper ONNX voices
- `xtts` for local XTTS-v2 voice cloning with a reference speaker audio file

## Project Structure

```text
text_to_speech/
├── txt_to_audio.py                  # Backward-compatible wrapper for the package CLI
├── input/                           # Input text and reference audio samples
├── models/
│   ├── mms-tts-hun/                 # Local MMS Hungarian model
│   ├── piper/
│   │   ├── anna/hu/...              # Local Piper Hungarian voice
│   │   └── imre/hu/...              # Local Piper Hungarian voice
│   └── xtts-v2/                     # Local XTTS-v2 model files
├── scripts/
│   ├── download_mms_hu.py           # CLI wrapper for MMS model download
│   ├── download_xtts_v2.py          # CLI wrapper for XTTS-v2 download
│   └── test_mms_hu_local.py         # Simple local MMS test
├── src/offline_hungarian_tts/
│   ├── cli.py                       # Main CLI implementation
│   ├── pipeline.py                  # Text-to-audio orchestration helpers
│   ├── audio.py                     # ffmpeg/audio utility helpers
│   ├── downloads.py                 # Reusable model download functions
│   └── engines/
│       ├── base.py
│       ├── mms_engine.py
│       ├── piper_engine.py
│       └── xtts_engine.py
└── tests/
    ├── test_cli_and_scripts.py
    ├── test_engines.py
    └── test_txt_to_audio.py
```

## How Input Text Is Parsed

The input file must be UTF-8 text. Empty lines are ignored.

Non-empty lines are treated as alternating entries:

1. line 1 = question
2. line 2 = answer
3. line 3 = new empty line (separator)
4. line 4 = question
5. line 5 = answer

Example:

```text
Mi a neved?
A nevem Anna.

Hány éves vagy?
Öt éves vagyok.
```

This structure matters because the CLI supports separate repeat counts for question lines and answer lines.

## Requirements

- Python 3
- `ffmpeg` available in `PATH`
- Python packages required by the selected engine

This repository now includes a `pyproject.toml` with optional dependency groups.

Minimal install:

```bash
pip install -e .
```

Install with MMS support:

```bash
pip install -e ".[mms]"
```

Install with XTTS support:

```bash
pip install -e ".[xtts]"
```

Install everything, including dev tooling:

```bash
pip install -e ".[all]"
```

Notes:

- `mms` uses `torch`, `transformers`, and `soundfile`
- `xtts` uses `TTS`
- `download` adds `huggingface_hub` for the helper download scripts
- `dev` adds `black`, `ruff`, and `mypy`
- `piper` requires the external `piper` executable to be installed and available in `PATH`, unless you pass `--piper-bin` with an absolute path

## Development Tooling

The repository is configured with:

- `black` for formatting
- `ruff` for linting and import sorting
- `mypy` for type checking
- `pytest` for unit tests
- `pre-commit` for local Git hooks
- GitHub Actions for CI on pushes and pull requests

Useful commands:

```bash
black .
ruff check .
mypy .
pytest
```

Install local hooks:

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push
```

Then you can run all hooks manually with:

```bash
pre-commit run --all-files
```

## Main Script

The main entrypoint is:

```bash
python txt_to_audio.py --engine <mms|piper|xtts> --input <file.txt> --output <file.mp3>
```

Common options:

- `--pause` pause in seconds between spoken items
- `--question-repeats` how many times each odd-numbered non-empty line is repeated
- `--answer-repeats` how many times each even-numbered non-empty line is repeated
- `--speaking-rate` post-process speed multiplier via `ffmpeg`
- `--device auto|mps|cpu` device selection for MMS loading

## Offline Models

The project is built around local model folders:

- MMS default path: `models/mms-tts-hun`
- XTTS default path: `models/xtts-v2`
- Piper models are passed explicitly via `--piper-model`

The code loads MMS and XTTS from disk and uses `local_files_only=True` for MMS. That means the expected workflow is:

1. download models once
2. keep them in `models/`
3. run synthesis locally afterward

Important:

- the `models/` directory is Git-ignored, so a fresh clone will not contain the model files
- you must download or copy the models locally before running synthesis

## Model Setup

### 1. MMS Hungarian model

Install the download dependencies and the MMS runtime:

```bash
pip install -e ".[mms,download]"
```

Download the model into the expected folder:

```bash
python scripts/download_mms_hu.py
```

This creates:

```text
models/mms-tts-hun/
```

### 2. XTTS-v2 model

Install the XTTS runtime and download dependencies:

```bash
pip install -e ".[xtts,download]"
```

Download XTTS-v2 into the expected folder:

```bash
python scripts/download_xtts_v2.py
```

This creates:

```text
models/xtts-v2/
```

XTTS also needs a reference speaker audio file. The repository includes:

```text
input/xtts_reference.wav
```

### 3. Piper Hungarian models

Piper is different from MMS and XTTS in this project:

- the repository does not currently provide a download helper for Piper voices
- you must place the Piper model files in `models/piper/...` yourself
- each voice needs both the `.onnx` model and the matching `.onnx.json` config

Expected examples from this project:

```text
models/piper/anna/hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx
models/piper/anna/hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx.json
models/piper/imre/hu/hu_HU/imre/medium/hu_HU-imre-medium.onnx
models/piper/imre/hu/hu_HU/imre/medium/hu_HU-imre-medium.onnx.json
```

You also need the external `piper` binary installed and available in `PATH`, unless you pass `--piper-bin`.

### 4. Quick verification

After the models are in place, the easiest quick checks are:

```bash
python scripts/test_mms_hu_local.py
python txt_to_audio.py --engine mms --input /path/to/input.txt --output output/test.mp3
```

## Usage Examples

### 1. MMS Hungarian TTS

Uses the local MMS model from `models/mms-tts-hun`.

```bash
python txt_to_audio.py \
  --engine mms \
  --input input/questions.txt \
  --output output/questions_mms.mp3
```

Example with repeats and slower speech:

```bash
python txt_to_audio.py \
  --engine mms \
  --input input/questions.txt \
  --output output/questions_mms.mp3 \
  --pause 1.2 \
  --question-repeats 2 \
  --answer-repeats 3 \
  --speaking-rate 0.95
```

### 2. Piper Hungarian TTS

The repository already contains local Hungarian Piper voices under `models/piper/`.

Example with the `anna` voice:

```bash
python txt_to_audio.py \
  --engine piper \
  --input input/questions.txt \
  --output output/questions_piper_anna.mp3 \
  --piper-model models/piper/anna/hu/hu_HU/anna/medium/hu_HU-anna-medium.onnx
```

Example with the `imre` voice:

```bash
python txt_to_audio.py \
  --engine piper \
  --input input/questions.txt \
  --output output/questions_piper_imre.mp3 \
  --piper-model models/piper/imre/hu/hu_HU/imre/medium/hu_HU-imre-medium.onnx
```

Optional Piper arguments supported by the CLI:

- `--piper-config`
- `--piper-bin`
- `--speaker`
- `--length-scale`
- `--noise-scale`
- `--noise-w-scale`

If `--piper-config` is omitted, the code expects `<model>.json`.

### 3. XTTS-v2 Hungarian Output

XTTS uses a local model directory plus a reference speaker WAV or MP3.

The repository already includes a sample reference file:

- `input/xtts_reference.wav`

Example:

```bash
python txt_to_audio.py \
  --engine xtts \
  --input input/questions.txt \
  --output output/questions_xtts.mp3 \
  --xtts-model-dir models/xtts-v2 \
  --xtts-speaker-wav input/xtts_reference.wav \
  --xtts-language hu
```

## Helper Scripts

- `scripts/download_mms_hu.py`: downloads and saves `facebook/mms-tts-hun` into `models/mms-tts-hun`
- `scripts/download_xtts_v2.py`: downloads and saves `coqui/XTTS-v2` into `models/xtts-v2`
- `scripts/test_mms_hu_local.py`: synthesizes a short Hungarian sentence to `test_hu_local.wav`

## What the Main Script Does

`txt_to_audio.py` performs the following steps:

1. reads UTF-8 input text
2. removes empty lines
3. interprets lines as alternating question/answer items
4. synthesizes each item to WAV using the selected local engine
5. inserts generated silence between segments with `ffmpeg`
6. optionally adjusts speaking speed using an `atempo` filter
7. concatenates everything into a final MP3 using `ffmpeg`

## Notes and Limitations

- Output audio is always written as MP3, even though intermediate files are WAV
- `ffmpeg` is mandatory for silence generation, tempo adjustment, and final MP3 creation
- XTTS currently resolves to CPU in `tts_engines/xtts_engine.py`
- Piper inference depends on an installed `piper` binary; the models alone are not enough
- Downloaded model files under `models/` are ignored by Git to avoid committing large offline assets
