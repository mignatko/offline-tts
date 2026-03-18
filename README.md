# Offline Hungarian Text-to-Speech

This project converts Hungarian text into MP3 files using locally stored text-to-speech models. The main goal is offline synthesis, with the downloaded models kept inside the `models/` folder so the project can run without fetching model files at runtime.

The current implementation supports three engines:

- `mms` for fully local Hugging Face MMS Hungarian TTS
- `piper` for local Piper ONNX voices
- `xtts` for local XTTS-v2 voice cloning with a reference speaker audio file

## Project Structure

```text
text_to_speech/
├── txt_to_audio.py                  # Main CLI entrypoint
├── input/                           # Input text and reference audio samples
├── models/
│   ├── mms-tts-hun/                 # Local MMS Hungarian model
│   ├── piper/
│   │   ├── anna/hu/...              # Local Piper Hungarian voice
│   │   └── imre/hu/...              # Local Piper Hungarian voice
│   └── xtts-v2/                     # Local XTTS-v2 model files
├── scripts/
│   ├── download_mms_hu.py           # Downloads MMS Hungarian model
│   ├── download_xtts_v2.py          # Downloads XTTS-v2 model
│   └── test_mms_hu_local.py         # Simple local MMS test
└── tts_engines/
    ├── base.py
    ├── mms_engine.py
    ├── piper_engine.py
    └── xtts_engine.py
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

Useful commands:

```bash
black .
ruff check .
mypy .
pytest
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
