from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

MODEL_DIR = Path("models/mms-tts-hun")

if not MODEL_DIR.exists():
    print('asdasd')
    raise FileNotFoundError(
        f"Local model directory not found: {MODEL_DIR}\n"
        "Run download_mms_hu.py first."
    )

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

model = VitsModel.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
).to(device)

text = "Szia! Ez egy teljesen helyi magyar beszédszintézis teszt."
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    waveform = model(**inputs).waveform

audio = waveform.cpu().float().numpy().squeeze()
sf.write("test_hu_local.wav", audio, model.config.sampling_rate)

print("Saved: test_hu_local.wav")
print("Sampling rate:", model.config.sampling_rate)