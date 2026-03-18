from pathlib import Path

from transformers import AutoTokenizer, VitsModel

MODEL_ID = "facebook/mms-tts-hun"
TARGET_DIR = Path("models/mms-tts-hun")

print(f"Downloading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Downloading model: {MODEL_ID}")
model = VitsModel.from_pretrained(MODEL_ID)

TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"Saving tokenizer to: {TARGET_DIR}")
tokenizer.save_pretrained(TARGET_DIR)

print(f"Saving model to: {TARGET_DIR}")
model.save_pretrained(TARGET_DIR)

print("Done.")