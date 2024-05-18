import torch
from custom_rubert import RubertTinyClassifier
from config import TOKENIZER_PATH, MODEL_PATH
from transformers import AutoTokenizer
import json


class ResourceClassifier:
	def __init__(self):
		if torch.cuda.is_avalable():
			device = "cuda"
		else:
			device = "cpu"

		self.tokenizer = AutoTokenizer(TOKENIZER_PATH)
		
		self.rubert = RubertTinyClassifier(2309).to(device).eval()
		self.rubert.load_state_dict(torch.load(MODEL_PATH))
		with open('classes.json', "r", encoding="utf-8") as f:
		    self.classes = json.load(f)


	def __call__(self, batch: list):
		data = self.tokenizer(batch)
		self.rubert