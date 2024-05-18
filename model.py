import torch
from custom_rubert import RubertTinyClassifier
from config import TOKENIZER_PATH, MODEL_PATH
from transformers import AutoTokenizer
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import fasttext.util
import fasttext


class ResourceClassifier:
	def __init__(self):
		if torch.cuda.is_available():
			self.device = "cuda"
		else:
			self.device = "cpu"

		self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
		
		self.rubert = RubertTinyClassifier(2309).to(self.device).eval()
		self.rubert.load_state_dict(torch.load(MODEL_PATH))
		
		with open('classes.json', "r", encoding="utf-8") as f:
		    self.classes = json.load(f)

		self.data = pd.read_csv("./all_embeds.csv")

		fasttext.util.download_model('ru', if_exists='ignore')  
		self.ft_model = fasttext.load_model('cc.ru.300.bin')

		
	def __call__(self, batch: list):
		if type(batch) != list:
			raise Exception("В качестве аргумента передайте список")

		data = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
		with torch.inference_mode():
			logits = self.rubert(**data)
			ids = logits.argmax(dim=1)
			embends = [self.ft_model.get_sentence_vector(line) for line in batch]
			
		titles = [self.classes[i] for i in ids]
		result = []
		for j in range(len(titles)):
			rows = self.data.loc[self.data["Название кластера"] == titles[j]]
			row_value = -1
			row_index = -1
			for i in rows.index:
				if row_value < cosine_similarity([eval(rows["fastextembed"][i])], [embends[j]])[0][0]:
					row_value = cosine_similarity([eval(rows["fastextembed"][i])], [embends[j]])[0][0]
					row_index = i


			if row_value < 0:
				result.append({"status": "error", "text": "Невозможно присвоить"})
			else:
				result.append({"status": "ok", "result": {"label": self.data["Наименование"][row_index], "probability": row_value, "group": titles[j]}, "rec-code": self.data["Код ресурса"][row_index]})
		return result




if __name__ == "__main__":
	model = ResourceClassifier()
	print("-----------------Start-----------------")
	while True:
		print(model([input()]))