from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity


class SentenceSimilarity:
    def __init__(self, device):
        self.model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens', device=device)

    def encoded_text(self, texts_1, texts_2):
        encode_texts_1 = self.model.encode(texts_1, batch_size= 512)
        encode_texts_2 = self.model.encode(texts_2, batch_size= 512)

    def cosine_similarity(self, text_1, text_2):
        encode_1 = self.model.encode(text_1)
        encode_2 = self.model.encode(text_2)
        return float(cosine_similarity([encode_1], [encode_2])[0][0])
