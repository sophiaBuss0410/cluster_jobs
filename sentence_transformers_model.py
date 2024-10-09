from sentence_transformers import SentenceTransformer

class LanguageModel():
    def __init__(self, model_name) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        encoded_data = self.model.encode(texts, show_progress_bar=True)
        if len(texts) == 1:
            return encoded_data.reshape(1,-1)
        return encoded_data