from helpers import read_config, singleton

from sentence_transformers_model import LanguageModel
import pickle


@singleton
class KnnClassifier():
    def __init__(self, visualization_method, config) -> None:
        self.model = pickle.load(open(config["inference"]["knn_model_path"], 'rb'))
        self.embedder = LanguageModel(config['sentence_transformers']['model'], visualization_method)

    
    def predict(self, text_embedding):
        if isinstance(text_embedding, str):
            text_embedding = self.embedder.encode([text_embedding])
        label =  self.model.predict(text_embedding)
        return label
    
    def predict_proba(self, text_embedding):
        if isinstance(text_embedding, str):
            text_embedding = self.embedder.encode([text_embedding])
        label =  self.model.predict_proba(text_embedding)
        return label


def predict_label(text):
    config = read_config()
    model = KnnClassifier(config)
    return model.predict(text)

if __name__ ==  "__main__":
    description = input("Write the description: ")
    print(predict_label(description))