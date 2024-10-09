from helpers import read_config

from sentence_transformers_model import LanguageModel
import pickle



class KnnClassifier():
    def __init__(self, config) -> None:
        self.model = pickle.load(open(config["inference"]["knn_model_path"], 'rb'))
        self.embedder = LanguageModel(config['sentence_transformers']['model'])

    
    def predict(self, text):
        text_embedding = self.embedder.encode([text])
        label =  self.model.predict(text_embedding)[0]
        return label


def predict_label(text):
    config = read_config()
    model = KnnClassifier(config)
    return model.predict(text)

if __name__ ==  "__main__":
    description = input("Write the description: ")
    print(predict_label(description))