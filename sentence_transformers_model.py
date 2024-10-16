from sentence_transformers import SentenceTransformer
from functools import cached_property
import pandas as pd
import numpy as np
import time
import os

from helpers import singleton, read_config, read_json
from tsne import tsne, pca
from mde import mde

@singleton
class LanguageModel():
    def __init__(self, model_name, visualization_method) -> None:
        self.model = SentenceTransformer(model_name)
        self.df = pd.read_csv("./data/roles_all_w_intern_wo_admin.csv")
        self.roles = self.df["Description"].values
        self.labels = self.df["Role"].values
        self.embeddings_file = "./data/embeddings.txt"
        self.coordinates_file = f"./data/{visualization_method}_coordinates.csv"
        self.visualization_method = visualization_method
        self.rol_to_id_dict = read_json("./data/role_to_id.json")

    def encode(self, texts):
        encoded_data = self.model.encode(texts, show_progress_bar=True)
        if len(texts) == 1:
            return encoded_data.reshape(1,-1)
        return encoded_data
    
    @cached_property
    def roles_encoded(self):
        
        return [self.rol_to_id_dict.get(s, -1) for s in self.labels]
    

    @cached_property
    def get_embeddings(self):
        if os.path.exists(self.embeddings_file):
            return np.loadtxt(self.embeddings_file)
        embeddings =  self.encode(self.roles)
        np.savetxt(self.embeddings_file, embeddings)
        return embeddings
    

    def get_initial_corrdinates(self):
        if os.path.exists(self.coordinates_file):
            return pd.read_csv(self.coordinates_file)
        roles_embedings = self.get_embeddings
        start_time = time.time()
        if self.visualization_method == "tsne":
            coordinates = tsne(roles_embedings)
        if self.visualization_method == "pca":
            coordinates = pca(roles_embedings)
        if self.visualization_method == "mde":
            coordinates = mde(roles_embedings)
        end_time = time.time() - start_time
        print(f"Dimensionality Reduction with {self.visualization_method} took: {end_time}")
        df_coordinates = pd.DataFrame(coordinates, columns=['x', 'y', 'z'])
        df_coordinates['role_id'] = self.roles_encoded
        df_coordinates.to_csv(self.coordinates_file, index=False)
        
        return df_coordinates
    
    def get_corrdinates(self, user_embeddings):
        if isinstance(user_embeddings, str):
            user_embeddings = self.encode(user_embeddings).reshape(1, -1)
        data = np.vstack([self.get_embeddings, user_embeddings])
        start_time = time.time()
        if self.visualization_method == "tsne":
            coordinates = tsne(data)
        if self.visualization_method == "pca":
            coordinates = pca(data)
        if self.visualization_method == "mde":
            coordinates = mde(data)
        end_time = time.time() - start_time
        print(f"Dimensionality Reduction with {self.visualization_method} took: {end_time}")
        roles_coordinates = coordinates[:-1]
        user_coordinates = coordinates[-1]
        return roles_coordinates, user_coordinates


if __name__ == "__main__":
    config = read_config()
    model = LanguageModel(visualization_method = "tsne", model_name = config["sentence_transformers"]["model"])
    df =  model.get_initial_corrdinates()
    print(df)

    # print(roles_coordinates)
    # print(user_coordinates)
