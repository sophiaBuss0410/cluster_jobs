from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS

def mde(embedding_df):
    # Calculate cosine distance from cosine similarity
    similarity_matrix = cosine_similarity(embedding_df)
    distance_matrix = 1 - similarity_matrix # cosine distance

    # Reduction objects
    manifold3D = MDS(n_components=3, dissimilarity='precomputed', random_state=42)

    # compressed data in 2 dimensions
    manifold_data3D = manifold3D.fit_transform(distance_matrix)
    return manifold_data3D
