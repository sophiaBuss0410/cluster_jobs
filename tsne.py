from sklearn.manifold import TSNE

def tsne(embedding_df):
    tsne = TSNE(n_components=3, perplexity=10, random_state=42)
    X_tsne = tsne.fit_transform(embedding_df)
    return X_tsne


from sklearn.decomposition import PCA

def pca(embedding_df):
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(embedding_df)
    return X_pca
