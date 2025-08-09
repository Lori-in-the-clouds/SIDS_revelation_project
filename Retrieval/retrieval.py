import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class ImageRetrieval:

    def __init__(self, embedding_path, keypoints_path,image_dataset_path):
        self.df = self.extract_df(embedding_path, keypoints_path)  # DataFrame completo
        self.embeddings = self.df.loc[:, 'embedding_0':'embedding_27'].to_numpy()
        self.labels = self.df['label'].to_numpy()
        self.image_paths = self.df['image_path'].to_list()
        self.embeddings_norm = self.normalize_embeddings(self.embeddings)
        self.nbrs = None  # sarà l’indice per nearest neighbor, costruito dopo
        self.image_dataset_path = image_dataset_path

    def extract_df(self, embedding_path, keypoints_path):

        df = pd.read_csv(embedding_path, header=None)
        kpt = np.load(keypoints_path, allow_pickle=True)

        image_paths = []

        for kpt in kpt:
            if "file_path" in kpt:
                image_paths.append(kpt["file_path"])
            else:
                image_paths.append("unknown")

        df['image_path'] = image_paths
        df.rename(columns={28: "label"}, inplace=True)

        for i in range(28):
            df.rename(columns={i: f"embedding_{i}"}, inplace=True)

        return df

    def normalize_embeddings(self, embeddings):
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def build_index(self, metric='euclidean', k=5):
        self.nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(self.embeddings_norm)

    def retrieve_similar(self, idx_query, k=5, verbose=True):

        embeddings = self.df.loc[:, 'embedding_0':'embedding_27'].to_numpy()
        labels = self.df['label'].to_numpy()
        image_paths = self.df['image_path'].to_list()

        embeddings_norm = self.embeddings_norm

        nbrs = self.build_index(metric='euclidean', k=k)

        distances, indices = self.nbrs.kneighbors(embeddings_norm[idx_query].reshape(1, -1))
        indices = indices[0][1:]  # escludi se stesso

        if verbose:
            print(f"Query image: {image_paths[idx_query]} (label: {labels[idx_query]})")
            print(f"Top {k} similar images:")

        image_paths_similar = []


        for rank, i in enumerate(indices, start=1):
            if verbose:
                print(f"{rank}. {image_paths[i]} - label: {labels[i]} - distance: {distances[0][rank]:.4f}")
            image_paths_similar.append(image_paths[i])

        return distances, image_paths_similar

    def show_images(self,image_paths_similar, n_cols=5, figsize=(15, 5)):

        os.chdir(self.image_dataset_path)

        n_images = len(image_paths_similar)
        n_rows = (n_images + n_cols - 1) // n_cols  # numero di righe necessario

        plt.figure(figsize=figsize)

        for i, path in enumerate(image_paths_similar):
            img = mpimg.imread(path)
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Image {i + 1}")

        plt.tight_layout()
        plt.show()

    '''METRICS'''
    def precision_retrieval_at_k(self,verbose=True):
        correct_counts = []
        for i in range(len(self.embeddings_norm)):
            distances, indices = self.nbrs.kneighbors(self.embeddings_norm[i].reshape(1, -1))
            neighbors = indices[0][1:]  # escludi se stesso
            neighbor_labels = self.labels[neighbors]
            correct_counts.append(np.mean(neighbor_labels == self.labels[i]))

        avg_accuracy = np.mean(correct_counts)
        if verbose:
            print(f"Average retrieval accuracy: {avg_accuracy:.3f}")





