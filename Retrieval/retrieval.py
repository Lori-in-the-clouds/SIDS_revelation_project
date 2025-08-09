import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class ImageRetrieval:

    def __init__(self, embedding_path, keypoints_path,image_dataset_path):
        """
        Initialize the ImageRetrieval object by loading embeddings, labels, and image paths.

        Parameters:
            embedding_path (str): Path to CSV file containing embeddings and labels.
            keypoints_path (str): Path to .npy file containing keypoints with image filenames.
        """
        self.df = self.extract_df(embedding_path, keypoints_path)  # DataFrame completo
        self.embeddings = self.df.loc[:, 'embedding_0':'embedding_27'].to_numpy()
        self.labels = self.df['label'].to_numpy()
        self.image_paths = self.df['image_path'].to_list()
        self.embeddings_norm = self.normalize_embeddings(self.embeddings)
        self.nbrs = None  # sarà l’indice per nearest neighbor, costruito dopo
        self.image_dataset_path = image_dataset_path

    def extract_df(self, embedding_path, keypoints_path):
        """
        Load the embeddings CSV and keypoints npy file, then combine them into a DataFrame.
        Adds image paths to the dataframe from keypoints metadata.

        Parameters:
            embedding_path (str): Path to CSV file with embeddings and labels.
            keypoints_path (str): Path to .npy file with keypoints (contains image filenames).

        Returns:
            pandas.DataFrame: DataFrame with embeddings, labels, and image paths.
        """
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
        """
        Build the nearest neighbor index on normalized embeddings.

        Parameters:
            metric (str): Distance metric to use ('euclidean' or 'cosine').
            k (int): Number of neighbors to consider (k+1 to account for self).
        """
        self.nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(self.embeddings_norm)

    def retrieve_similar(self, idx_query, k=5, verbose=True, external_embeddings=False, external_embd= None):
        """
        Retrieve top-k similar images to the query image indexed by idx_query.

        Parameters:
            idx_query (int): Index of the query image in the dataset.
            k (int): Number of similar images to retrieve.
            verbose (bool): Whether to print detailed results.

        Returns:
            tuple: (distances, image_paths_similar)
                distances: array of distances to retrieved images
                image_paths_similar: list of image file paths retrieved
        """
        embeddings = self.df.loc[:, 'embedding_0':'embedding_27'].to_numpy()
        labels = self.df['label'].to_numpy()
        image_paths = self.df['image_path'].to_list()

        embeddings_norm = self.embeddings_norm

        if external_embeddings:
            nbrs = self.build_index(metric='euclidean', k=k-1)
        else:
            nbrs = self.build_index(metric='euclidean', k=k)

        if external_embeddings == False:
            distances, indices = self.nbrs.kneighbors(embeddings_norm[idx_query].reshape(1, -1))
        else:
            distances, indices = self.nbrs.kneighbors(self.normalize_embeddings(external_embd).reshape(1, -1))

        if external_embeddings == False:
            indices = indices[0][1:]  # escludi se stesso
            distances = distances[0][1:]
        else:
            indices = indices[0]
            distances = distances[0]

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

    def precision_at_k(self,k=5,verbose=True):
        correct_counts = []
        for i in range(len(self.embeddings_norm)):
            distances, indices = self.nbrs.kneighbors(self.embeddings_norm[i].reshape(1, -1), n_neighbors=k + 1)
            neighbors = indices[0][1:]  # escludi se stesso
            neighbor_labels = self.labels[neighbors]
            correct_counts.append(np.mean(neighbor_labels == self.labels[i]))

        avg_accuracy = np.mean(correct_counts)
        if verbose:
            print(f"Average retrieval accuracy at {k}: {avg_accuracy:.3f}")


    def recall_at_k(self, k=5, verbose=True):
        recalls = []
        for i in range(len(self.embeddings_norm)):
            distances, indices = self.nbrs.kneighbors(self.embeddings_norm[i].reshape(1, -1), n_neighbors=k + 1)
            neighbors = indices[0][1:]  # escludi se stesso
            neighbor_labels = self.labels[neighbors]

            y_true = self.labels[i]

            # Numero totale di immagini con label y_true nel dataset (escludendo la query stessa)
            total_relevant = np.sum(self.labels == y_true) - 1

            # Numero di rilevanti trovati tra i k vicini
            relevant_found = np.sum(neighbor_labels == y_true)

            recall = relevant_found / total_relevant if total_relevant > 0 else 0
            recalls.append(recall)

        avg_recall = np.mean(recalls)
        if verbose:
            print(f"Recall at {k}: {avg_recall:.3f}")






