import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class ImageRetrieval:

    def __init__(self, embeddings, y, image_paths):
        """
        Build a pandas DataFrame from embeddings, feature names, and labels.

        Parameters
        ----------

        y : list
            Labels corresponding to each embedding.
        image_paths : list[str], optional
            List of image file paths (same length as X). If None, this column is skipped.

        Returns
        -------
        pandas.DataFrame
            DataFrame with embeddings, labels, and optional image paths.
        """
        self.embeddings = embeddings.values
        self.labels = np.array(y)
        self.image_paths = image_paths
        self.image_dataset_path = "/home/terra/Documents/AI_engineering/SIDS-project/python_project/SIDS_revelation_project/datasets/onback_onstomach_v3"

        df = embeddings.copy()
        df["label"] = y
        df['image_path'] = image_paths
        self.df = df

        self.embeddings_norm = self.normalize_embeddings(self.embeddings)
        self.nbrs = None  # sarà l’indice per nearest neighbor, costruito dopo

    def normalize_embeddings(self, embeddings):
        epsilon = 1e-10
        return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + epsilon)

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
        embeddings =self.embeddings
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

    def plot_tsne(self):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(self.embeddings)


        plt.figure(figsize=(8, 6))

        # maschere per le due classi
        mask_onback = (self.labels == 1)
        mask_onstomach = (self.labels == 2)

        # scatter separati per avere legenda chiara
        plt.scatter(X_tsne[mask_onback, 0], X_tsne[mask_onback, 1],
                    c="blue", alpha=0.7, label="on back")
        plt.scatter(X_tsne[mask_onstomach, 0], X_tsne[mask_onstomach, 1],
                    c="red", alpha=0.7, label="on stomack")

        plt.legend()
        plt.title(f"t-SNE degli embedding ({self.embeddings.shape[1]}D → 2D)")
        plt.show()


    def plot_umap(self):
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(self.embeddings)
        plt.figure(figsize=(8, 6))

        # maschere per le due classi
        mask_onback = (self.labels == 1)
        mask_onstomach = (self.labels == 2)

        # scatter separati per avere legenda chiara
        plt.scatter(X_umap[mask_onback, 0], X_umap[mask_onback, 1],
                    c="blue", alpha=0.7, label="on back")
        plt.scatter(X_umap[mask_onstomach, 0], X_umap[mask_onstomach, 1],
                    c="red", alpha=0.7, label="on stomack")

        plt.legend()
        plt.title(f"UMAP degli embedding ({self.embeddings.shape[1]}D → 2D)")
        plt.show()


    def plot_lda(self):
        labels = np.array(self.labels, dtype=int)

        # LDA con 1 componente (2 classi → 1 dimensione)
        lda = LDA(n_components=1)
        X_lda = lda.fit_transform(self.embeddings, labels)

        plt.figure(figsize=(8, 6))

        # istogrammi separati per le due classi
        plt.hist(X_lda[labels == 1], bins=30, alpha=0.7, color="blue", label="On back (1)")
        plt.hist(X_lda[labels == 2], bins=30, alpha=0.7, color="red", label="On stomach (2)")

        plt.xlabel("LDA Component 1")
        plt.ylabel("Frequency")
        plt.title(f"Distribuzione LDA degli embedding ({self.embeddings.shape[1]}D → 1D)")
        plt.legend()
        plt.show()





