import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples


class ImageRetrieval:

    def __init__(self, embeddings, y, image_paths, image_dataset_path, classes_bs):
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
        self.image_dataset_path = image_dataset_path
        self.classes_bs = classes_bs

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

    def retrieve_similar(self, idx_query, k=5, verbose=True, external_embeddings=False, external_embd=None):
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
        embeddings = self.embeddings
        labels = self.df['label'].to_numpy()
        image_paths = self.df['image_path'].to_list()

        embeddings_norm = self.embeddings_norm

        if external_embeddings:
            nbrs = self.build_index(metric='euclidean', k=k - 1)
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

    def show_images(self, image_paths_similar, n_cols=5, figsize=(15, 5)):

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

    def precision_at_k(self, k=5, verbose=False):
        correct_counts = []
        for i in range(len(self.embeddings_norm)):
            distances, indices = self.nbrs.kneighbors(self.embeddings_norm[i].reshape(1, -1), n_neighbors=k + 1)
            neighbors = indices[0][1:]  # escludi se stesso
            neighbor_labels = self.labels[neighbors]
            correct_counts.append(np.mean(neighbor_labels == self.labels[i]))

        avg_accuracy = np.mean(correct_counts)
        if verbose:
            print(f"Average retrieval accuracy at {k}: {avg_accuracy:.3f}")

        return avg_accuracy

    def plot_precision_at_k(self, k_values=None):
        if k_values is None:
            k_values = [5, 10, 20, 50, 100]

        # evaluate precisions
        precisions = []
        for k in k_values:
            precision = self.precision_at_k(k=k, verbose=False)
            precisions.append(precision)

        # plot
        plt.figure(figsize=(7, 5))
        plt.plot(k_values, precisions, marker="o", color="blue", linewidth=2)
        plt.title("Precision at different k", fontsize=14)
        plt.xlabel("k", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.xticks(k_values)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

        return precisions

    def recall_at_k(self, k=5, verbose=False):
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

    def recall_at_R(self, verbose=False):
        recalls = []
        n_samples = len(self.embeddings_norm)

        for i in range(n_samples):
            # Numero totale di relevant per questo sample
            y_true = self.labels[i]
            total_relevant = np.sum(self.labels == y_true) - 1  # escludo se stesso
            if total_relevant == 0:
                continue  # classe con un solo campione → skip

            # Recupera tutti i vicini possibili (N-1)
            distances, indices = self.nbrs.kneighbors(
                self.embeddings_norm[i].reshape(1, -1),
                n_neighbors=n_samples
            )
            neighbors = indices[0][1:]  # escludo se stesso

            # Prendi solo i primi R vicini
            top_R_neighbors = neighbors[:total_relevant]
            neighbor_labels = self.labels[top_R_neighbors]

            # Quanti tra i primi R hanno la stessa label
            relevant_found = np.sum(neighbor_labels == y_true)

            # Recall@R per questo sample
            recall = relevant_found / total_relevant
            recalls.append(recall)

        avg_recall = np.mean(recalls)
        if verbose:
            print(f"Recall at R: {avg_recall:.3f}")

        return avg_recall

    def plot_silhouette_per_class(self, metric="euclidean", verbose=False):
        n_samples = len(self.embeddings_norm)
        distances, indices = self.nbrs.kneighbors(self.embeddings_norm, n_neighbors=n_samples)
        dist_matrix = distances[:, 1:]  # shape: N x (N-1)
        dist_matrix_full = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            # inserisci le distanze nei rispettivi posti
            neighbors = indices[i, 1:]  # escludi se stesso
            dist_matrix_full[i, neighbors] = distances[i, 1:]
        np.fill_diagonal(dist_matrix_full, 0)

        sample_scores = silhouette_samples(
            X=dist_matrix_full,
            labels=self.labels,
            metric='precomputed'
        )
        silhouette_score_value = sample_scores.mean()
        classes = np.unique(self.labels)

        plt.figure(figsize=(8, 5))

        for cls in classes:
            cls_scores = sample_scores[self.labels == cls]
            plt.hist(cls_scores, bins=20, alpha=0.6,
                     label=f"Class {next((k for k, v in self.classes_bs.items() if v == cls), None)}", density=False)

        plt.axvline(sample_scores.mean(), color="red", linestyle="--", label=f"Mean = {silhouette_score_value:.3f}")
        plt.xlabel("Silhouette coefficient")
        plt.ylabel("Numero di campioni")
        plt.title("Distribuzione Silhouette score per classe")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

        if verbose:
            print(f"Silhouette score ({metric}): {silhouette_score_value:.3f}")

    def plot_tsne(self):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(self.embeddings)

        plt.figure(figsize=(8, 6))

        # maschere per le due classi
        mask_safe = (self.labels == self.classes_bs["baby_safe"])
        mask_unsafe = (self.labels == self.classes_bs["baby_unsafe"])

        # scatter separati per avere legenda chiara
        plt.scatter(X_tsne[mask_safe, 0], X_tsne[mask_safe, 1],
                    c="blue", alpha=0.7, label="baby_safe")
        plt.scatter(X_tsne[mask_unsafe, 0], X_tsne[mask_unsafe, 1],
                    c="red", alpha=0.7, label="baby_unsafe")

        plt.legend()
        plt.title(f"t-SNE degli embedding ({self.embeddings.shape[1]}D → 2D)")
        plt.show()

    def plot_umap(self):
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(self.embeddings)
        plt.figure(figsize=(8, 6))

        # maschere per le due classi
        mask_safe = (self.labels == self.classes_bs["baby_safe"])
        mask_unsafe = (self.labels == self.classes_bs["baby_unsafe"])

        # scatter separati per avere legenda chiara
        plt.scatter(X_umap[mask_safe, 0], X_umap[mask_safe, 1],
                    c="blue", alpha=0.7, label="baby_safe")
        plt.scatter(X_umap[mask_unsafe, 0], X_umap[mask_unsafe, 1],
                    c="red", alpha=0.3, label="baby_unsafe")

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
        mask_safe = (self.labels == self.classes_bs["baby_safe"])
        mask_unsafe = (self.labels == self.classes_bs["baby_unsafe"])

        plt.hist(X_lda[mask_safe], bins=30, alpha=0.7, color="blue",
                 label="baby_safe")
        plt.hist(X_lda[mask_unsafe], bins=30, alpha=0.7, color="red",
                 label="baby_unsafe")

        plt.xlabel("LDA Component 1")
        plt.ylabel("Frequency")
        plt.title(f"Distribuzione LDA degli embedding ({self.embeddings.shape[1]}D → 1D)")
        plt.legend()
        plt.show()

    def report(self, metric="euclidean"):
        self.build_index(metric, k= len(self.embeddings_norm))

        # plot precisions at different k
        print("Precision at different k:".ljust(90, "-"))
        self.plot_precision_at_k()

        # recall at R
        print()
        print(f"Recall at R".ljust(90, "-"))
        print(f"{self.recall_at_R()}")

        print()
        print(f"Silhouette score".ljust(90, "-"))
        self.plot_silhouette_per_class()

        print()
        print(f"Embeddings distributions".ljust(90, "-"))
        self.plot_lda()
        self.plot_tsne()
        self.plot_umap()
