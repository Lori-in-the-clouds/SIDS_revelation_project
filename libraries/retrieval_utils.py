import pandas as pd
import numpy as np
import umap
from numpy.linalg import pinv, inv
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_samples

from matplotlib.lines import Line2D

class ImageRetrieval:

    def __init__(self, embeddings, y, image_paths, image_dataset_path, classes_bs, figsize=(5.6,4.2)):
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

        self.figsize = figsize

    def normalize_embeddings(self, embeddings):
        epsilon = 1e-10
        return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + epsilon)

    def build_index(self, metric='euclidean'):
        """
        Build the nearest neighbor index on normalized embeddings.

        Parameters:
            metric (str): Distance metric to use ('euclidean' or 'cosine').
            k (int): Number of neighbors to consider (k+1 to account for self).
        """

        k = len(self.embeddings_norm)-1
        self.nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(self.embeddings_norm)

    def retrieve_similar(self, idx_query, k=5, verbose=True):
        if self.nbrs is None:
            print("ERROR: build indexes and the re-run this function")
            return None, None

        distances, indices = self.nbrs.kneighbors(self.embeddings_norm[idx_query].reshape(1, -1))

        # escludi la query stessa
        distances = distances[0][1:]
        indices = indices[0][1:]

        # prendi solo i primi k
        distances = distances[:k]
        indices = indices[:k]

        if verbose:
            print(f"Query image: {self.df['image_path'].iloc[idx_query]} (label: {self.labels[idx_query]})")
            print(f"Top {k} similar images:")
            for rank, i in enumerate(indices, start=1):
                print(
                    f"{rank}. {self.df['image_path'].iloc[i]} - label: {self.labels[i]} - distance: {distances[rank - 1]:.4f}")

        image_paths_similar = [self.df['image_path'].iloc[i] for i in indices]
        return distances, image_paths_similar

    def show_images(self, image_paths_similar, n_cols=5, figsize=(10, 3)):
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

    def precision_at_k(self, k=None, verbose=False):
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

    def plot_precision_at_k(self, k_values=None, verbose = True):
        if k_values is None:
            k_values = [5, 10, 20, 50, 100]

        # evaluate precisions
        precisions = []
        for k in k_values:
            precision = self.precision_at_k(k=k, verbose=False)
            precisions.append(precision)

        if verbose:
            # plot
            plt.figure(figsize=self.figsize)
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
        if verbose:
            plt.figure(figsize=self.figsize)

            for cls in classes:
                cls_scores = sample_scores[self.labels == cls]
                plt.hist(cls_scores, bins=20, alpha=0.6,
                         label=f"Class {next((k for k, v in self.classes_bs.items() if v == cls), None)}", density=False)

            plt.axvline(sample_scores.mean(), color="red", linestyle="--", label=f"Mean = {silhouette_score_value:.3f}")
            plt.xlabel("Silhouette coefficient")
            plt.xlim(-1, 1)
            plt.ylabel("Numero di campioni")
            plt.title("Distribuzione Silhouette score per classe")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.show()
            print(f"Silhouette score ({metric}): {silhouette_score_value:.3f}")
        return silhouette_score_value

    def plot_tsne(self):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(self.embeddings)

        plt.figure(figsize=self.figsize)

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
        proj = reducer.fit_transform(self.embeddings_norm)

        cmap = plt.colormaps["coolwarm"].resampled(2)
        legend_elements = []
        for label_name, label_idx in self.classes_bs.items():
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(label_idx), markersize=6, label=label_name)
            )

        plt.scatter(proj[:, 0], proj[:, 1], c=self.labels, s=6, cmap=cmap)
        plt.legend(handles=legend_elements, title="Labels", loc="best")
        plt.show()

    def plot_lda(self):
        labels = np.array(self.labels, dtype=int)

        # LDA con 1 componente (2 classi → 1 dimensione)
        lda = LDA(n_components=1)
        X_lda = lda.fit_transform(self.embeddings, labels)

        plt.figure(figsize=self.figsize)

        # istogrammi separati per le due classi
        mask_safe = (self.labels == self.classes_bs["baby_safe"])
        mask_unsafe = (self.labels == self.classes_bs["baby_unsafe"])

        plt.hist(X_lda[mask_safe], bins=30, alpha=0.7, color="blue",
                 label="baby_safe")
        plt.hist(X_lda[mask_unsafe], bins=30, alpha=0.7, color="red",
                 label="baby_unsafe")

        plt.xlabel("LDA Component 1")
        plt.title(f"Distribuzione LDA degli embedding ({self.embeddings.shape[1]}D → 1D)")
        plt.legend()
        plt.show()

    from numpy.linalg import inv, pinv
    from sklearn.covariance import LedoitWolf

    def build_mahalanobis_index(self, pca_dim=None, use_pinv=False):
        """
        Costruisce l'indice NearestNeighbors usando Mahalanobis distance.
        """
        Xc = self.embeddings_norm - self.embeddings_norm.mean(axis=0, keepdims=True)

        if pca_dim is not None:
            from sklearn.decomposition import PCA
            Xc = PCA(n_components=pca_dim).fit_transform(Xc)

        cov = LedoitWolf().fit(Xc).covariance_
        VI = pinv(cov) if use_pinv else inv(cov)

        self.nbrs = NearestNeighbors(metric='mahalanobis', metric_params={'VI': VI})
        self.nbrs.fit(Xc)

        self.mahalanobis_VI = VI  # salva per eventuale uso futuro
        self.embeddings_maha = Xc

    def report(self, metric="euclidean"):
        self.build_index(metric)

        # plot precisions at different k
        print("Precision at different k:".ljust(90, "-"))
        self.plot_precision_at_k()

        # recall at R
        print()
        print(f"Recall at R".ljust(90, "-"))
        print(f"{self.recall_at_R()}")

        print()
        print(f"Silhouette score".ljust(90, "-"))
        self.plot_silhouette_per_class(verbose=True)

        print()
        print(f"Embeddings distributions".ljust(90, "-"))
        self.plot_umap()
