import joblib
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import os
import random
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_metric_learning.losses import SupConLoss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd


def get_model_name(clf_untrained):
    # Ottieni il nome della classe
    class_name = clf_untrained.__class__.__name__
    return class_name

class Classifier:
    def __init__(self, embeddings, y, classes_bs, image_paths= None ,figsize = (5.6, 4.2)):
        self.X = embeddings.values
        self.features = embeddings.columns
        self.len_features = len(embeddings.columns)
        self.images_paths = image_paths if image_paths else self.X

        self.y = np.array(y)
        self.classes_bs = classes_bs
        self.X_train, self.X_test, self.y_train, self.y_test, self.images_paths_train, self.images_paths_test = train_test_split(self.X, self.y,self.images_paths, test_size=0.2, random_state=42)

        self.figsize = figsize
    def logistic_regression(self,verbose =True):
        clf = LogisticRegression(
            penalty="l2",  # Ridge regularization
            C=0.1,  # smaller C = stronger regularization
            solver="liblinear",
            max_iter=10000,
            random_state=42
        )
        result = self.evaluation_pipeline(clf, verbose = verbose)
        return clf, result

    def random_forest(self,verbose=True):
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,              # Ridotto per limitare la complessità del modello
            min_samples_split=20,     # Aumentato per richiedere più campioni per uno split
            min_samples_leaf=10,      # Aumentato per richiedere più campioni per foglia
            max_features="sqrt",      # Ottima scelta, mantiene la diversità tra gli alberi
            bootstrap=True,
            random_state=None
        )
        results = self.evaluation_pipeline(clf,verbose = verbose)
        return clf,results

    def XGBC(self,verbose=True, n_top_features= [25,10], shortAnalysis= False):
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1,
            reg_alpha=0.5,
            random_state=None
        )
        if not shortAnalysis:
            results = self.evaluation_pipeline(clf,verbose = verbose, n_top_features= n_top_features)
            return clf,results
        if shortAnalysis:
            clf_trained = clone(clf)

            clf_trained.fit(self.X_train, self.y_train)
            self.evaluate_metrics(clf_trained)
            return


    def bagging(self,base_clf,verbose=True):
        clf = BaggingClassifier(
            estimator=base_clf,
            n_estimators=100,
            max_samples=0.8,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )
        results = self.evaluation_pipeline(clf,verbose = verbose)
        return clf,results

    def evaluation_pipeline(self,clf_untrained, verbose=True, is_ensemble=False,optimized:bool=False,n_top_features=[10,25]):
        if verbose:
            print("".center(90, '-'))
            print("FIRST ANALYSIS".center(90, '-'))
            print("".center(90, '-'))

        clf_trained = clone(clf_untrained)


        clf_trained.fit(self.X_train, self.y_train)


        project_dir = f"{os.getcwd().split('SIDS_revelation_project')[0]}SIDS_revelation_project/"
        joblib.dump(clf_trained,f"{project_dir}classifiers/{get_model_name(clf_untrained)}_{self.len_features}_features{'_optimized' if optimized else ''}.pkl")

        importances, indices = None, None
        if not is_ensemble:
            importances, indices = self.plot_feature_importance(clf_trained, verbose=verbose)

        if verbose:
            self.plot_learning_curve(clf_trained)
            self.evaluate_metrics(clf_trained)

        results = {
            'all_features': {
                'model': clf_trained,
                'X': self.X,
                'y': self.y
            }
        }

        for n_features in n_top_features:
            if not is_ensemble and importances is not None and len(self.features) > n_features:
                if verbose:
                    print("".center(90, '-'))
                    print(f"TOP {n_features} FEATURES ANALYSIS".center(90, '-'))
                    print("".center(90, '-'))
                    plt.figure(figsize=self.figsize)
                    plt.title(f"Top {n_features} feature importances")
                    plt.bar(range(n_features), importances[indices[:n_features]])
                    plt.xticks(range(n_features), [f"{self.features[i]}" for i in indices[:n_features]], rotation=90)
                    plt.tight_layout()
                    plt.show()

                top_k = n_features
                top_features_idx = indices[:top_k]
                X_train_selected = self.X_train[:, top_features_idx]
                X_test_selected = self.X_test[:, top_features_idx]
                X_selected = self.X[:, top_features_idx]
                clf_retrained = clone(clf_untrained)
                clf_retrained.fit(X_train_selected, self.y_train)


                if verbose:
                    self.plot_learning_curve(clf_retrained, X_selected)
                    self.evaluate_metrics(clf_retrained, X_test_selected)

                results[f"top_{n_features}_features"] = {
                    'model': clf_retrained,
                    'X': X_selected,
                    'y': self.y,
                    'top_features_idx': top_features_idx  # Added this to the dictionary
                }
                joblib.dump(clf_retrained,f"{project_dir}classifiers/{get_model_name(clf_untrained)}_{n_features}_features{'_optimized' if optimized else ''}.pkl")

        return results

    def evaluation_pipeline_with_cv(self, clf_untrained, n_splits=5, verbose=True, is_ensemble=False, optimized=False,
                                    n_top_features=[10, 25]):
        if verbose:
            print("".center(90, '-'))
            print("K-FOLD CROSS-VALIDATION ANALYSIS".center(90, '-'))
            print("".center(90, '-'))

        # Inizializza StratifiedKFold per i problemi di classificazione
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Liste per salvare i risultati di ogni fold
        accuracies = []

        # Loop su ogni fold
        for train_index, test_index in skf.split(self.X, self.y):
            X_train_fold, X_test_fold = self.X[train_index], self.X[test_index]
            y_train_fold, y_test_fold = self.y[train_index], self.y[test_index]

            clf_trained_fold = clone(clf_untrained)
            clf_trained_fold.fit(X_train_fold, y_train_fold)

            y_pred_fold = clf_trained_fold.predict(X_test_fold)
            fold_accuracy = accuracy_score(y_test_fold, y_pred_fold)
            accuracies.append(fold_accuracy)

        avg_accuracy = np.mean(accuracies)
        if verbose:
            print(f"Accuracy for each fold: {accuracies}")
            print(f"Average cross-validation accuracy: {avg_accuracy:.4f}")


        return avg_accuracy


    def evaluate_metrics(self, clf, X_test_selected = None):
        X_test = X_test_selected if X_test_selected is not None else self.X_test

        y_pred = clf.predict(X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        # Evaluate
        print(f"\nDataset labels:----------------------------------------\n{self.classes_bs}\n")
        print(f"Report-------------------------------------------------")
        print(classification_report(self.y_test, y_pred, target_names=self.classes_bs.keys()))

        print(f"Confusion matrix---------------------------------------")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes_bs.keys())
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

    def plot_feature_importance(self, clf, top_k = None,verbose=True):

        importances = None
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])

        if importances is None:
            return None, None

        indices = np.argsort(importances)[::-1]
        n_features = top_k if top_k is not None else len(self.features)

        # Visualize all feature importances
        if verbose:
            plt.figure(figsize=self.figsize)
            plt.title("Ordered feature importances")
            plt.bar(range(n_features), importances[indices[:n_features]])
            plt.xticks(range(n_features), [f"{self.features[i]}" for i in indices[:n_features]], rotation=90)
            plt.tight_layout()

        return importances, indices

    '''
    def plot_learning_curve(self, clf, X_selected= None, verbose=True ):
        X = X_selected if X_selected is not None else self.X

        train_sizes, train_scores, test_scores = learning_curve(
            clf,
            X, self.y,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )

        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)

        if verbose:
            plt.figure(figsize=self.figsize)
            plt.plot(train_sizes, train_mean, 'o-', label="Training accuracy")
            plt.plot(train_sizes, test_mean, 'o-', label="Validation accuracy")
            plt.grid(True)
            plt.xlabel("Training set size")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.title("Learning Curve")
            plt.ylim(0.5, 1)
            plt.show()


        return train_sizes,train_scores,test_scores,train_mean,test_mean
    '''

    def optimize_model(self,model, param_grid,verbose=True):

        print("\nStart random search...")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=200,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(self.X_train, self.y_train)
        # Print the best parameters and the score

        print("\nRandom Search Results:")
        print("Best parameters found: ", random_search.best_params_)
        print("Best mean cross-validation accuracy: ", random_search.best_score_)

        # Evaluate the best model on the test set
        if verbose:
            print("\nEvaluation of the best model on the test set:")

        self.evaluation_pipeline(random_search.best_estimator_,verbose=verbose,optimized=True)
        return random_search.best_params_


    def plot_learning_curve_comparison(self, data_sets, title, ylim=None, cv=None, n_jobs=-1,
                                       train_sizes=np.linspace(.1, 1.0, 5),figsize=(10,6)):
        """
        Generates a graph comparing the learning curves of multiple models,
        each with its own dataset.

        Arguments:
            self: The class instance containing the X and y data.
            data_sets: A list of tuples, where each tuple contains (model, name, X, y).
            title: Title of the graph.
            ylim: Limits of the graph's y-axis.
            cv: Number of folds for cross-validation.
            n_jobs: Number of cores to use.
            train_sizes: Sizes of the training subsets to use for plotting the curve.
        """
        plt.figure(figsize=figsize)
        if title:
            plt.title(title)
        if ylim:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:green', 'tab:blue', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tomato', 'gold', 'lime', 'teal', 'darkviolet']
        line_styles = ['o-', 'o--']

        for i, (estimator, name, X_data, y_data) in enumerate(data_sets):
            estimator_clone = clone(estimator)

            train_sizes_i, train_scores_i, test_scores_i = learning_curve(
                estimator_clone, X_data, y_data, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

            train_scores_mean_i = np.mean(train_scores_i, axis=1)
            test_scores_mean_i = np.mean(test_scores_i, axis=1)

            color_test = colors[i % len(colors)]

            # Training score con linea tratteggiata
            plt.plot(train_sizes_i, train_scores_mean_i, line_styles[1], color=color_test,
                     label=f"Training score ({name})")

            # Cross-validation score con linea continua
            plt.plot(train_sizes_i, test_scores_mean_i, line_styles[0], color=color_test,
                     label=f"Cross-validation score ({name})")

        # Sposta la legenda fuori dal grafico
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.subplots_adjust(right=0.75)
        plt.show()

    def ensemble_on_top_features(self, ensemble_clf, top_features_idx, verbose=True):
        """
        Trains a VotingClassifier on the 10 most relevant features.

        Arguments:
        ensemble_clf: The VotingClassifier instance to be trained.
        top_features_idx: An array of indices for the top K features.
        verbose: Whether to display graphs and reports.
        """
        if verbose:
            print("".center(90, '-'))
            print(f"ENSEMBLE MODEL EVALUATION ON TOP {len(top_features_idx)} FEATURES".center(90, '-'))
            print("".center(90, '-'))

        # Clona l'istanza del classificatore per non modificare l'originale
        ensemble_clf_cloned = clone(ensemble_clf)

        X_train_selected = self.X_train[:, top_features_idx]
        X_test_selected = self.X_test[:, top_features_idx]
        X_selected = self.X[:, top_features_idx]

        ensemble_clf_cloned.fit(X_train_selected, self.y_train)

        if verbose:
            self.plot_learning_curve(ensemble_clf_cloned, X_selected)
            self.evaluate_metrics(ensemble_clf_cloned, X_test_selected)

        return {
            f"top_{len(top_features_idx)}_features": {
                'model': ensemble_clf_cloned,
                'X': X_selected,
                'y': self.y
            }
        }

    def ablation_analysis(self, model, feature_groups=None, verbose=True, plot=True,plot_columns=30):
        """
        Esegue ablation test completo: singolo e a gruppi di feature, con report e grafico.

        Args:
            model: modello sklearn-like non addestrato
            feature_groups: dict nome_gruppo -> lista_feature_da_rimuovere
            verbose: stampa risultati
            plot: se True genera bar chart comparativo

        Returns:
            results: dict con accuracy per ogni feature e gruppo
        """
        results = {}

        # --- 1. Accuracy con tutte le feature ---
        clf_all = clone(model)
        clf_all.fit(self.X_train, self.y_train)

        y_pred_all = clf_all.predict(self.X_test)
        acc_all = accuracy_score(self.y_test, y_pred_all)
        results["all_features"] = acc_all
        if verbose:
            print(f"All features accuracy: {acc_all:.4f}")

        # --- 2. Ablation singola ---
        single_results = {}
        for i, feature in enumerate(self.features):
            X_train_sub = np.delete(self.X_train, i, axis=1)
            X_test_sub = np.delete(self.X_test, i, axis=1)

            clf_sub = clone(model)
            clf_sub.fit(X_train_sub, self.y_train)

            y_pred_sub = clf_sub.predict(X_test_sub)
            acc_sub = accuracy_score(self.y_test, y_pred_sub)
            single_results[feature] = acc_sub

            if verbose:
                print(f"Removed feature '{feature}': accuracy = {acc_sub:.4f}")

        results["single_features"] = single_results

        # --- 3. Ablation per gruppi ---
        group_results = {}
        if feature_groups:
            for group_name, group_features in feature_groups.items():
                indices = [i for i, f in enumerate(self.features) if f in group_features]
                X_train_sub = np.delete(self.X_train, indices, axis=1)
                X_test_sub = np.delete(self.X_test, indices, axis=1)

                clf_sub = clone(model)

                clf_sub.fit(X_train_sub, self.y_train)
                y_pred_sub = clf_sub.predict(X_test_sub)
                acc_sub = accuracy_score(self.y_test, y_pred_sub)
                group_results[group_name] = acc_sub

                if verbose:
                    print(f"Removed group '{group_name}': accuracy = {acc_sub:.4f}")

        results["groups"] = group_results

        # --- 4. Grafico comparativo ---
        if verbose:
            if plot:
                # Combina singole feature e gruppi in un unico dizionario
                combined_results = {}
                for name, acc in single_results.items():
                    combined_results[f"S: {name}"] = acc  # S: per singola feature
                if group_results:
                    for name, acc in group_results.items():
                        combined_results[f"G: {name}"] = acc  # G: per gruppo

                # Ordina per accuracy crescente
                combined_sorted = sorted(combined_results.items(), key=lambda x: x[1])

                # Prendi solo gli ultimi 15 (migliori accuracy)
                combined_sorted = combined_sorted[-plot_columns:]

                labels, accs = zip(*combined_sorted)
                x_pos = np.arange(len(labels))

                plt.figure(figsize=(max(15, len(labels) * 0.5), 8))
                plt.bar(x_pos, accs, color=['skyblue' if l.startswith('S:') else 'salmon' for l in labels], alpha=0.7)

                # Linee di riferimento
                plt.axhline(acc_all, color='green', linestyle='--', label='All features')
                plt.axhline(0.900, color='orange', linestyle='--', label='Reference 0.9')

                plt.xticks(x_pos, labels, rotation=90)
                plt.ylabel("Accuracy after removal")
                plt.title(f"Top {plot_columns} Ablation Results (Highest Accuracy)")
                plt.legend()

                # Scala y ridotta
                plt.ylim(min(accs) - 0.005, max(accs) + 0.005)
                plt.tight_layout()
                plt.show()
        return results

    import random

    def run_random_ablation(self, model, embeddings, n_cycles=15,
                            n_very_big_groups=20, very_big_group_size=10,
                            n_big_groups=20, big_group_size=7,
                            n_medium_groups=20, medium_group_size=5,
                            n_small_groups=20, small_group_size=3,
                            n_very_small_groups=20, very_small_group_size=2,
                            verbose=True):
        """
        Esegue cicli di ablation test con gruppi predefiniti, random e piccoli gruppi.

        Args:
            clf: oggetto con metodo ablation_analysis
            model: modello sklearn-like non addestrato
            embeddings: DataFrame contenente le feature
            n_cycles: numero di cicli da eseguire
            n_random_groups: numero di gruppi random per ciclo
            random_group_size: dimensione dei gruppi random
            n_small_groups: numero di piccoli gruppi per ciclo
            small_group_size: dimensione dei piccoli gruppi
            verbose: se True stampa i progressi

        Returns:
            best_result: dict con il miglior gruppo/feature per ciascun ciclo
        """
        all_features = list(embeddings.columns)
        best_result = {}

        for k in range(n_cycles):
            # --- Definizione gruppi di base ---
            feature_groups = {}

            # Flags
            feature_groups["flags"] = [f for f in all_features if "flags" in f]

            # Coordinate raw
            feature_groups["positions_X"] = [f for f in all_features if
                                             "positions_" in f and "_X" in f and "normalized" not in f]
            feature_groups["positions_Y"] = [f for f in all_features if
                                             "positions_" in f and "_Y" in f and "normalized" not in f]

            # Coordinate normalizzate
            feature_groups["positions_norm_X"] = [f for f in all_features if "positions_normalized" in f and "_X" in f]
            feature_groups["positions_norm_Y"] = [f for f in all_features if "positions_normalized" in f and "_Y" in f]

            # Geometric info faccia / spalle / corpo
            feature_groups["geometric_face"] = [f for f in all_features if
                                                "geometric_info" in f and "face" in f and "k_" not in f]
            feature_groups["geometric_shoulders"] = [f for f in all_features if
                                                     "geometric_info" in f and "shoulder" in f and "k_" not in f]
            feature_groups["geometric_body"] = [f for f in all_features if
                                                "geometric_info" in f and "body" in f and "k_" not in f]

            # k-keypoints version
            feature_groups["k_positions_norm_X"] = [f for f in all_features if
                                                    "k_positions_normalized" in f and "_X" in f]
            feature_groups["k_positions_norm_Y"] = [f for f in all_features if
                                                    "k_positions_normalized" in f and "_Y" in f]
            feature_groups["k_geometric_face"] = [f for f in all_features if "k_geometric_info" in f and "face" in f]
            feature_groups["k_geometric_shoulders"] = [f for f in all_features if
                                                       "k_geometric_info" in f and "shoulder" in f]
            feature_groups["k_geometric_body"] = [f for f in all_features if "k_geometric_info" in f and "body" in f]

            # --- Gruppi casuali per ciclo ---
            for i in range(n_very_big_groups):
                group_name = f"random_group_{i + 1}"
                feature_groups[group_name] = random.sample(all_features, very_big_group_size)

            # --- Gruppi casuali per ciclo ---
            for i in range(n_big_groups):
                group_name = f"random_group_{i + 1}"
                feature_groups[group_name] = random.sample(all_features, big_group_size)

            # --- Gruppi medi per ciclo ---
            for i in range(n_medium_groups):
                group_name = f"medium_group_{i + 1}"
                feature_groups[group_name] = random.sample(all_features, medium_group_size)

            # --- Piccoli gruppi per ciclo ---
            for i in range(n_small_groups):
                group_name = f"small_group_{i + 1}"
                feature_groups[group_name] = random.sample(all_features, small_group_size)

            # --- Piccoli piccoli gruppi per ciclo ---
            for i in range(n_very_small_groups):
                group_name = f"small_group_{i + 1}"
                feature_groups[group_name] = random.sample(all_features, very_small_group_size)

            # --- Esegui ablation analysis ---
            results = self.ablation_analysis(model, feature_groups=feature_groups, verbose=verbose)

            # --- Trova il migliore tra singole e gruppi ---
            max_acc = 0
            best_feature_or_group = None
            best_features_list = None

            # Singole feature
            for feature, acc in results["single_features"].items():
                if acc > max_acc:
                    max_acc = acc
                    best_feature_or_group = feature
                    best_features_list = [feature]

            # Gruppi
            for group_name, acc in results["groups"].items():
                if acc > max_acc:
                    max_acc = acc
                    best_feature_or_group = group_name
                    best_features_list = feature_groups[group_name]

            # Salva il migliore di questo ciclo
            if best_feature_or_group is not None:
                best_result[f"cycle_{k + 1}_{best_feature_or_group}"] = [max_acc, best_features_list]

            if verbose:
                print(f"Cycle {k + 1} finished - Best: {best_feature_or_group} (acc={max_acc:.4f})")

        return best_result


    def iterative_ablation(self, model, embeddings, y, classes_bs,
                           max_cycles=50,
                           n_very_big_groups=20, very_big_group_size=10,
                           n_big_groups=20, big_group_size=7,
                           n_medium_groups=20, medium_group_size=5,
                           n_small_groups=20, small_group_size=3,
                           n_very_small_groups=20, very_small_group_size=2,
                           verbose=True):
        """
        Esegue ablation iterativa: rimuove progressivamente feature o gruppi
        se la loro rimozione mantiene o migliora l'accuracy.

        Args:
            clf: istanza di Classifier
            model: modello sklearn-like non addestrato
            embeddings: DataFrame con le feature correnti
            y: target
            classes_bs: dizionario label -> nome
            max_cycles: numero massimo di iterazioni
            n_random_groups, random_group_size, n_small_groups, small_group_size: parametri per run_random_ablation
            verbose: se True stampa log

        Returns:
            embeddings_final: embeddings ridotto con le feature rimanenti
            history: lista di tuple (ciclo, best_removed, accuracy)
        """
        embeddings_current = embeddings.copy()
        history = []

        # Calcola l'accuracy iniziale su tutte le feature
        clf_temp = Classifier(embeddings_current, y, classes_bs)
        results_all = clf_temp.ablation_analysis(model, feature_groups={}, verbose=verbose)
        base_acc = results_all["all_features"]

        if verbose:
            print(f"Starting accuracy with all features: {base_acc:.4f}")

        for cycle in range(1, max_cycles + 1):
            if verbose:
                print(f"\n=== Iterative ablation cycle {cycle} ===")

            # Esegui run_random_ablation sul set di feature corrente
            best_result = self.run_random_ablation(
                model=model,
                embeddings=embeddings_current,
                n_cycles=1,
                n_very_big_groups=n_very_big_groups,
                very_big_group_size=very_big_group_size,
                n_big_groups=n_big_groups,
                big_group_size=big_group_size,
                medium_group_size=medium_group_size,
                n_medium_groups=n_medium_groups,
                n_small_groups=n_small_groups,
                small_group_size=small_group_size,
                n_very_small_groups=n_very_small_groups,
                very_small_group_size=very_small_group_size,
                verbose=verbose
            )
            # Prendi il miglior elemento del ciclo
            best_key, (best_acc, best_features_list) = list(best_result.items())[0]

            if verbose:
                print(f"Best removal this cycle: {best_key} -> accuracy {best_acc:.4f}")



            # Se la rimozione migliora o mantiene l'accuracy
            if best_acc >= base_acc:
                embeddings_current = embeddings_current.drop(columns=best_features_list, errors='ignore')
                base_acc = best_acc
                history.append((cycle, best_key, best_acc))
                if verbose:
                    print(f"Removed features: {best_features_list}")
                    print(f"New accuracy: {base_acc:.4f}")
            else:
                if verbose:
                    print("No further improvement, stopping iterative ablation.")
                break

        return embeddings_current, history


    def plot_learning_curve(self, clf, X_selected=None, verbose=True):
        X = X_selected if X_selected is not None else self.X

        # ---- accuracy ----
        train_sizes, train_scores_acc, test_scores_acc = learning_curve(
            clf,
            X, self.y,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring="accuracy"
        )
        train_mean_acc = train_scores_acc.mean(axis=1)
        test_mean_acc = test_scores_acc.mean(axis=1)

        # ---- log loss ----
        train_sizes2, train_scores_loss, test_scores_loss = learning_curve(
            clf,
            X, self.y,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring="neg_log_loss"
        )
        # attenzione: sklearn restituisce "neg_log_loss", quindi invertiamo il segno
        train_mean_loss = -train_scores_loss.mean(axis=1)
        test_mean_loss = -test_scores_loss.mean(axis=1)

        if verbose:
            plt.figure(figsize=(12, 5))

            # Accuracy
            plt.subplot(1, 2, 1)
            plt.plot(train_sizes, train_mean_acc, 'o-', label="Train acc")
            plt.plot(train_sizes, test_mean_acc, 'o-', label="Val acc")
            plt.xlabel("Training set size")
            plt.ylabel("Accuracy")
            plt.title("Learning Curve - Accuracy")
            plt.grid(True)
            plt.legend()

            # Log loss
            plt.subplot(1, 2, 2)
            plt.plot(train_sizes2, train_mean_loss, 'o-', label="Train loss")
            plt.plot(train_sizes2, test_mean_loss, 'o-', label="Val loss")
            plt.xlabel("Training set size")
            plt.ylabel("Log Loss")
            plt.title("Learning Curve - Log Loss")
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.show()

    def evaluation_pipeline_save_misclassified(self,clf_untrained, verbose=True, is_ensemble=False,optimized:bool=False,n_top_features=[25, 10]):
        if verbose:
            print("".center(90, '-'))
            print("FIRST ANALYSIS".center(90, '-'))
            print("".center(90, '-'))

        clf_trained = clone(clf_untrained)


        clf_trained.fit(self.X_train, self.y_train)


        project_dir = f"{os.getcwd().split('SIDS_revelation_project')[0]}SIDS_revelation_project/"
        joblib.dump(clf_trained,f"{project_dir}classifiers/{get_model_name(clf_untrained)}_{self.len_features}_features{'_optimized' if optimized else ''}.pkl")

        importances, indices = None, None
        if not is_ensemble:
            importances, indices = self.plot_feature_importance(clf_trained, verbose=verbose)

        if verbose:
            self.plot_learning_curve(clf_trained)
            self.evaluate_metrics(clf_trained)

        results = {
            'all_features': {
                'model': clf_trained,
                'X': self.X,
                'y': self.y,
                'y_predicted': clf_trained.predict(self.X_test)
            }
        }

        for n_features in n_top_features:
            if not is_ensemble and importances is not None and len(self.features) > n_features:
                if verbose:
                    print("".center(90, '-'))
                    print(f"TOP {n_features} FEATURES ANALYSIS".center(90, '-'))
                    print("".center(90, '-'))
                    plt.figure(figsize=self.figsize)
                    plt.title(f"Top {n_features} feature importances")
                    plt.bar(range(n_features), importances[indices[:n_features]])
                    plt.xticks(range(n_features), [f"{self.features[i]}" for i in indices[:n_features]], rotation=90)
                    plt.tight_layout()
                    plt.show()

                top_k = n_features
                top_features_idx = indices[:top_k]
                X_train_selected = self.X_train[:, top_features_idx]
                X_test_selected = self.X_test[:, top_features_idx]
                X_selected = self.X[:, top_features_idx]
                clf_retrained = clone(clf_untrained)
                clf_retrained.fit(X_train_selected, self.y_train)


                if verbose:
                    self.plot_learning_curve(clf_retrained, X_selected)
                    self.evaluate_metrics(clf_retrained, X_test_selected)

                results[f"top_{n_features}_features"] = {
                    'model': clf_retrained,
                    'X': X_selected,
                    'y': self.y,
                    'top_features_idx': top_features_idx, # Added this to the dictionary,
                    'y_predicted': clf_retrained.predict(X_test_selected),
                }
                joblib.dump(clf_retrained,f"{project_dir}classifiers/{get_model_name(clf_untrained)}_{n_features}_features{'_optimized' if optimized else ''}.pkl")

        return results












