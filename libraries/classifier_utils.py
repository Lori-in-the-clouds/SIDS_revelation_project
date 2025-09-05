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
import pickle

def get_model_name(clf_untrained):
    # Ottieni il nome della classe
    class_name = clf_untrained.__class__.__name__
    return class_name

class Classifier:
    def __init__(self, embeddings, y, classes_bs, figsize = (5.6, 4.2)):
        self.X = embeddings.values
        self.features = embeddings.columns
        self.len_features = len(embeddings.columns)

        self.y = np.array(y)
        self.classes_bs = classes_bs

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

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

    def XGBC(self,verbose=True):
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
        results = self.evaluation_pipeline(clf,verbose = verbose)
        return clf,results

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
            self.plot_learning_curve(clf_untrained)
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
            plt.show()

        return train_sizes,train_scores,test_scores,train_mean,test_mean

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












