from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from xgboost import XGBClassifier

from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression


class Classifier:
    def __init__(self, embeddings, y, classes_bs, figsize=(5.6, 4.2)):
        self.X = embeddings.values
        self.features = embeddings.columns

        self.y = np.array(y)
        self.classes_bs = classes_bs

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

        self.figsize = figsize

    def logistic_regression(self):
        clf = LogisticRegression(
            penalty="l2",  # Ridge regularization
            C=0.1,  # smaller C = stronger regularization
            solver="liblinear"
        )
        self.evaluation_pipeline(clf)

    def random_forest(self):
        clf = RandomForestClassifier(n_estimators=300,
                                     max_depth=8,  # limit tree depth
                                     min_samples_split=10,  # require more samples to split
                                     min_samples_leaf=5,  # require more samples per leaf
                                     max_features="sqrt",  # random feature selection
                                     bootstrap=True,
                                     random_state=42)
        self.evaluation_pipeline(clf)

    def XGBC(self):
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1,
            reg_alpha=0.5,
            random_state=42

        )
        self.evaluation_pipeline(clf)

    def evaluation_pipeline(self, clf_untrained):
        print("".center(90, '-'))
        print("FIRST ANALYSIS".center(90, '-'))
        print("".center(90, '-'))

        clf_trained = clone(clf_untrained)

        clf_trained.fit(self.X_train, self.y_train)

        # Check feature importance
        importances, indices = self.plot_feature_importance(clf_trained)

        # learning curve
        self.plot_learning_curve(clf_untrained)

        # Evaluate metrics
        self.evaluate_metrics(clf_trained)

        # Re-train with top 10 features
        n_features = len(self.features)
        if n_features > 10 and importances is not None:
            print("".center(90, '-'))
            print("TOP 10 FEATURES ANALYSIS".center(90, '-'))
            print("".center(90, '-'))

            # Visualize top 10 feature importances
            plt.figure(figsize=self.figsize)
            plt.title("Top 10 feature importances")
            plt.bar(range(10), importances[indices[:10]])
            plt.xticks(range(10), [f"{self.features[i]}" for i in indices[:10]], rotation=90)
            plt.tight_layout()
            plt.show()

            # Select the 10 most important features
            top_k = 10
            top_features_idx = indices[:top_k]

            # Filter features
            X_train_selected = self.X_train[:, top_features_idx]
            X_test_selected = self.X_test[:, top_features_idx]
            X_selected = self.X[:, top_features_idx]

            # Train
            clf_retrained = clone(clf_untrained)
            clf_retrained.fit(X_train_selected, self.y_train)

            # learning curve
            self.plot_learning_curve(clf_untrained, X_selected)

            # Evaluate metrics
            self.evaluate_metrics(clf_retrained, X_test_selected)

    def evaluate_metrics(self, clf, X_test_selected=None):
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

    def plot_feature_importance(self, clf, top_k=None):
        if not hasattr(clf, "feature_importances_"):
            return None, None

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        n_features = top_k if top_k is not None else len(self.features)

        # Visualize all feature importances
        plt.figure(figsize=self.figsize)
        plt.title("Ordered feature importances")
        plt.bar(range(n_features), importances[indices[:n_features]])
        plt.xticks(range(n_features), [f"{self.features[i]}" for i in indices[:n_features]], rotation=90)
        plt.tight_layout()

        return importances, indices

    def plot_learning_curve(self, clf, X_selected=None, verbose=True):
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
            plt.xlabel("Training set size")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.title("Learning Curve")
            plt.show()

        return train_sizes, train_scores, test_scores, train_mean, test_mean
