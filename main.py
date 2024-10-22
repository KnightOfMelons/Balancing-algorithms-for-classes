import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pacmap
import umap
import trimap
from sklearn.manifold import TSNE

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
# from sklearn.tree import plot_tree

from imblearn.over_sampling import SMOTE, BorderlineSMOTE


def load_data():
    # Вместо soybean используем horse-colic.data и horse-colic.test
    data = pd.read_csv('horse-colic.data', header=None, sep=r'\s+', na_values='?')
    test = pd.read_csv('horse-colic.test', header=None, sep=r'\s+', na_values='?')

    # Удаление строк с NaN в целевой переменной
    data = data.dropna(subset=[0])
    test = test.dropna(subset=[0])

    data_train = data.drop(columns=[0])
    X_train, y_train = data_train.fillna(data_train.mean()).values, data.iloc[:, 0]

    test_test = test.drop(columns=[0])
    X_test, y_test = test_test.fillna(test_test.mean()).values, test.iloc[:, 0]

    return X_train, y_train, X_test, y_test


def balance_data(X_test, y_test):
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_smote, y_smote = sm.fit_resample(X_test, y_test)

    bsm = BorderlineSMOTE(random_state=42, kind='borderline-1', k_neighbors=3)
    X_bsmote, y_bsmote = bsm.fit_resample(X_test, y_test)

    b2sm = BorderlineSMOTE(random_state=42, kind='borderline-2', k_neighbors=3)
    X_b2smote, y_b2smote = b2sm.fit_resample(X_test, y_test)

    return [X_test, X_smote, X_bsmote, X_b2smote], [y_test, y_smote, y_bsmote, y_b2smote]


def evaluate_model(model, X_methods, y_methods):
    for i, sample_name in enumerate(['Без балансировки', 'SMOTE', 'Borderline SMOTE', 'Borderline SMOTE 2']):
        evaluate_model_per_sample(model, sample_name, X_methods[i], y_methods[i])


def evaluate_model_per_sample(model, sample_name, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    print(
        f"[{sample_name}] Точность: {accuracy:.4f}, Precision: {precision:.4f},"
        f" Recall: {recall:.4f}, F1-мера: {f1:.4f}")


# Тут происходит визуализация если что
def visualize(X: np.ndarray, y: np.ndarray, method_name: str, support_vectors=None, ax=None):
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab20', edgecolor='k', label="Данные")
    ax.set_title(f"Визуализация методом {method_name}")

    # Добавляем опорные векторы (если есть) с другим стилем отображения
    if support_vectors is not None:
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='r',
                   label="Опорные векторы")

    ax.legend(loc='best')


def perform_visualization(X, y, model, support_vectors=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Сравнение методов уменьшения размерности", fontsize=16)

    # t-SNE
    tsne_reducer = TSNE(n_components=2, random_state=42)
    X_tsne = tsne_reducer.fit_transform(X)
    visualize(X_tsne, y, "t-SNE", support_vectors, ax=axes[0, 0])

    # UMAP
    umap_reducer = umap.UMAP(n_components=2)
    X_umap = umap_reducer.fit_transform(X)
    visualize(X_umap, y, "UMAP", support_vectors, ax=axes[0, 1])

    # TriMAP
    trimap_reducer = trimap.TRIMAP()
    X_trimap = trimap_reducer.fit_transform(X)
    visualize(X_trimap, y, "TriMAP", support_vectors, ax=axes[1, 0])

    # PacMAP
    pacmap_reducer = pacmap.PaCMAP(n_components=2)
    X_pacmap = pacmap_reducer.fit_transform(X)
    visualize(X_pacmap, y, "PacMAP", support_vectors, ax=axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Основная программа здесь
if __name__ == '__main__':

    X_train, y_train, X_test, y_test = load_data()
    X_methods, y_methods = balance_data(X_test, y_test)

    while True:
        choose = int(input("\n1 - SVM,\n2 - KNN,\n3 - Random forest\n0 - Выход\nВаш выбор: "))

        if choose == 1:
            print("Запуск SVM...")
            parameters = {
                'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'svc__C': [0.1, 1, 10, 100],
                'svc__gamma': ['scale', 'auto'],
                'svc__degree': [2, 3, 4]
            }
            clf = make_pipeline(MinMaxScaler(), SVC())
            grid = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
            grid.fit(X_test, y_test)
            best_model = grid.best_estimator_
            print(f"\nЛучшие параметры: {grid.best_params_}\n")
            evaluate_model(best_model, X_methods, y_methods)
            perform_visualization(X_test, y_test, best_model)

        elif choose == 2:
            print("Запуск KNN...")
            parameters = {
                'kneighborsclassifier__n_neighbors': [3, 5, 7, 10],
                'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                'kneighborsclassifier__weights': ['uniform', 'distance']
            }
            clf = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
            grid = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
            grid.fit(X_test, y_test)
            best_model = grid.best_estimator_
            print(f"\nЛучшие параметры: {grid.best_params_}\n")
            evaluate_model(best_model, X_methods, y_methods)
            perform_visualization(X_test, y_test, best_model)

        elif choose == 3:
            print("Запуск Random Forest...")
            parameters = {
                'randomforestclassifier__n_estimators': [50, 100, 200],
                'randomforestclassifier__max_depth': [None, 10, 20, 30],
                'randomforestclassifier__min_samples_split': [2, 5, 10],
                'randomforestclassifier__min_samples_leaf': [1, 2, 4],
                'randomforestclassifier__bootstrap': [True, False]
            }
            clf = make_pipeline(MinMaxScaler(), RandomForestClassifier())
            grid = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
            grid.fit(X_test, y_test)
            best_model = grid.best_estimator_
            print(f"\nЛучшие параметры: {grid.best_params_}\n")
            evaluate_model(best_model, X_methods, y_methods)
            perform_visualization(X_test, y_test, best_model)

        elif choose == 0:
            print("\nЗавершение программы")
            break
        else:
            continue
