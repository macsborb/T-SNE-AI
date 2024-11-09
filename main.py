# Importations nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
column_names = [f"feature_{i+1}" for i in range(54)] + ["target"]
data = pd.read_csv(url, header=None, names=column_names)

# Séparer les caractéristiques (X) et la cible (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split du dataset en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fonction de visualisation avec T-SNE
def visualize_tsne(X, y, title):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="tab10", marker='o')
    plt.title(f"T-SNE Visualization - {title}")
    plt.show()

# 1. Visualisation des classes avec T-SNE
print("T-SNE visualization on original dataset")
visualize_tsne(X_train, y_train, "Original Dataset")

# 2. Visualisation avec UMAP
def visualize_umap(X, y, title, parametric=False):
    if parametric:
        umap_model = umap.parametric_umap.ParametricUMAP(n_components=2, random_state=42)
    else:
        umap_model = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(X)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette="tab10", marker='o')
    plt.title(f"UMAP Visualization - {title}")
    plt.show()

print("UMAP visualization on original dataset")
visualize_umap(X_train, y_train, "Original Dataset", parametric=False)
