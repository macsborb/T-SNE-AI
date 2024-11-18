# Importations nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import umap
#import umap.parametric_umap
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 

# Paramètres globaux pour les visualisations
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 100  # meilleure qualité d'image

# Chargement du dataset
covertype = fetch_ucirepo(id=31) 
X = covertype.data.features 
y = covertype.data.targets 

# Échantillonnage pour réduire la taille et la complexité
# (optionnel, mais utile pour accélérer T-SNE sur des jeux de données larges)
sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)

# Split du dataset en train et test
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# Fonction de visualisation avec T-SNE
def visualize_tsne(X, y, title):
    # Paramètres ajustés de T-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=100, max_iter=1500, learning_rate=600)
    X_tsne = tsne.fit_transform(X)
    
    # Création de la figure
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=np.ravel(y), palette="tab10", marker='o', s=30, alpha=0.7, edgecolor="k")
    
    # Personnalisation du graphique
    plt.title(f"T-SNE Visualization - {title}")
    plt.legend(title="Classes", loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.xlabel("T-SNE Dimension 1")
    plt.ylabel("T-SNE Dimension 2")
    plt.tight_layout()
    plt.show()

# 1. Visualisation des classes avec T-SNE
print("T-SNE visualization on original dataset")
visualize_tsne(X_train, y_train, "Original Dataset")

# 2. Visualisation avec UMAP
def visualize_umap(X, y, title, parametric=False):
    if parametric:
        umap_model = umap.parametric_umap.ParametricUMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.1)
    else:
        umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.5)
    X_umap = umap_model.fit_transform(X)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=np.ravel(y), palette="tab10", marker='o')
    plt.title(f"UMAP Visualization - {title}")
    plt.show()


print("UMAP visualization on original dataset")
visualize_umap(X_train, y_train, "Original Dataset", parametric=False)

# 3. Classification (RandomForest)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

