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
  
# fetch dataset 
covertype = fetch_ucirepo(id=31) 
  
# data (as pandas dataframes) 
X = covertype.data.features 
y = covertype.data.targets 

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

# 3. Classification (RandomForest)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# 4. Under-sampling avec Imbalanced-Learn et répétition des étapes 1, 2, 3
under_sampler = RandomUnderSampler(random_state=42)
X_res, y_res = under_sampler.fit_resample(X_train, y_train)
print("T-SNE visualization after under-sampling")
visualize_tsne(X_res, y_res, "Under-sampled Dataset")

print("UMAP visualization after under-sampling")
visualize_umap(X_res, y_res, "Under-sampled Dataset", parametric=False)

clf.fit(X_res, y_res)
y_pred_res = clf.predict(X_test)
print("Under-sampling Accuracy:", accuracy_score(y_test, y_pred_res))
print("Under-sampling F1 Score:", f1_score(y_test, y_pred_res, average='weighted'))

# 5. Over-sampling avec SMOTE et répétition des étapes 1, 2, 3
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print("T-SNE visualization after SMOTE")
visualize_tsne(X_res, y_res, "SMOTE Oversampled Dataset")

print("UMAP visualization after SMOTE")
visualize_umap(X_res, y_res, "SMOTE Oversampled Dataset", parametric=False)

clf.fit(X_res, y_res)
y_pred_res = clf.predict(X_test)
print("SMOTE Oversampling Accuracy:", accuracy_score(y_test, y_pred_res))
print("SMOTE Oversampling F1 Score:", f1_score(y_test, y_pred_res, average='weighted'))
