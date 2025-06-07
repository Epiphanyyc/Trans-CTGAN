
from sklearn.model_selection import ParameterGrid
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load reference data
from ctgan.synthesizers.ctgan import CTGAN
from ctgan import load_demo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# third party
from sklearn.datasets import load_breast_cancer,load_diabetes


def evaluate_model(model):
    generated_data = model.sample(1000) 

    combined_data = np.vstack([real_data, generated_data])
    labels = np.hstack([np.ones(len(real_data)), np.zeros(len(generated_data))])

    X_train, X_test, y_train, y_test = train_test_split(combined_data, labels, test_size=0.2, random_state=42)
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    return 1 - score

real_data = pd.read_csv('../CTGAN-main/CTGAN-main/Adult_datasets.csv')
discrete_columns = ['education-num','sex','salary']
# 'num_layers': [2, 4, 6],
# 'num_heads': [4, 8, 16],
# epochs 250
param_grid = {
    'num_layers': [4],
    'num_heads': [8],
    'batch_size': [500],
    'epochs' : [100],
    'generator_lr': [2e-4],
    'discriminator_lr' :[2e-4]
}

best_params = None
best_score = -np.inf 

for params in ParameterGrid(param_grid):

    ctgan = CTGAN(**params)
    ctgan.fit(real_data, discrete_columns)

    score = evaluate_model(ctgan)

    print(f"Current Parameters: {params}, Score: {score}")

    if score > best_score:
        best_score = score
        best_params = params

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")