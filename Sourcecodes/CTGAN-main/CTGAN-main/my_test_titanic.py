from ctgan.synthesizers.ctgan import CTGAN
from ctgan import load_demo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

def calculate_fid(real_data, fake_data):

    real_data = real_data.values
    fake_data = fake_data.values
    
    mu_real, sigma_real = np.mean(real_data, axis=0), np.cov(real_data, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_data, axis=0), np.cov(fake_data, rowvar=False)
    
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    
    fid = np.sum(diff ** 2) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def plot_loss(loss_values, title="Training Loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values['Epoch'], loss_values['Generator Loss'], label="Generator Loss", color="blue")
    plt.plot(loss_values['Epoch'], loss_values['Discriminator Loss'], label="Discriminator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def calculate_diversity(fake_data):

    fake_data = fake_data.values
    dist = pairwise_distances(fake_data, metric='euclidean')
    diversity = np.mean(dist)  
    return diversity

def calculate_feature_correlation(fake_data):

    corr_matrix = fake_data.corr()
    return corr_matrix

def plot_correlation_matrix(corr_matrix, title="Feature Correlation"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()

def evaluate_model(real_data, fake_data):

    fid_score = calculate_fid(real_data, fake_data)
    print(f"FID Score: {fid_score}")

    diversity = calculate_diversity(fake_data)
    print(f"Data Diversity (Euclidean Distance Mean): {diversity}")

    corr_matrix = calculate_feature_correlation(fake_data)
    print("Feature Correlation Matrix:")
    plot_correlation_matrix(corr_matrix)



data = pd.read_csv('../CTGAN-main/CTGAN-main/examples/csv/train_clean.csv')
# Names of the columns that are discrete
discrete_columns = [
    'Survived',
    'Pclass',
    'Sex',
    'SibSp',
    'Parch',
    'Embarked'
]

ctgan = CTGAN(epochs=150)
ctgan.fit(data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
print(synthetic_data)
synthetic_data.to_csv('./fakeData_CTGAN-titanic1.csv', index=False)

evaluate_model(data, synthetic_data)

plot_loss(ctgan.loss_values)

sex_mapping = {0: 'male', 1: 'female'}  
survived_mapping = {0: 'died', 1: 'survived'}  

data['Sex'] = data['Sex'].map(sex_mapping)  
data['Survived'] = data['Survived'].map(survived_mapping)  
  
synthetic_data['Sex'] = synthetic_data['Sex'].map(sex_mapping)  
synthetic_data['Survived'] = synthetic_data['Survived'].map(survived_mapping)  

def plot_sex_survived_comparison(data, ax, title):  
    crosstab = pd.crosstab(data['Sex'], data['Survived'])  
    crosstab_long = crosstab.reset_index().melt(id_vars=['Sex'], var_name='Survived', value_name='count')  
      
    sns.barplot(x='Sex', y='count', hue='Survived', data=crosstab_long, ax=ax, palette='viridis')  
    ax.set_title(title)  
    ax.set_xlabel('Sex')  
    ax.set_ylabel('Passengers')  
    ax.legend(title='Survived or not?', bbox_to_anchor=(1.05, 1), loc='upper left') 
  

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  
  

plot_sex_survived_comparison(data, ax1, ' Real Data : Sex and Survived Distribution')  

plot_sex_survived_comparison(synthetic_data, ax2, ' Synthetic Data From Trans-CTGAN: Sex and Survived Distribution')  
  

plt.tight_layout(rect=[0, 0, 1, 0.96])   
plt.show()

# 0，0，0，0，0，1，0，0，0，0

# correlation_matrix_real = data.corr()
# sns.heatmap(correlation_matrix_real, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix Heatmap')
# plt.show()


# correlation_matrix_fake = synthetic_data.corr()
# sns.heatmap(correlation_matrix_fake, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix Heatmap')
# plt.show()