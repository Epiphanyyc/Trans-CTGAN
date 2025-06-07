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
from scipy.stats import wasserstein_distance

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

    scaler = StandardScaler()

    real_data_scaled = scaler.fit_transform(real_data)
    fake_data_scaled = scaler.transform(fake_data)

    wasserstein_distances = []
    for i in range(real_data_scaled.shape[1]):
        dist = wasserstein_distance(real_data_scaled[:, i], fake_data_scaled[:, i])
        wasserstein_distances.append(dist)
    average_wasserstein_distance = sum(wasserstein_distances) / len(wasserstein_distances)
    print(f"Average Wasserstein Distance: {average_wasserstein_distance:.4f}")


    fid_score = calculate_fid(real_data, fake_data)
    print(f"FID Score: {fid_score}")
    
    diversity = calculate_diversity(fake_data)
    print(f"Data Diversity (Euclidean Distance Mean): {diversity}")

    corr_matrix = calculate_feature_correlation(fake_data)
    print("Feature Correlation Matrix:")
    plot_correlation_matrix(corr_matrix)

data = pd.read_csv('../CTGAN-main/CTGAN-main/examples/csv/train_clean.csv')
synthetic_data = pd.read_csv("../fakeData_trans-CTGAN--titanic.csv")

evaluate_model(data, synthetic_data)

#Wasserstein Distance
#TransCTGAN:   0.22
#CTGAN:  0.1760
#TransCTGAN:   FID : 103       Data Diversity: 41.4
#CTGAN:      349      37


