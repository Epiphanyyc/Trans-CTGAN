from scipy.linalg import sqrtm
import numpy as np
import torch
from sklearn.metrics import pairwise_distances

# Function to calculate FID score
def calculate_fid(real_data, generated_data, model, device='cuda'):
    # Assuming real_data and generated_data are numpy arrays
    # Use model to get features (e.g., from a pre-trained InceptionV3 model)
    
    # Convert data to torch tensors
    real_data = torch.tensor(real_data).float().to(device)
    generated_data = torch.tensor(generated_data).float().to(device)
    
    # Extract features using a pretrained model (you can use any feature extractor)
    real_features = model(real_data)  # Placeholder for actual feature extraction
    gen_features = model(generated_data)  # Placeholder for actual feature extraction
    
    # Calculate mean and covariance of real and generated features
    real_mean = torch.mean(real_features, dim=0)
    gen_mean = torch.mean(gen_features, dim=0)
    real_cov = torch.cov(real_features.T)
    gen_cov = torch.cov(gen_features.T)
    
    # Calculate FID score
    mean_diff = real_mean - gen_mean
    cov_sqrt = sqrtm(real_cov @ gen_cov)
    
    fid = np.linalg.norm(mean_diff) + np.trace(real_cov + gen_cov - 2 * cov_sqrt)
    return fid

# Function to calculate Inception Score (IS)
def calculate_inception_score(generated_data, model, device='cuda'):
    # This function assumes `generated_data` is a batch of images or tabular data
    generated_data = torch.tensor(generated_data).float().to(device)
    
    # Get predictions from the model
    predictions = model(generated_data)  # Placeholder for actual prediction
    
    # Compute the inception score (Placeholder)
    p_y = torch.mean(predictions, dim=0)
    kl_div = predictions * (torch.log(predictions) - torch.log(p_y))
    is_score = torch.mean(torch.sum(kl_div, dim=1))  # Inception Score calculation
    return is_score

# Function to calculate training loss (for stability)
def calculate_loss(generator, discriminator, data, device='cuda'):
    real_data = torch.tensor(data).float().to(device)
    fake_data = generator(torch.randn(real_data.size(0), generator.embedding_dim).to(device))
    
    # Calculate discriminator's real and fake losses
    real_validity = discriminator(real_data)
    fake_validity = discriminator(fake_data)
    
    # Placeholder loss calculations
    real_loss = torch.mean((real_validity - 1) ** 2)
    fake_loss = torch.mean(fake_validity ** 2)
    
    total_loss = (real_loss + fake_loss) / 2
    return total_loss


generated_data = best_model(torch.randn(100, best_params['embedding_dim']).to('cuda')).cpu().detach().numpy()  

fid_score = calculate_fid(real_data=train_data, generated_data=generated_data, model=best_model, device='cuda')
print(f"FID Score: {fid_score}")

is_score = calculate_inception_score(generated_data=generated_data, model=best_model, device='cuda')
print(f"Inception Score: {is_score}")

train_loss = calculate_loss(generator=best_model, discriminator=best_model, data=train_data, device='cuda')
print(f"Training Loss: {train_loss}")
