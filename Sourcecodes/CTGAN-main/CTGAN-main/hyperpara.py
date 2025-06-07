import itertools
import torch
from torch import optim

def hyperparameter_search(train_data, model_class, hyperparams, device='cuda'):
    best_loss = float('inf')
    best_params = None
    best_model = None

    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(*hyperparams.values()))
    
    for combination in param_combinations:
        params = dict(zip(hyperparams.keys(), combination))
        print(f"Training with params: {params}")
        
        # Initialize the model with the current hyperparameters
        model = model_class(
            embedding_dim=params['embedding_dim'],
            generator_dim=params['generator_dim'],
            data_dim=train_data.shape[1],  # Assuming train_data is a pandas dataframe or numpy array
            num_layers=params['num_layers'],
            num_heads=params['num_heads'],
            hidden_dim=params['hidden_dim']
        ).to(device)
        
        optimizerG = optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            betas=(0.5, 0.9),
            weight_decay=params['generator_decay']
        )

        # Train the model and calculate loss (placeholder function)
        loss = train_model(model, train_data, optimizerG, device)
        
        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_model = model
    
    print(f"Best Hyperparameters: {best_params}")
    return best_model, best_params

def train_model(model, train_data, optimizer, device):
    # Placeholder function for training loop, should be replaced with actual training logic
    # Return a dummy loss for the sake of the example
    return torch.rand(1).item()


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

train_data = np.random.rand(1000, 50) 
train_data_tensor = torch.tensor(train_data).float()

train_dataset = TensorDataset(train_data_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

hyperparams = {
    'embedding_dim': [128, 256], 
    'generator_dim': [(256, 256), (512, 512)], 
    'num_layers': [4, 6],  
    'num_heads': [4, 8], 
    'hidden_dim': [128, 256],  
    'learning_rate': [2e-4, 1e-4], 
    'generator_decay': [1e-6, 1e-5] 
}

best_model, best_params = hyperparameter_search(
    train_data=train_data,
    model_class=TransformerGenerator, 
    hyperparams=hyperparams,
    device='cuda'
)

print(f"Best Hyperparameters: {best_params}")
