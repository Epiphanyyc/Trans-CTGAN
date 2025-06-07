class TransformerGenerator(Module):
    """Transformer-based Generator for CTGAN."""
    
    def __init__(self, embedding_dim, generator_dim, data_dim, num_layers=4, num_heads=8, hidden_dim=256):
        super(TransformerGenerator, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.data_dim = data_dim
        
        # Embedding input latent space to the feature space
        self.embedding = Linear(embedding_dim, hidden_dim)
        
        # Transformer Encoder layers
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        
        # Final output layer to map the transformer output to data space
        self.fc_out = Linear(hidden_dim, data_dim)

    def forward(self, z):
        """Forward pass through the generator."""
        x = self.embedding(z)  # Input embedding
        x = x.unsqueeze(0)  # Add sequence dimension for Transformer: [1, batch_size, hidden_dim]
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Remove the sequence dimension
        
        # Output layer to generate data
        generated_data = self.fc_out(x)  # [batch_size, data_dim]
        return generated_data


class TransformerDiscriminator(Module):
    """Transformer-based Discriminator for CTGAN."""
    
    def __init__(self, input_dim, discriminator_dim, num_layers=4, num_heads=8, hidden_dim=256):
        super(TransformerDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        
        # Embedding input to feature space
        self.embedding = Linear(input_dim, hidden_dim)
        
        # Transformer Encoder layers
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim),
            num_layers=num_layers
        )
        
        # Final output layer to classify as real or fake
        self.fc_out = Linear(hidden_dim, 1)

    def forward(self, x):
        """Forward pass through the discriminator."""
        x = self.embedding(x)  # [batch_size, input_dim]
        x = x.unsqueeze(0)  # Add sequence dimension: [1, batch_size, hidden_dim]
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Remove sequence dimension
        
        # Output layer for binary classification
        validity = torch.sigmoid(self.fc_out(x))  # [batch_size, 1]
        return validity


class CTGAN(BaseSynthesizer):
    """Modified CTGAN with Transformer-based Generator and Discriminator."""

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256), 
                 generator_lr=2e-4, discriminator_lr=2e-4, batch_size=500, epochs=300, pac=10, cuda=True):
        # ... (initialize other parameters)
        
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self._batch_size = batch_size
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            self._device = torch.device('cpu')
        else:
            self._device = torch.device('cuda')

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None

    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the modified CTGAN with Transformer-based models."""
        # Prepare the data transformer and sampler
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        self._data_sampler = DataSampler(train_data, self._transformer.output_info_list)
        
        data_dim = self._transformer.output_dimensions

        # Initialize the modified generator and discriminator
        self._generator = TransformerGenerator(
            embedding_dim=self._embedding_dim + self._data_sampler.dim_cond_vec(),
            generator_dim=self._generator_dim,
            data_dim=data_dim
        ).to(self._device)
        
        discriminator = TransformerDiscriminator(
            input_dim=data_dim + self._data_sampler.dim_cond_vec(),
            discriminator_dim=self._discriminator_dim
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )
        optimizerD = optim.Adam(
            discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6
        )
        
        # Training loop...
