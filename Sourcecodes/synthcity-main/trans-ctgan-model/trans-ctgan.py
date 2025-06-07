"""CTGAN module."""

import warnings
import numpy as np
import pandas as pd
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from torch.nn.utils import spectral_norm
from tqdm import tqdm
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from torch import Tensor
from typing import Optional, Any
from typing import List
#########################################
class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))

class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)

class Generator(Module):
    """Generator for the CTGAN."""
    #256--linear256 bn256 relu cat[512]    512 linear256 bn256 relu cat 512  512linear data_dim
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data
#######################################################
# class ResidualBlock(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.linear = nn.Linear(in_dim, out_dim)
#         self.activation = nn.ReLU()
        
#     def forward(self, x):
#         residual = x if x.shape[-1] == self.linear.out_features else \
#                    nn.Linear(x.shape[-1], self.linear.out_features, device=x.device)(x)
#         return self.activation(self.linear(x)) + residual
    
import torch
import torch.nn as nn
from torch.nn import Linear, MultiheadAttention, TransformerEncoder
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            #nn.BatchNorm1d(dim),
            #nn.ReLU(),
            nn.LeakyReLU()
            #nn.Linear(dim, dim) 
        )
    def forward(self, x):
        return x + self.layers(x)
######################################################
# 3layer   linear 256 leaky relu (Res)
class ETransformerGenerator(Module):
    """Transformer-based Generator for CTGAN."""
    #4
    def __init__(self,noise_dim,cond_dim,embedding_dim, generator_dim,data_dim,dropout=0.1,num_first=False,batch_first=False, num_layers=1, num_heads=2, hidden_dim=256, num_fc_layers=2, fc_dim=256):
        super(ETransformerGenerator, self).__init__()
        self.embedding_dim = embedding_dim
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        # Embedding input latent space to the feature space
        self.embedding = Linear(noise_dim+cond_dim, hidden_dim)
        
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=batch_first,norm_first=num_first,dropout=dropout),
            num_layers=num_layers
        )
        #self.fc_out = Linear(hidden_dim, data_dim)
        #nn.BatchNorm1d(hidden_dim),
        #nn.LeakyReLU(0.2),
        self.blocks = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            #nn.ReLU(),
            nn.LeakyReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, data_dim)
        )

        # # Additional fully connected layers
        # self.fc_layers = Sequential()
        # prev_dim = hidden_dim
        # for i in range(num_fc_layers):
            
        #     #self.fc_layers.add_module(f'res_{i}', ResidualBlock(prev_dim, fc_dim))

        #     self.fc_layers.add_module(f'fc_{i}', Linear(prev_dim, fc_dim))
        #     #self.fc_layers.add_module(f'relu_{i}', nn.LeakyReLU(0.2))
        #     self.fc_layers.add_module(f'relu_{i}', ReLU())
        #     #self.fc_layers.add_module(f'bn_{i}', nn.BatchNorm1d(fc_dim))
        #     #self.fc_layers.add_module(f'dropout_{i}', nn.Dropout(0.0))

        #     prev_dim = fc_dim

        # self.fc_out = Linear(fc_dim, data_dim)

    def forward(self, z):
        """Forward pass through the generator."""
        x = self.embedding(z)  # Input embedding
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        #x = self.fc_layers(x)
        generated_data = self.blocks(x)
        # Output layer to generate data
        #generated_data = self.fc_out(x)  # [batch_size, data_dim]
        return generated_data


class ETransformerDiscriminator(Module):
    """Transformer-based Discriminator for CTGAN."""
    #5
    def __init__(self, input_dim, discriminator_dim,dropout=0.1,num_first=False,batch_first=False,num_layers=1, num_heads=2, hidden_dim=256, pac=10, num_fc_layers=2, fc_dim=256):
        super(ETransformerDiscriminator, self).__init__()
        from torch.nn import TransformerEncoder
        # self.pac = pac
        # input_dim = input_dim * self.pac
        # self.pacdim = input_dim
        # Embedding input to feature space
        self.embedding = Linear(input_dim, hidden_dim)
        
        #Transformer Encoder layers with batch_first=True
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=batch_first,norm_first=num_first,dropout=dropout),
            num_layers=num_layers
        )
        self.blocks = nn.Sequential(
            #nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        # #self.fc_out = Linear(hidden_dim, 1)
        # # Additional fully connected layers
        # self.fc_layers = Sequential()
        # prev_dim = hidden_dim
        # for i in range(num_fc_layers):

        #     # self.fc_layers.add_module(f'res_{i}', ResidualBlock(prev_dim, fc_dim))
            
        #     self.fc_layers.add_module(f'fc_{i}', Linear(prev_dim, fc_dim))
        #     #relf.fc_layers.add_module(f'relu_{i}', nn.LeakyReLU(0.2))
        #     self.fc_layers.add_module(f'relu_{i}', ReLU())
        #     #self.fc_layers.add_module(f'bn_{i}', nn.BatchNorm1d(fc_dim))
        #     #self.fc_layers.add_module(f'dropout_{i}', nn.Dropout(0.1))

        #     prev_dim = fc_dim

        # # Final output layer to classify as real or fake
        # self.fc_out = Linear(fc_dim, 1)
    # def calc_gradient_penalty(self, real_data, fake_data, device='cpu', lambda_=10):
    #     """Compute the gradient penalty for the WGAN discriminator."""
        
    #     alpha = torch.rand(real_data.size(0) // self.pac, 1, 1, device=device)
    #     alpha = alpha.repeat(1, self.pac, real_data.size(1))
    #     alpha = alpha.view(-1, real_data.size(1))

    #     interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    #     disc_interpolates = self(interpolates)

    #     gradients = torch.autograd.grad(
    #         outputs=disc_interpolates,
    #         inputs=interpolates,
    #         grad_outputs=torch.ones(disc_interpolates.size(), device=device),
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True,
    #     )[0]

    #     gradients_view = gradients.view(-1, self.pac * real_data.size(1)).norm(2, dim=1) - 1
    #     gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

    #     return gradient_penalty
    def calc_gradient_penalty_pac(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty
    
    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', lambda_=10):
        """Compute the gradient penalty for the WGAN discriminator."""
        
        # Generate random alpha values for interpolation
        alpha = torch.rand(real_data.size(0), 1, device=device)
        alpha = alpha.view(-1, 1)

        # Interpolate between real and fake data
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        # Compute discriminator output for interpolates
        disc_interpolates = self(interpolates)

        # Compute gradients of discriminator output w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute gradient penalty
        gradients_view = gradients.view(-1, real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, x):
        """Forward pass through the discriminator."""
        # x = x.view(-1, self.pacdim) 
        x = self.embedding(x)  # [batch_size, input_dim]
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        #x = self.fc_layers(x)
        validity = torch.sigmoid(self.blocks(x))  # [batch_size, 1]
        # Output layer for binary classification
        #validity = torch.sigmoid(self.fc_out(x))  # [batch_size, 1]
        return validity

#######################################################
class TransformerGenerator(Module):
    """Transformer-based Generator for CTGAN."""
    #4
    def __init__(self,noise_dim,cond_dim,embedding_dim, generator_dim,data_dim,dropout=0.1,num_first=True,batch_first=True, num_layers=1, num_heads=4, hidden_dim=256, num_fc_layers=2, fc_dim=256):
        super(TransformerGenerator, self).__init__()
        self.embedding_dim = embedding_dim
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        # Embedding input latent space to the feature space
        self.embedding = Linear(noise_dim+cond_dim, hidden_dim)
        
        # Transformer Encoder layers with batch_first=True     (dropout=Dropout)(norm_first=True)
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=batch_first,norm_first=num_first,dropout=dropout),
            num_layers=num_layers
        )
        #self.fc_out = Linear(hidden_dim, data_dim)

        # self.blocks = nn.Sequential(
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2),
        #     ResidualBlock(256),
        #     ResidualBlock(256),
        #     nn.Linear(256, data_dim)
        # )

        # Additional fully connected layers
        self.fc_layers = Sequential()
        prev_dim = hidden_dim
        for i in range(num_fc_layers):
            
            #self.fc_layers.add_module(f'res_{i}', ResidualBlock(prev_dim, fc_dim))

            self.fc_layers.add_module(f'fc_{i}', Linear(prev_dim, fc_dim))
            #self.fc_layers.add_module(f'relu_{i}', nn.LeakyReLU(0.2))
            self.fc_layers.add_module(f'relu_{i}', ReLU())
            #self.fc_layers.add_module(f'bn_{i}', nn.BatchNorm1d(fc_dim))
            #self.fc_layers.add_module(f'dropout_{i}', nn.Dropout(0.0))

            prev_dim = fc_dim

        self.fc_out = Linear(fc_dim, data_dim)

    def forward(self, z):
        """Forward pass through the generator."""
        x = self.embedding(z)  # Input embedding
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        x = self.fc_layers(x)
        #generated_data = self.blocks(x)
        # Output layer to generate data
        generated_data = self.fc_out(x)  # [batch_size, data_dim]
        return generated_data


class TransformerDiscriminator(Module):
    """Transformer-based Discriminator for CTGAN."""
    #5
    def __init__(self, input_dim, discriminator_dim,dropout=0.1,num_first=True,batch_first=True,num_layers=1, num_heads=4, hidden_dim=256, pac=10, num_fc_layers=2, fc_dim=256):
        super(TransformerDiscriminator, self).__init__()
        from torch.nn import TransformerEncoder
        self.input_dim = input_dim
        self.pac = pac
        
        # Embedding input to feature space
        self.embedding = Linear(input_dim, hidden_dim)
        
        #Transformer Encoder layers with batch_first=True
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=batch_first,norm_first=num_first,dropout=dropout),
            num_layers=num_layers
        )
        # self.blocks = nn.Sequential(
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(0.2),
        #     ResidualBlock(256),
        #     ResidualBlock(256),
        #     nn.Linear(256, 1)
        # )

        #self.fc_out = Linear(hidden_dim, 1)
        # Additional fully connected layers
        self.fc_layers = Sequential()
        prev_dim = hidden_dim
        for i in range(num_fc_layers):

            # self.fc_layers.add_module(f'res_{i}', ResidualBlock(prev_dim, fc_dim))
            
            self.fc_layers.add_module(f'fc_{i}', Linear(prev_dim, fc_dim))
            #relf.fc_layers.add_module(f'relu_{i}', nn.LeakyReLU(0.2))
            self.fc_layers.add_module(f'relu_{i}', ReLU())
            #self.fc_layers.add_module(f'bn_{i}', nn.BatchNorm1d(fc_dim))
            #self.fc_layers.add_module(f'dropout_{i}', nn.Dropout(0.1))

            prev_dim = fc_dim

        # # Final output layer to classify as real or fake
        self.fc_out = Linear(fc_dim, 1)

    def forward(self, x):
        """Forward pass through the discriminator."""
        x = self.embedding(x)  # [batch_size, input_dim]
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        x = self.fc_layers(x)
        #validity = torch.sigmoid(self.blocks(x))  # [batch_size, 1]
        # Output layer for binary classification
        validity = torch.sigmoid(self.fc_out(x))  # [batch_size, 1]
        return validity

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', lambda_=10):
        """Compute the gradient penalty for the WGAN discriminator."""
        
        alpha = torch.rand(real_data.size(0) // self.pac, 1, 1, device=device)
        alpha = alpha.repeat(1, self.pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, self.pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.
    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
        #1
        dropout = 0.3,
        num_first = False,
        batch_first = False,
        num_layers = 1,  #1 2 4 6 8
        num_heads = 4,   #2 4 8
        hidden_dim = 256
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        #2
        self.dropout = dropout
        self.num_first = num_first
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self.cond_dim = None
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.
        
        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)
        from torch.nn import TransformerEncoder
        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead'
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions
        self.cond_dim = self._data_sampler.dim_cond_vec() 
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)
        #3
        # self._generator = TransformerGenerator(
        #     noise_dim = self._embedding_dim,
        #     cond_dim = self.cond_dim,
        #     embedding_dim=self._embedding_dim + self._data_sampler.dim_cond_vec(),
        #     generator_dim=self._generator_dim,
        #     data_dim=data_dim,
        #     dropout=self.dropout,
        #     num_first=self.num_first,
        #     batch_first=self.batch_first,
        #     num_heads=self.num_heads,
        #     num_layers=self.num_layers,
        #     hidden_dim = self.hidden_dim
        # ).to(self._device)
        
        # discriminator = TransformerDiscriminator(
        #     input_dim=data_dim + self._data_sampler.dim_cond_vec(),
        #     discriminator_dim=self._discriminator_dim,
        #     dropout=self.dropout,
        #     num_first=self.num_first,
        #     batch_first=self.batch_first,
        #     num_heads=self.num_heads,
        #     num_layers=self.num_layers,
        #     hidden_dim = self.hidden_dim
        # ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))
        print(epoch_iterator)
        print(f"Hype-parameter:[dropout:{self.dropout},num_first:{self.num_first},batch_first:{self.batch_first},num_heads:{self.num_heads},num_layers:{self.num_layers}]")
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            print(f"epoch {i}")
            print("newtype2")
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)


class EnhancedCTGAN(CTGAN):
    def __init__(self,cuda=True,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cond_dim = None  #to save the cond
        self.has_cond = 1
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
            print("cpu")
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'
        self._device = torch.device(device)
    # pearson
    def _custom_corr_loss(self, real, fake, has_cond):
        if has_cond:
            data_real = real[:, :-self.cond_dim]
            data_fake = fake[:, :-self.cond_dim]
        else:
            data_real = real
            data_fake = fake
    
        def _corr_matrix(x):
            x_centered = x - x.mean(dim=0)
            cov = x_centered.T @ x_centered / (x.size(0) - 1)
            std = torch.sqrt(torch.diag(cov)).clamp_min(1e-6)
            return cov / (std[:, None] * std[None, :] + 1e-6)
    
        real_corr = _corr_matrix(data_real)
        fake_corr = _corr_matrix(data_fake)
        corr_loss = torch.norm(real_corr - fake_corr, p='fro')  # Frobenius
        return corr_loss
    #mixed
    def _custom_mix_corr_loss(self, real, fake):
        # Split data and condition (if needed)
        data_real = real[:, :-self.cond_dim]
        data_fake = fake[:, :-self.cond_dim]
        
        # ----------------------------
        # 1. Pearson Correlation Loss (Existing)
        # ----------------------------
        def _pearson_corr(x):
            x_centered = x - x.mean(dim=0)
            cov = x_centered.T @ x_centered / (x.size(0) - 1)
            std = torch.sqrt(torch.diag(cov)).clamp_min(1e-6)
            return cov / (std[:, None] * std[None, :])

        pearson_real = _pearson_corr(data_real)
        pearson_fake = _pearson_corr(data_fake)
        pearson_loss = torch.norm(pearson_real - pearson_fake, p='fro')  # Frobenius norm

        # ----------------------------
        # 2. Distance Correlation Loss (Fixed)
        # ----------------------------
        def _distance_corr(x, y):
            # Ensure input tensors are 2D [batch_size, 1]
            x = x.unsqueeze(-1) if x.dim() == 1 else x  # Add feature dimension
            y = y.unsqueeze(-1) if y.dim() == 1 else y
            
            # Compute pairwise distance matrices
            a = torch.cdist(x, x, p=2)          # Shape: [batch_size, batch_size]
            b = torch.cdist(y, y, p=2)
            
            # Double centering
            n = x.size(0)
            a_row = a.mean(dim=1, keepdim=True)
            a_col = a.mean(dim=0, keepdim=True)
            a_mean = a.mean()
            a_centered = a - a_row - a_col + a_mean
            
            b_row = b.mean(dim=1, keepdim=True)
            b_col = b.mean(dim=0, keepdim=True)
            b_mean = b.mean()
            b_centered = b - b_row - b_col + b_mean
            
            # Compute distance covariance and variances
            dcov_xy = (a_centered * b_centered).sum() / (n**2)
            dcov_xx = (a_centered * a_centered).sum() / (n**2)
            dcov_yy = (b_centered * b_centered).sum() / (n**2)
            
            # Avoid division by zero
            dcor = dcov_xy / (torch.sqrt(dcov_xx * dcov_yy) + 1e-6)
            return dcor

        # Compute pairwise dCor for all features
        dcor_loss = 0.0
        num_features = data_real.size(1)
        for i in range(num_features):
            for j in range(i+1, num_features):
                # Extract features with explicit 2D shape [batch_size, 1]
                x_real = data_real[:, i].unsqueeze(-1)  # Shape: [batch_size, 1]
                y_real = data_real[:, j].unsqueeze(-1)
                x_fake = data_fake[:, i].unsqueeze(-1)
                y_fake = data_fake[:, j].unsqueeze(-1)
                
                real_dcor = _distance_corr(x_real, y_real)
                fake_dcor = _distance_corr(x_fake, y_fake)
                dcor_loss += (real_dcor - fake_dcor)**2
        
        # ----------------------------
        # 3. Combined Loss
        # ----------------------------
        total_loss = pearson_loss + torch.sqrt(dcor_loss)  # Weighted combination
        return total_loss
    
    def _dcor_corr_loss(self, real, fake, has_cond):
        
        if has_cond:
            data_real = real[:, :-self.cond_dim]
            data_fake = fake[:, :-self.cond_dim]
        else:
            data_real = real
            data_fake = fake
        def compute_dcor_matrix(data):

            n, d = data.shape
            

            x = data.unsqueeze(1)  # (n, 1, d)
            a = torch.abs(x - x.permute(1, 0, 2))  # (n, n, d)
            
            a_row_mean = a.mean(dim=1, keepdim=True)  # (n, 1, d)
            a_col_mean = a.mean(dim=0, keepdim=True)  # (1, n, d)
            a_grand_mean = a.mean(dim=(0, 1), keepdim=True)  # (1, 1, d)
            
            a_centered = a - a_row_mean - a_col_mean + a_grand_mean
            
            a_centered = a_centered.permute(2, 0, 1)  # (d, n, n)
            
            d_cov = torch.einsum('ikl,jkl->ij', a_centered, a_centered) / (n ** 2)

            d_var = (a_centered ** 2).mean(dim=(1, 2))  # (d,)
            
            d_var_sqrt = torch.sqrt(d_var)
            d_cor = d_cov / (d_var_sqrt[:, None] * d_var_sqrt[None, :] + 1e-9)
            
            return d_cor
        
        real_dcor = compute_dcor_matrix(data_real)
        fake_dcor = compute_dcor_matrix(data_fake)

        corr_loss = torch.norm(real_dcor - fake_dcor, p='fro')
        
        return corr_loss

    def dcor_new_loss(self, real, fake, has_cond):

        def compute_dcor_matrix(data):
            n, d = data.shape
            chunk_size = 512
            dcor = torch.zeros(d, d, device=data.device)
            
            for i in range(0, d, chunk_size):
                i_end = min(i + chunk_size, d)
                for j in range(i, d, chunk_size):
                    j_end = min(j + chunk_size, d)
                    
                    x_block = data[:, i:i_end].unsqueeze(1)  # (n, 1, chunk_i)
                    y_block = data[:, j:j_end].unsqueeze(0)  # (1, n, chunk_j)
                    
                    a = torch.sqrt(
                        torch.sum((x_block - y_block)**2, 
                        dim=-1, 
                        keepdim=True
                    ))
                    
                    a_row_mean = a.mean(dim=1, keepdim=True)  # (n, 1, 1)
                    a_col_mean = a.mean(dim=0, keepdim=True)  # (1, n, 1)
                    a_grand_mean = a.mean(dim=(0,1), keepdim=True)  # (1, 1, 1)
                    a_centered = a - a_row_mean - a_col_mean + a_grand_mean
                    
                    cov_block = torch.einsum('ijk,ijl->kl', a_centered, a_centered) / (n**2)

                    var_i = (a_centered**2).mean(dim=(0,1))  # (1,)
                    var_j = (a_centered**2).mean(dim=(0,1))  # (1,)

                    dcor[i:i_end, j:j_end] = cov_block / (torch.sqrt(var_i * var_j) + 1e-9)
                    if i != j: 
                        dcor[j:j_end, i:i_end] = dcor[i:i_end, j:j_end].T
            return dcor
        if has_cond:
            data_real = real[:, :-self.cond_dim]
            data_fake = fake[:, :-self.cond_dim]
        else:
            data_real = real
            data_fake = fake

        real_dcor = compute_dcor_matrix(data_real)
        fake_dcor = compute_dcor_matrix(data_fake)

        corr_loss = torch.norm(real_dcor - fake_dcor, p='fro') / (real_dcor.numel()**0.5 + 1e-9)
        
        return corr_loss


    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.
        
        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)
        if not discrete_columns:
            self.has_cond = 0
        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead'
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions
        self.cond_dim = self._data_sampler.dim_cond_vec() 
        # self._generator = Generator(
        #     self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        # ).to(self._device)

        # discriminator = Discriminator(
        #     data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        # ).to(self._device)

        self._generator = ETransformerGenerator(
            noise_dim = self._embedding_dim,
            cond_dim = self.cond_dim,
            embedding_dim=self._embedding_dim + self._data_sampler.dim_cond_vec(),
            generator_dim=self._generator_dim,
            data_dim=data_dim,
            dropout=self.dropout,
            num_first=self.num_first,
            batch_first=self.batch_first,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            hidden_dim = self.hidden_dim
        ).to(self._device)
        discriminator = ETransformerDiscriminator(
            input_dim=data_dim + self._data_sampler.dim_cond_vec(),
            discriminator_dim=self._discriminator_dim,
            dropout=self.dropout,
            num_first=self.num_first,
            batch_first=self.batch_first,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            hidden_dim = self.hidden_dim
        ).to(self._device)

        # self._generator = MLPGenerator(
        # embedding_dim=self._embedding_dim + self._data_sampler.dim_cond_vec(),
        # data_dim=data_dim,         
        # n_layers_hidden=3,    
        # n_units_hidden=256,   
        # nonlin="leaky_relu",   
        # residual=True,         
        # dropout=0.2,         
        # batch_norm=True       
        # ).to(self._device)
        # self._generator = EnhancedTransformerGenerator(
        #     cond_dim=self.cond_dim,
        #     noise_dim=self._embedding_dim, 
        #     embedding_dim=self._embedding_dim + self.cond_dim,
        #     generator_dim=self._generator_dim,
        #     data_dim=data_dim,
        #     num_heads=self.num_heads,
        #     num_layers=self.num_layers
        # ).to(self._device)

        # optimizerG = optim.NAdam(
        #     self._generator.parameters(),
        #     lr=1e-4, 
        #     betas=(0.9, 0.999), 
        #     weight_decay=1e-5
        # )

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))
        print(epoch_iterator)
        print(f"Hype-parameter:[dropout:{self.dropout},num_first:{self.num_first},batch_first:{self.batch_first},num_heads:{self.num_heads},num_layers:{self.num_layers},hidden_dim:{self.hidden_dim}]")
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            print(f"epoch!{i}")
            #print("newtype2")
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty_pac(
                        real_cat, fake_cat, self._device, lambda_=10
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                    optimizerD.zero_grad(set_to_none=False)
                    # from torch.backends.cuda import sdp_kernel
                    # with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False) :
                    #with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy
                #if i >= 50:
                    #lambda_corr = 0.01  
                lambda_corr = 0.1 * (1 - i / epochs) 
                #lambda_corr = 0.08
                    #lambda_corr = min(2.0, 0.1 + i/50) 
                    #corr_loss = self._custom_corr_loss(real_cat, fake_cat.detach(),has_cond=self.has_cond)

                corr_loss = self._dcor_corr_loss(real_cat, fake_cat.detach(),has_cond=self.has_cond)
                    #corr_loss = self.dcor_new_loss(real_cat, fake_cat.detach(),has_cond=self.has_cond)

                    #print(corr_loss)
                    #corr_loss = self._custom_mix_corr_loss(real_cat, fake_cat.detach())
                loss_g += lambda_corr*corr_loss 
                corr_loss_value = corr_loss.item()
                #else:
                #    corr_loss_value = 0
                #corr_loss_value = 0
                #corr_loss = self._custom_corr_loss(real_cat, fake_cat.detach(),has_cond=self.has_cond)
                # corr_loss_value = corr_loss.item()

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
                'Corr Loss': [corr_loss_value],
            })
            self.loss_values = pd.concat([self.loss_values, epoch_loss_df], ignore_index=True)

            current_epoch_loss = self.loss_values.loc[self.loss_values['Epoch'] == i]
            print(f"Epoch {i} - Generator Loss: {current_epoch_loss['Generator Loss'].values[0]:.4f}, "
                  f"Discriminator Loss: {current_epoch_loss['Discriminator Loss'].values[0]:.4f}, "
                  f"Corr Loss: {current_epoch_loss['Corr Loss'].values[0]:.4f}")

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(
                        gen=current_epoch_loss['Generator Loss'].values[0],
                        dis=current_epoch_loss['Discriminator Loss'].values[0]
                    )
                )

        self._plot_loss_convergence("C:/Users/26332/pic/loss_convergence.png")

    def _plot_loss_convergence(self, save_path="C:/Users/26332/pic/loss_convergence.png"):
        """Plot the convergence of generator and discriminator losses."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.loss_values['Epoch'], self.loss_values['Generator Loss'], label='Generator Loss', color='blue')
        plt.plot(self.loss_values['Epoch'], self.loss_values['Discriminator Loss'], label='Discriminator Loss', color='red')
        plt.plot(self.loss_values['Epoch'], self.loss_values['Corr Loss'], label='Corr Loss', color='green')

        plt.title('Loss Convergence During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Loss convergence plot saved to {save_path}")
            
            # if not self.loss_values.empty:
            #     self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
            #         drop=True
            #     )
            # else:
            #     self.loss_values = epoch_loss_df

            # if self._verbose:
            #     epoch_iterator.set_description(
            #         description.format(gen=generator_loss, dis=discriminator_loss)
            #     )
