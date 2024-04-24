from skorch import NeuralNetRegressor
from torch import nn
import torch

"""

Code adapted from: https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/Advanced_Usage.ipynb

Two autoencoder architectures: Autoencoder_1 and Autoencoder_2

Variational Autoencoder is also provided but has not been used for final results.


"""

class Encoder_1(nn.Module):
    def __init__(self, num_units=3, input_size=48):
        super().__init__()
        self.num_units = num_units
        self.input_size = input_size
        
        self.encode = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_units),
            nn.Tanh(),
        )
        
    def forward(self, X):
        encoded = self.encode(X)
        return encoded
    
class Decoder_1(nn.Module):
    def __init__(self, num_units, output_size=48):
        super().__init__()
        self.num_units = num_units
        self.output_size = output_size
        
        self.decode = nn.Sequential(
            nn.Linear(self.num_units, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, self.output_size),
        )
        
    def forward(self, X):
        decoded = self.decode(X)
        return decoded

class AutoEncoder_1(nn.Module):
    def __init__(self, num_units, input_size):
        super().__init__()
        self.num_units = num_units
        self.input_size = input_size

        self.encoder = Encoder_1(num_units=self.num_units, input_size=self.input_size)
        self.decoder = Decoder_1(num_units=self.num_units, output_size=self.input_size)
        
    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded, encoded  # <- return a tuple of two values
    


# ------------  model 2 ------------- # 

class Encoder_2(nn.Module):
    def __init__(self, num_units=3, input_size=48):
        super().__init__()
        self.num_units = num_units
        self.input_size = input_size
        
        self.encode = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 64),            
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_units),
            nn.Tanh(),
        )
        
    def forward(self, X):
        encoded = self.encode(X)
        return encoded
    
class Decoder_2(nn.Module):
    def __init__(self, num_units, output_size=48):
        super().__init__()
        self.num_units = num_units
        self.output_size = output_size
        
        self.decode = nn.Sequential(
            nn.Linear(self.num_units, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, self.output_size),
        )
        
    def forward(self, X):
        decoded = self.decode(X)
        return decoded

class AutoEncoder_2(nn.Module):
    def __init__(self, num_units, input_size):
        super().__init__()
        self.num_units = num_units
        self.input_size = input_size

        self.encoder = Encoder_2(num_units=self.num_units, input_size=self.input_size)
        self.decoder = Decoder_2(num_units=self.num_units, output_size=self.input_size)
        
    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded, encoded  # <- return a tuple of two values
    



class VariationalAutoEncoder(nn.Module):
    def __init__(self, num_units, input_size):
        super().__init__()
        self.num_units = num_units
        self.input_size = input_size

        self.encoder_mean = Encoder_1(num_units=self.num_units, input_size=self.input_size)
        self.encoder_logvar = Encoder_1(num_units=self.num_units, input_size=self.input_size)
        self.decoder = Decoder_1(num_units=self.num_units, output_size=self.input_size)
        self.deterministic = False
        
    def forward(self, X):
        encoded_mean = self.encoder_mean(X)
        encoded_logvar = self.encoder_logvar(X)

        # reparametrization trick
        if not self.deterministic:
            encoded = self.reparametrize(encoded_mean, encoded_logvar)
        else:
            encoded = encoded_mean

        decoded = self.decoder(encoded)
        return decoded, encoded, encoded_mean, encoded_logvar  # <- return a tuple of four values
    
    def reparametrize(self, encoded_mean, encoded_logvar):
        encoded_std = torch.exp(0.5 * encoded_logvar)
        eps = torch.randn_like(encoded_std)
        encoded = encoded_mean + eps * encoded_std
        return encoded

    

class AutoEncoderNet(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, encoded = y_pred  # <- unpack the tuple that was returned by `forward`
        loss_reconstruction = super().get_loss(decoded, y_true, *args, **kwargs)
        loss_unit_sphere = 1e-1 * (torch.norm(encoded).mean() - 1)
        return loss_reconstruction + loss_unit_sphere
    

class VariationalAutoEncoderNet(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        decoded, encoded, encoded_mean, encoded_logvar = y_pred  # <- unpack the tuple that was returned by `forward`
        loss_reconstruction = super().get_loss(decoded, y_true, *args, **kwargs)
        kl_beta = 1e-1
        loss_kl_divergence = kl_beta * (-0.5 * torch.sum(1 + encoded_logvar - encoded_mean.pow(2) - encoded_logvar.exp()) )# KL Divergence between two Gaussian distributions 
        loss_unit_sphere = 1e-2 * (torch.norm(encoded).mean() - 1)
        return loss_reconstruction + loss_kl_divergence + loss_unit_sphere