# Here we add the auxiliary age and sex prediction heads to the SimCLR model. 
# The age and sex predictions are made from the representation (h_i and h_j) obtained from the encoder.

import torch.nn as nn

from simclr.modules.identity import Identity
from torch.utils.checkpoint import checkpoint

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )
        
        # Age prediction head (from representation)
        self.age_head = nn.Sequential(
            nn.Linear(self.n_features, 256), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # Single output for age
        )
        
        # Sex prediction head (from representation)
        self.sex_head = nn.Sequential(
            nn.Linear(self.n_features, 256), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # Single output for sex
        )

    def forward(self, x_i, x_j):
        h_i = checkpoint(self.encoder, x_i, use_reentrant=False)
        h_j = checkpoint(self.encoder, x_j, use_reentrant=False)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        # Now, we predict age from the representation (h_i and h_j should be similar, so we can use either)
        # but for robustness, we predict from both and average the predictions
        age_i = self.age_head(h_i)
        age_j = self.age_head(h_j)
        avg_age = (age_i + age_j) / 2.0
        
        # In a similar manner, predict sex from the representations
        sex_i = self.sex_head(h_i)
        sex_j = self.sex_head(h_j)
        avg_sex_prob = (sex_i + sex_j) / 2.0
        
        return z_i, z_j, avg_age, avg_sex_prob