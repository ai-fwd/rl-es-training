import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class JEPAModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        action_dim: int,
        metadata_dim: int = 1,  # e.g. energy
        tau: float = 0.995,
    ):
        super().__init__()
        self.tau = tau
        self.latent_dim = latent_dim

        # 1. Encoders
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # 2. Transition Head: (z_t, a_t) -> z_hat_{t+1} (in target space)
        self.transition_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # 3. Learn how the latent structure JEPA captures maps to the agent's energy (passed as metadata)
        self.probe = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, metadata_dim),
        )

        # 4. Decoder (for debug/vis only)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def update_target_encoder(self):
        """EMA update for target encoder."""
        with torch.no_grad():
            # This creates a teacher that updates slowly, giving a more stable representation for the online network to match.
            for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                target_param.data.mul_(self.tau).add_(param.data, alpha=1 - self.tau)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target_encoder(x)

    def imagine(self, z: torch.Tensor, action_one_hot: torch.Tensor) -> torch.Tensor:
        """Predict next target latent from current online latent and action."""
        x = torch.cat([z, action_one_hot], dim=-1)
        return self.transition_head(x)

    def get_validity(self, z: torch.Tensor, z_hat_next: torch.Tensor) -> torch.Tensor:
        """Compute validity score: exp(-||z - z_hat||^2)."""
        dist = torch.norm(z - z_hat_next, p=2, dim=-1).pow(2)
        return torch.exp(-dist)

    def compute_losses(
        self,
        obs: torch.Tensor,
        action_one_hot: torch.Tensor,
        next_obs: torch.Tensor,
        next_metadata: torch.Tensor,
    ) -> dict:
        """Compute losses:
            JEPA - latent representation z of states
            Transition - model that maps (z_t, a_t) -> z_pred_next
            Probe - maps z_next -> metadata (e.g. energy)
        """
        
        # Forward passes
        z_t = self.encoder(obs)
        with torch.no_grad(): # turns the target encoder into a slow teacher for encoder
            z_next = self.target_encoder(next_obs)
        
        z_pred_next = self.transition_head(torch.cat([z_t, action_one_hot], dim=-1))
        
        # predict metadata from target embedding (i.e. energy at t+1)
        pred_metadata = self.probe(z_next.detach()) # detach to avoid gradients flowing into target encoder
        
        # Transition Loss (MSE)
        loss_trans = F.mse_loss(z_pred_next, z_next)

        # VICReg Regularization (Variance + Covariance)
        def vicreg_reg(x):
            # Variance
            std = torch.sqrt(x.var(dim=0) + 1e-4)
            std_loss = torch.mean(F.relu(1 - std))
            # Covariance
            x = x - x.mean(dim=0)
            cov = (x.T @ x) / (x.size(0) - 1)
            off_diag = cov.flatten()[:-1].view(x.size(1) - 1, x.size(1) + 1)[:, 1:].flatten()
            cov_loss = off_diag.pow(2).sum() / x.size(1)
            return std_loss, cov_loss

        # Apply VICReg to both predicted and target batches
        # Without this kind of regularizer the network could minimize loss_trans by collapsing everything to a constant vector
        if z_t.size(0) > 1:
            std_z, cov_z = vicreg_reg(z_pred_next)
            std_zp, cov_zp = vicreg_reg(z_next)
            loss_vicreg = (std_z + std_zp) + (cov_z + cov_zp)
        else:
            loss_vicreg = torch.tensor(0.0)

        # Probe Loss
        loss_probe = F.mse_loss(pred_metadata, next_metadata)
        
        # Combine to get total Loss       
        total_loss = loss_trans + 0.1 * loss_vicreg + loss_probe

        return {
            "loss": total_loss,
            "loss_trans": loss_trans.item(),
            "loss_vicreg": loss_vicreg.item() if isinstance(loss_vicreg, torch.Tensor) else loss_vicreg,
            "loss_probe": loss_probe.item(),
        }
