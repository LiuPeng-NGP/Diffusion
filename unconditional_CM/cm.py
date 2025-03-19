import torch
import torch.nn as nn
import numpy as np

class WeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, t):
        return self.mlp(t)
    
class sCM():
    def __init__(self, model, sigma_data=0.5, P_mean=-1.0, P_std=1.4, sigma_max=80):
        
        self.model = model                  # F_θ
        self.sigma_data = sigma_data        # σ_d
        self.P_mean = P_mean                # Mean for τ sampling
        self.P_std = P_std                  # Std for τ sampling
        self.sigma_max = sigma_max          # Maximum σ
        self.step = 0                       # Training step counter
        self.weight_model = WeightModel().to(next(model.parameters()).device) # w_φ(t), learnable weight function

    def loss(self, x0, class_labels=None):
        
        # Sample τ ~ N(P_mean, P_std^2), compute σ = e^τ, t = arctan(σ / σ_d)
        batch_size = x0.shape[0]
        tau = torch.randn(batch_size, device=x0.device).reshape(-1, 1, 1, 1) * self.P_std + self.P_mean
        e_tau = torch.exp(tau)
        t = torch.arctan(e_tau / self.sigma_data)  # Shape: [batch_size, 1, 1, 1]

        # Sample z ~ N(0, σ_d^2 I), compute x_t = cos(t) x0 + sin(t) z
        z = torch.randn_like(x0) * self.sigma_data
        x_t = torch.cos(t) * x0 + torch.sin(t) * z
        
        # Compute dx_t/dt using analytical derivative
        dxt_dt = torch.cos(t) * z - torch.sin(t) * x0
        
        # Warmup factor r = min(1, step / 10000)
        r = min(1.0, self.step / 10000)
        # Wrapper for model to compute $ F_\theta $
        def model_wrapper(scaled_x_t, t_val):
            pred = self.model.module(scaled_x_t, t_val.flatten(), class_labels=class_labels)
            return pred
        
        # Compute F_θ(x_t / σ_d, t) and F_θ^- (detached) directly
        scaled_x_t = x_t / self.sigma_data

        # Jacobian-vector product (JVP) for tangent normalization (stabilization technique)
        cos_t_sin_t = torch.cos(t) * torch.sin(t)
        v_x = cos_t_sin_t * dxt_dt  # ∂(x_t / σ_d) direction, scaled by cos(t) sin(t) σ_d
        v_t = cos_t_sin_t * self.sigma_data  # ∂t direction, scaled by cos(t) sin(t) σ_d
        
        # Compute JVP to get $ \nabla F_\theta \cdot v $ (aligns with gradient stabilization)
        F_theta, scaled_dF_dt = torch.func.jvp(
            model_wrapper,
            (scaled_x_t, t),
            (v_x, v_t),
        )
        F_theta_minus = F_theta.detach()  # F_θ^-
        scaled_dF_dt = scaled_dF_dt.detach()
        
        # Compute gradient $ g $ for consistency loss (modified from paper's Eq. 6)
        g = -torch.cos(t)**2 * (self.sigma_data * F_theta_minus - dxt_dt)
        # g -= r * torch.cos(t) * torch.sin(t) * (x_t + self.sigma_data * dF_dt)
        # g -= r * (torch.cos(t) * torch.sin(t) * x_t + self.sigma_data * dF_dt)
        g -= r * (torch.cos(t) * torch.sin(t) * x_t + scaled_dF_dt)
        
        # Gradient normalization (stabilization technique)
        g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
        g_norm = g_norm * np.sqrt(g_norm.numel() / g.numel())  # Adjust norm magnitude
        g = g / (g_norm + 0.1)  # Stabilize division
        
        # Compute aligned loss with learnable weights w_φ(t)
        t_input = t[:, 0, 0, 0].unsqueeze(1)
        w_phi = self.weight_model(t_input).squeeze()  # [batch_size]
        error = F_theta - F_theta_minus - g  # Prediction error
        mse = error.pow(2).mean(dim=(1, 2, 3))  # Mean squared error per sample
        
        # Weighted loss for sCM paper
        loss_per_sample = torch.exp(w_phi) * mse - w_phi  # Weighted loss
        
        # Weighted loss for sCM-mnist and EDM 2
        # loss_per_sample = (e_tau/ torch.exp(w_phi)) * mse + w_phi
        
        loss = loss_per_sample.mean()  # Average over batch
        
        self.step+=1
        
        return loss
    
    @torch.no_grad()
    def sample(self, latents, class_labels=None, sigma_max=80):
        
        latents *= self.sigma_data 
        
        sigma_max = min(sigma_max, self.sigma_max)
        t = torch.arctan(torch.tensor([sigma_max / self.sigma_data], device=latents.device))
        
        # # Initial noisy sample x_t
        x_t = torch.sin(t) * latents
        
        # Model prediction F_θ(x_t / σ_d, t)
        scaled_x_t = x_t / self.sigma_data
        t_flat = t.repeat(latents.shape[0])
        
        pred = self.model(scaled_x_t, t_flat, class_labels=class_labels)
        
        # # Compute denoised sample x0 = cos(t) x_t - sin(t) σ_d F_θ
        x0 = torch.cos(t) * x_t - torch.sin(t) * self.sigma_data * pred
        
        return x0
    
