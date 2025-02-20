import torch
import numpy as np

#----------------------------------------------------------------------------
# EDM model
class EDM():
    def __init__(self, 
    model,
    p_mean=-1.2, p_std=1.2, sigma_data=0.5, # For training
    sigma_min=0.002, sigma_max=80, rho=7,   # For sampling
    ):
        self.model = model
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def precond(self, x, sigma, class_labels=None, **kwargs):
        # edm preconditioning for input and output
        # https://github.com/NVlabs/edm/blob/main/training/networks.py#L632
        
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        model_output = self.model(c_in * x, c_noise.flatten(), class_labels=class_labels, **kwargs)
        
        return c_skip * x + c_out * model_output
        
    def loss(self, data, class_labels=None, **kwargs):
        # https://github.com/NVlabs/edm/blob/main/training/loss.py#L66
        
        rnd_normal = torch.randn([data.shape[0], 1, 1, 1], device=data.device) # For sigma
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(data) * sigma
        denoised_data = self.precond(data + noise, sigma, class_labels=class_labels, **kwargs)
        loss = weight * ((denoised_data - data) ** 2)
        return loss.mean()
    
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
    @torch.no_grad()
    def sample(self, latents, class_labels=None, randn_like=torch.randn_like, 
               num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
               S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float32) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = self.precond(x_hat, t_hat, class_labels).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.precond(x_next, t_next, class_labels).to(torch.float32)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next