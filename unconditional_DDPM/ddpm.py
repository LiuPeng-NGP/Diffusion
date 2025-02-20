import torch
from tqdm import tqdm

class DDPM():
    def __init__(self, model, T=1000):
        self.model = model
        
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.T = T

        # Correct concatenation of beta_t
        self.betas = torch.cat([torch.tensor([0.0]), torch.linspace(self.beta_start, self.beta_end, self.T)])
        self.sqrt_betas = torch.sqrt(self.betas)
        self.alphas = 1 - self.betas
        self.log_alphas = torch.log(self.alphas)
        self.alphas_bar = torch.cumsum(self.log_alphas, dim=0).exp()

        # Training Algorithm 1
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar) 
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alphas_bar)

        # Sampling Algorithm 2
        self.one_over_sqrt_alphas = 1 / torch.sqrt(self.alphas)
        self.one_minus_alpha_over_sqrt_one_minus_alpha_bar = (1 - self.alphas) / self.sqrt_one_minus_alphas_bar 

    def perturb(self, x):
        t = torch.randint(1, self.T + 1, (x.shape[0], )).to(x.device)
        noise = torch.randn_like(x)
        x_noised = (self.sqrt_alphas_bar.to(x.device)[t, None, None, None] * x +
                self.sqrt_one_minus_alphas_bar.to(x.device)[t, None, None, None] * noise)
        return x_noised, t, noise

    def loss(self, x):
        x_noised, t, noise = self.perturb(x)
        loss = (noise - self.model(x_noised, t/self.T))**2 
        return loss.mean()

    def sample(self, latents, no_tqdm=False):
        size = latents.shape
        for i in tqdm(range(self.T, 0, -1), disable=no_tqdm):
            t = torch.tensor([i / self.T]).to(latents.device).repeat(latents.shape[0])
            z = torch.randn(*size).to(latents.device) if i > 1 else 0
            eps, _ = self.pred_eps_(latents, t, self.alphas_bar[i])

            # Algorithm 2
            mean = self.one_over_sqrt_alphas[i] * (latents - self.one_minus_alpha_over_sqrt_one_minus_alpha_bar[i] * eps)
            variance = self.sqrt_betas[i] # Let variance sigma = sqrt_beta
            latents = mean + variance * z

        x_0 = latents
        return x_0

    def pred_eps_(self, x, t, alpha, clip_x=True):
        def pred_eps_from_x0(x0):
            return (x - x0 * alpha.sqrt()) / (1 - alpha).sqrt()

        def pred_x0_from_eps(eps):
            return (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()
        
        eps = self.model(x, t).float()
        denoised = pred_x0_from_eps(eps)

        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
            eps = pred_eps_from_x0(denoised)
        return eps, denoised