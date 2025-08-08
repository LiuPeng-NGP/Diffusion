import torch
import torch.nn as nn

def logit_normal_timestep_sample(P_mean: float, P_std: float, num_samples: int, device: torch.device) -> torch.Tensor:
    """
    Samples from a logit-normal distribution, as used in the official MeanFlow code.
    """
    rnd_normal = torch.randn((num_samples,), device=device)
    time = torch.sigmoid(rnd_normal * P_std + P_mean)
    time = torch.clip(time, min=0.0, max=1.0)
    return time


def sample_two_timesteps(num_samples: int, device: torch.device, P_mean_t, P_std_t, P_mean_r, P_std_r, ratio):
    """
    Sampler (t, r): Independently samples t and r using the logit-normal strategy,
    with post-processing to ensure t >= r and to control the probability of t==r.
    This is based on the 'v1' sampler from the official MeanFlow implementation.
    """
    # Step 1: sample two independent timesteps from the logit-normal distribution
    t = logit_normal_timestep_sample(P_mean_t, P_std_t, num_samples, device=device)
    r = logit_normal_timestep_sample(P_mean_r, P_std_r, num_samples, device=device)

    # Step 2: make t and r different with a probability of `ratio`
    prob = torch.rand(num_samples, device=device)
    # If prob >= ratio, the mask is True, and r is set to t
    mask = prob < (1 - ratio) 
    r = torch.where(mask, t, r)

    # Step 3: ensure t >= r by taking the minimum
    r = torch.minimum(t, r)

    return t, r


class MeanFlow(nn.Module):
    def __init__(self, model, P_mean_t=-0.6, P_std_t=1.6, P_mean_r=-4.0, P_std_r=1.6, ratio=0.75, norm_eps=1e-3, norm_p=0.75):
        """
        Initializes the MeanFlow framework. The instance is tied to the provided model.
        
        Args:
            model: The neural network model (e.g., a U-Net).
            P_mean_t, P_std_t: Parameters for the logit-normal distribution for 't'.
            P_mean_r, P_std_r: Parameters for the logit-normal distribution for 'r'.
            ratio: Probability of sampling r different from t.
            norm_eps: Epsilon for the adaptive weighting.
            norm_p: Power for the adaptive weighting.
        """
        super().__init__()
        self.model = model
        
        # Sampler parameters
        self.P_mean_t = float(P_mean_t)
        self.P_std_t = float(P_std_t)
        self.P_mean_r = float(P_mean_r)
        self.P_std_r = float(P_std_r)
        self.ratio = float(ratio)

        # Adaptive weighting parameters
        self.norm_eps = float(norm_eps)
        self.norm_p = float(norm_p)

    def loss(self, x, class_labels=None):
        """
        Computes the MeanFlow loss using self.model.
        """
        device = x.device
        batch_size = x.shape[0]
        
        noise = torch.randn_like(x).to(device)
        
        # --- MODIFIED: Use the official logit-normal sampler ---
        t, r = sample_two_timesteps(
            num_samples=batch_size,
            device=device,
            P_mean_t=self.P_mean_t,
            P_std_t=self.P_std_t,
            P_mean_r=self.P_mean_r,
            P_std_r=self.P_std_r,
            ratio=self.ratio
        )
        
        t_4d = t.view(-1, 1, 1, 1)
        r_4d = r.view(-1, 1, 1, 1)

        z = (1 - t_4d) * x + t_4d * noise
        v = noise - x

        # Define the network function to be compatible with torch.func.jvp
        def u_func(z_in, t_in, r_in):
            # jvp promotes t_in and r_in to 4D. We must flatten them back to 1D
            # before passing them to the U-Net model.
            t_1d = t_in.view(-1)
            h_1d = (t_in - r_in).view(-1)
            
            # Pass the corrected 1D tensors to the model in a tuple
            return self.model.module(z_in, (t_1d, h_1d), class_labels)

        # jvp expects primals (z, t, r) and tangents (v, dtdt, drdt)
        # to have matching shapes. The original 1D t and r are correct here.
        dtdt = torch.ones_like(t)
        drdt = torch.zeros_like(r)

        with torch.amp.autocast("cuda", enabled=False):
            u_pred, dudt = torch.func.jvp(u_func, (z, t, r), (v, dtdt, drdt))
        
            u_tgt = (v - (t_4d - r_4d) * dudt).detach()

            loss_sq = (u_pred - u_tgt)**2
            loss_sum = loss_sq.sum(dim=(1, 2, 3))

            adp_wt = (loss_sum.detach() + self.norm_eps) ** self.norm_p
            loss = loss_sum / adp_wt

            return loss.mean()

    @torch.no_grad()
    def sample(self, latents, class_labels=None):
        """
        Generates samples in a single step using the instance's self.model.
        """
        self.model.eval()
        t = torch.ones(latents.shape[0], device=latents.device)
        h = torch.ones(latents.shape[0], device=latents.device)
        
        u_pred = self.model(latents, (t, h), class_labels)
        
        z_0 = latents - u_pred
        return z_0