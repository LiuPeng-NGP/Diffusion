import torch
from tqdm import tqdm

class FM():
    def __init__(self, model, sigma_min=1e-4, ln=True):
        """
        Initializes the Flow Matching framework.

        Args:
            model (nn.Module): The neural network v_theta(x, t) that predicts the vector field.
            sigma_min (float): Standard deviation of the noise distribution p_0 ~ N(0, sigma_min^2 I).
            ln (bool): If True, samples t using a logistic-normal transformation.
        """
        self.model = model
        self.sigma_min = float(sigma_min)
        self.ln = ln

    def loss(self, x1, class_labels=None):
        """
        Computes the Flow Matching loss.
        
        Args:
            x1 (torch.Tensor): Samples from the data distribution p_1.
            class_labels (torch.Tensor, optional): Conditioning labels.
        
        Returns:
            torch.Tensor: Mean loss over the batch.
        """
        batch_size = x1.shape[0]
        device = x1.device

        if self.ln:
            nt = torch.randn((batch_size,), device=device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((batch_size,), device=device) * (1 - 2 * self.sigma_min) + self.sigma_min
        
        t_expanded = t.view(-1, 1, 1, 1)
        x0 = torch.randn_like(x1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        u_t = x1 - x0
        v_pred = self.model(x_t, t, class_labels)
        loss = torch.mean((v_pred - u_t) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, latents, class_labels=None, null_cond=None, num_steps=50, cfg=2.0):
        """
        Generates samples using the learned vector field via ODE integration.

        Args:
            latents (torch.Tensor): Initial noise samples from N(0, I).
            class_labels (torch.Tensor, optional): Conditioning labels.
            null_cond (torch.Tensor, optional): Unconditioned class labels for classifier-free guidance.
            num_steps (int): Number of integration steps.
            cfg (float): Classifier-free guidance strength.
        
        Returns:
            torch.Tensor: Generated samples x(T1).
        """
        device = latents.device
        batch_size = latents.shape[0]
        dt = 1.0 / num_steps
        x = latents  # Start from noise
        
        for step in tqdm(range(num_steps), desc="Flow Matching Sampling", leave=False):
            t = step / num_steps  # t from 0 to (num_steps-1)/num_steps
            t = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            v_cond = self.model(x, t, class_labels)
            
            if null_cond is not None:
                v_uncond = self.model(x, t, null_cond)
                v_cond = v_uncond + cfg * (v_cond - v_uncond)
            
            x = x + dt * v_cond  # Integrate forward
        
        return x