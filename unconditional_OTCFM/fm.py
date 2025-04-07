import math
from functools import partial
import numpy as np
import torch
from tqdm import tqdm
import ot as pot

# Reuse OTPlanSampler from your earlier code
class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an squared L2 OT plan with
    different implementations of the plan calculation."""
    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost=False,
        **kwargs,
    ):
        if method == "exact":
            self.ot_fn = pot.emd
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.kwargs = kwargs

    def get_map(self, x0, x1):
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        return p

    def sample_map(self, pi, batch_size):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch_size)
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1):
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0])
        return x0[i], x1[j]

# Modified FM class for ExactOptimalTransportConditionalFlowMatcher
class FM:
    def __init__(self, model, sigma_min=1e-4, ln=True):
        """
        Initializes the Flow Matching framework for ExactOptimalTransportConditionalFlowMatcher.

        Args:
            model (nn.Module): The neural network v_theta(x, t) that predicts the vector field.
            sigma_min (float): Standard deviation of the noise added to x_t (sigma in OT-CFM).
            ln (bool): If True, samples t using a logistic-normal transformation.
        """
        self.model = model
        self.sigma_min = float(sigma_min)  # This is sigma in the OT-CFM formulation
        self.ln = ln
        self.ot_sampler = OTPlanSampler(method="exact")  # Exact OT for pairing

    def loss(self, x1, class_labels=None):
        """
        Computes the Flow Matching loss using OT pairing.

        Args:
            x1 (torch.Tensor): Samples from the data distribution p_1.
            class_labels (torch.Tensor, optional): Conditioning labels.

        Returns:
            torch.Tensor: Mean loss over the batch.
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Sample t
        if self.ln:
            nt = torch.randn((batch_size,), device=device)
            t = torch.sigmoid(nt)  # t in [0, 1], skewed toward boundaries
        else:
            t = torch.rand((batch_size,), device=device) * (1 - 2 * self.sigma_min) + self.sigma_min  # t in [sigma_min, 1-sigma_min]

        # Sample x0 (noise) and pair with x1 using OT
        x0 = torch.randn_like(x1)  # x0 ~ N(0, I), source distribution
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)  # OT pairing

        # Generate x_t using the OT-CFM formulation
        t_expanded = t.view(-1, 1, 1, 1)
        mu_t = (1 - t_expanded) * x0 + t_expanded * x1  # Mean of the probability path
        epsilon = torch.randn_like(x0)  # Noise for x_t
        x_t = mu_t + self.sigma_min * epsilon  # x_t = mu_t + sigma * epsilon

        # Compute the true vector field
        u_t = x1 - x0  # u_t(x1 | x0) = x1 - x0, constant in OT-CFM

        # Predict the vector field with the model
        v_pred = self.model(x_t, t, class_labels)

        # Compute the loss
        loss = torch.mean((v_pred - u_t) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, latents, class_labels=None, null_cond=None, num_steps=200, cfg=2.0):
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
                v_cond = v_uncond + cfg * (v_cond - v_uncond)  # Classifier-free guidance

            x = x + dt * v_cond  # Euler step for ODE integration

        return x