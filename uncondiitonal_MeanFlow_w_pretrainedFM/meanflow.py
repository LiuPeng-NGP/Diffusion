# meanflow.py
import torch
import torch.nn as nn

class MeanFlow(nn.Module):
    def __init__(self, model, flow_matching_total_steps=0, t_min=1e-5, norm_eps=1e-5, norm_p=-0.5, **kwargs):
        """
        Initializes the MeanFlow framework. The instance is tied to the provided model.
        The `flow_matching_total_steps` argument controls the two-stage training.
        All other diffusion parameters are passed via kwargs.
        """
        super().__init__()
        self.model = model
        
        # Parameters for Mean Flow loss
        self.t_min = float(t_min)
        self.norm_eps = float(norm_eps)
        self.norm_p = float(norm_p)

        # State for two-stage training
        self.flow_matching_total_steps = flow_matching_total_steps
        self.current_step = 0

    def loss(self, x, class_labels=None):
        """
        Computes the loss. Switches from Flow Matching to Mean Flow loss
        after flow_matching_total_steps have passed.
        """
        # Determine current training stage and compute loss
        if self.training and self.current_step < self.flow_matching_total_steps:
            loss = self.flow_matching_loss(x, class_labels)
        else:
            loss = self.mean_flow_loss(x, class_labels)
        
        # Increment step counter only during training
        if self.training:
            self.current_step += 1
        return loss

    def flow_matching_loss(self, x, class_labels=None):
        """
        Computes the simple Flow Matching loss.
        """
        device = x.device
        batch_size = x.shape[0]
        
        noise = torch.randn_like(x).to(device)
        t = torch.rand(batch_size, device=device)
        t_4d = t.view(-1, 1, 1, 1)

        z = (1 - t_4d) * x + t_4d * noise
        v = noise - x
        
        # For simple flow matching, the second time condition `h` can be equal to `t`
        u_pred = self.model.module(z, (t, t), class_labels)
        
        loss_sq = (u_pred - v)**2
        return loss_sq.mean()

    def mean_flow_loss(self, x, class_labels=None):
        """
        Computes the original MeanFlow loss using self.model.
        """
        device = x.device
        batch_size = x.shape[0]
        
        noise = torch.randn_like(x).to(device)
        
        t = torch.rand(batch_size, device=device) * (1.0 - self.t_min) + self.t_min
        r = torch.rand(batch_size, device=device) * t
        
        t_4d = t.view(-1, 1, 1, 1)
        r_4d = r.view(-1, 1, 1, 1)

        z = (1 - t_4d) * x + t_4d * noise
        v = noise - x

        # Define the network function to be compatible with torch.func.jvp
        def u_func(z_in, t_in, r_in):
            t_1d = t_in.view(-1)
            h_1d = (t_in - r_in).view(-1)
            return self.model.module(z_in, (t_1d, h_1d), class_labels)

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