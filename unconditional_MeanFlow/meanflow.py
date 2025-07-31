# fm.py
import torch
import torch.nn as nn

class MeanFlow(nn.Module):
    def __init__(self, model, t_min=1e-5, norm_eps=1e-5, norm_p=-0.5):
        """
        Initializes the MeanFlow framework. The instance is tied to the provided model.
        """
        super().__init__()
        self.model = model
        
        self.t_min = float(t_min)
        self.norm_eps = float(norm_eps)
        self.norm_p = float(norm_p)

    def loss(self, x, class_labels=None):
        """
        Computes the MeanFlow loss using self.model.
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
            # --- FIX: Reshape the time tensors that jvp has broadcasted ---
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
            loss = loss_sum * adp_wt

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