def forward_diffusion(x_0, t, betas):
    noise = torch.randn_like(x_0)
    alpha = 1 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)
    x_t = torch.sqrt(alpha_bar[t]) * x_0 + torch.sqrt(1 - alpha_bar[t]) * noise
    return x_t, noise

def reverse_diffusion(x_t, t, model, betas):
    noise_pred = model(x_t, t)  # Predict the noise
    alpha = 1 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)
    x_0_pred = (x_t - torch.sqrt(1 - alpha_bar[t]) * noise_pred) / torch.sqrt(alpha_bar[t])
    return x_0_pred
