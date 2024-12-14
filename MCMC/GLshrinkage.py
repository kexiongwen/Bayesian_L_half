import torch
from torch.distributions.gamma import Gamma

def inv_gauss(mu):
        
    ink = mu * torch.randn_like(mu).pow(2)
    a = 1 + 0.5 * (ink - ((ink + 2).square() - 4).sqrt())
    
    return torch.where((1 / (1 + a)) >= torch.rand_like(mu), mu * a,mu / a)


def GIG(mu):
    
    ink = (1e1 * torch.randn_like(mu)).square()/(1e2 * mu)
    a = 1 + 0.5 * (ink - ((ink + 2).square() - 4).sqrt())
    
    return torch.where((1 / (1 + a)) >= torch.rand_like(mu), mu / a, mu * a)


def shrinkage1(beta_sample, a = 1, b = 1):
        
    # Sample lam
    ink = beta_sample.abs().sqrt()

    lam = Gamma(2 * beta_sample.shape[0] + a,ink.sum() + b).sample()
    
    ink = ink.mul_(lam)
    
    # Sample V
    v = 2 / inv_gauss(1 / ink)
    
    # Sample tau
    tau = v / inv_gauss(v / ink.pow(2)).sqrt()
    
    D = torch.where(torch.isinf(tau), 500, tau / lam.square())
    
    return torch.where(torch.isnan(tau), 500, D)


def shrinkage2(beta_sample, a = 1, b = 1):
    
    # Sample lam
    ink = beta_sample.abs().pow(0.25)
    lam = Gamma(4 * beta_sample.shape[0] + a,ink.sum() + b).sample()
    
    ink1 = (lam * ink).pow(4)
    ink2 = ink1.sqrt() / lam
    
    # Sampling v2
    v2 = 2 / inv_gauss(1 / (lam * ink))
    
    # Sampling v1
    v1 = 2 * v2.square() / inv_gauss(v2 / ink2)
    
    # Sampling tau
    tau = v1 / inv_gauss(v1 / ink1).sqrt()     
    
    D = torch.where(torch.isinf(tau) ,2000, tau / lam.square())
    
    return torch.where(torch.isnan(tau) ,2000, D)
