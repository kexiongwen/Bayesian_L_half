import torch 
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from MCMC.GLshrinkage import shrinkage1,shrinkage2

def Bayes_regression(Y, X, alpha, a, b, M = 10000, burn_in = 10000):
    
    X = X.to(torch.float32)
    Y = Y.to(torch.float32)
    
    N,P = X.size()
    device = X.device
    
    w = 0.5
    
    ## Initialization
    beta_samples = []
    sigma2_samples = []
    beta_sample = torch.randn(P, 1, device = device)
    sigma2_sample = torch.ones(1, device = device)

    if P < N:
        
        XTX = X.T @ X
        XTY = X.T @ Y
    
    ## MCMC Loop
    
    for i in tqdm(range(1, M + burn_in)):
        
        # Sample Global-local shrinkage parameter
        
        if alpha == 1:
            D = shrinkage1(beta_sample, a, b)
        elif alpha == 2:
            D = shrinkage2(beta_sample, a, b)
            
        # Sample beta
        
        sigma = sigma2_sample.sqrt()
        
        if P > N:
        
            DXT = D * X.T
            mu = torch.randn_like(beta_sample) * D
            omega = torch.eye(N, device = device) + DXT.T @ DXT / sigma2_sample
            v = torch.linalg.solve(omega, Y / sigma - X @ mu / sigma - torch.randn_like(Y))
            beta_sample = mu + D * DXT @ v / sigma
            
        else:
            
            beta_sample = torch.linalg.solve(D * XTX * D.T / sigma2_sample + torch.eye(P, device = device),D * XTY / sigma2_sample\
                + D * X.T / sigma @ torch.randn_like(Y) + torch.randn_like(beta_sample)) * D
            
        #Sample sigma2
        
        rss = Y - X @ beta_sample
        sigma2_sample = 0.5 * (w + rss.T @ rss) / Gamma((w+N) / 2, torch.tensor(1, dtype = torch.float32, device = device)).sample()
        
        if (i+1) > burn_in:
            
            beta_samples.append(beta_sample)
            sigma2_samples.append(sigma2_sample)
        
    return torch.stack(beta_samples).squeeze(),torch.stack(sigma2_samples).squeeze()