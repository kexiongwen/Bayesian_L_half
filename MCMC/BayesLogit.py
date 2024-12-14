import torch 
from tqdm import tqdm
from polyagamma import random_polyagamma
from MCMC.GLshrinkage import shrinkage1,shrinkage2

def Bayes_Logit(Y, X, alpha, a, b, M = 10000, burn_in = 10000):

    N,P = X.size()
    
    X = X.to(torch.float32)
    Y = Y.to(torch.float32)
    
    device = X.device
        
    #Initialization
    beta_samples = []
    beta_sample = 0.1 * torch.randn(P, 1, device = device)
    omega_sample = torch.ones_like(Y)
    
    for i in tqdm(range(1, M + burn_in)):
        
        # Sample Global-local shrinkage parameter
        
        if alpha == 1:
            G = shrinkage1(beta_sample, a, b)
        elif alpha == 2:
            G = shrinkage2(beta_sample, a, b)
        
        #Sample beta

        D = omega_sample.sqrt()
        XTD = X.T * D.T
        DY = (Y - 0.5) / D        
        GXTD = G * XTD
        
        if P > N:    
            
            mu = torch.randn_like(beta_sample) * G  
            U = DY - XTD.T @ mu - torch.randn_like(Y)         
            v = torch.linalg.solve(torch.eye(N, device = device) + GXTD.T @ GXTD,U)
            beta_sample = mu + G * GXTD @ v
            
        else:
            
            beta_sample = torch.linalg.solve(GXTD @ GXTD.T + torch.eye(P, device = device),GXTD @ DY + GXTD @ torch.randn_like(Y)\
                + torch.randn_like(beta_sample)) * G
                    
        # Sample omega
        omega_sample = torch.from_numpy(random_polyagamma(z = (X @ beta_sample).cpu().numpy())).type(torch.float32).to(device)
        
        if (i + 1) > burn_in:
            
            beta_samples.append(beta_sample)
        
    return torch.stack(beta_samples).squeeze()