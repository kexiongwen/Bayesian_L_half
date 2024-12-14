import torch 
from tqdm import tqdm
from MCMC.GLshrinkage import inv_gauss,shrinkage1,shrinkage2

def BQR(Y, X, alpha, a, b, Q = 0.5, M = 10000, burn_in = 10000):

    N,P = X.size()
    
    X = X.to(torch.float64)
    Y = Y.to(torch.float64)
    
    device = X.device
    
    #Initialization
    beta_samples = []
    beta_sample = 0.1 * torch.randn(P, 1, device = device, dtype = torch.float64)
    omega_sample = torch.ones_like(Y)
    c1 = (0.5 * Q * (1 - Q)) ** 0.5
    c2 = (1 - 2 * Q) / (Q * (1 - Q))
    
    for i in tqdm(range(1,M + burn_in)):
        
        # Sample Global-local shrinkage parameter
        
        if alpha == 1:
            G = shrinkage1(beta_sample, a, b)
        elif alpha == 2:
            G = shrinkage2(beta_sample, a, b)
    
        #Sample beta
        D = c1 * omega_sample.sqrt()
        XTD = X.T * D.T
        DY = D * (Y - c2 / omega_sample)         
        GXTD = G * XTD
        
        if P > N:
            
            mu = torch.randn_like(beta_sample) * G  
            U = DY - XTD.T @ mu - torch.randn_like(Y)    
            v = torch.linalg.solve(torch.eye(N, device = device, dtype = torch.float64) + GXTD.T @ GXTD,U)
            beta_sample = mu + G * GXTD @ v 
        else:
            beta_sample = torch.linalg.solve(GXTD @ GXTD.T + torch.eye(P, device = device, dtype = torch.float64),GXTD @ DY \
                + GXTD @ torch.randn_like(Y) + torch.randn_like(beta_sample)) * G
        
        #Sample omega
        omega_sample = (inv_gauss(2 / torch.maximum(torch.abs((Y - X @ beta_sample).ravel()),torch.tensor(1e-3, device = device, dtype = torch.float64))) \
            / (2 * Q * (1 - Q))).view(-1,1)
        
        if (i+1) > burn_in:
            
            beta_samples.append(beta_sample)
    
    return torch.stack(beta_samples).squeeze()





