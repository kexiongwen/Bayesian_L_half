import torch 
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from MCMC.global_local_parm import shrinkage1
from MCMC.global_local_parm import shrinkage2

def Bayes_regression(Y,X,alpha,a,b,M=10000,burn_in=10000):
    
    X=X.to(torch.float32)
    Y=Y.to(torch.float32)
    
    N,P=X.size()
    device=X.get_device()
    
    if device==-1:
        
        device='cpu'
    
    w=0.5
    
    ## Initialization
    beta_sample=torch.randn(P,M+burn_in,device=device)
    sigma2_sample=torch.ones(M+burn_in,device=device)
    
    g=Gamma((w+N)/2,torch.tensor(1,dtype=torch.float32,device=device))
    
    if P<N:
        
        XTX=X.T@X
        XTY=X.T@Y
    
    ## MCMC Loop
    
    for i in tqdm(range(1,M+burn_in)):
        
        # Sample Global-local shrinkage parameter
        
        if alpha==1:
            D=shrinkage1(beta_sample[:,i-1:i],a,b)
        elif alpha==2:
            D=shrinkage2(beta_sample[:,i-1:i],a,b)
            
        # Sample beta
        
        sigma=sigma2_sample[i-1]**0.5
        
        if P>N:
        
            DXT=D*X.T
            mu=torch.randn(P,1,device=device)*D
            omega=torch.eye(N,device=device)+DXT.T@DXT/sigma2_sample[i-1]
            v=torch.linalg.solve(omega,Y/sigma-X@mu/sigma-torch.randn(N,1,device=device))
            beta_sample[:,i:i+1]=mu+D*DXT@v/sigma
            
        else:
            
            beta_sample[:,i:i+1]=torch.linalg.solve(D*XTX*D.T/sigma2_sample[i-1]+torch.eye(P,device=device),D*XTY/sigma2_sample[i-1]+D*X.T/sigma@torch.randn(N,1,device=device)+torch.randn(P,1,device=device))*D
            
        #Sample sigma2
        rss=Y-X@beta_sample[:,i:i+1]
        sigma2_sample[i]=0.5*(w+rss.T@rss)/g.sample()
        
    return beta_sample[:,burn_in:],sigma2_sample[burn_in:]

