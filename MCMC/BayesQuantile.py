import torch 
from tqdm import tqdm
from torch.distributions.gamma import Gamma
from MCMC.invgauss import inverse_gaussian
from MCMC.global_local_parm import shrinkage1
from MCMC.global_local_parm import shrinkage2


def BQR(Y,X,alpha,a,b,Q=0.5,Accelate=False,M=10000,burn_in=10000):

    N,P=X.size()
    
    X=X.to(torch.float32)
    Y=Y.to(torch.float32)
    
    device=X.get_device()
    
    if device==-1:
        
        device='cpu'
    
    #Initialization
    beta_sample=0.1*torch.randn(P,M+burn_in,device=device)
    beta_tilde=beta_sample[:,0:1]
    omega_sample=torch.ones(N,1,device=device)
    c1=(0.5*Q*(1-Q))**0.5
    c2=(1-2*Q)/(Q*(1-Q))
    
    for i in tqdm(range(1,M+burn_in)):
        
        # Sample Global-local shrinkage parameter
        
        if alpha==1:
            G=shrinkage1(beta_sample[:,i-1:i],a,b)
        elif alpha==2:
            G=shrinkage2(beta_sample[:,i-1:i],a,b)
    
        #Sample beta
        D=c1*omega_sample.sqrt()
        XTD=X.T*D.T
        DY=D*(Y-c2/omega_sample)         
        GXTD=G*XTD
        
        if P>N:
            mu=torch.randn(P,1,device=device)*G  
            U=DY-XTD.T@mu-torch.randn(N,1,device=device)    
            v=torch.linalg.solve(torch.eye(N,device=device)+GXTD.T@GXTD,U)
            beta_sample[:,i:i+1]=mu+G*GXTD@v 
        else:
            beta_sample[:,i:i+1]=torch.linalg.solve(GXTD@GXTD.T+torch.eye(P,device=device),GXTD@DY+GXTD@torch.randn(N,1,device=device)+torch.randn(P,1,device=device))*G
        
        #Sample omega
        g3=inverse_gaussian(2/torch.maximum(torch.abs((Y-X@beta_sample[:,i:i+1]).ravel()),torch.tensor(1e-2,device=device)))
        omega_sample=(g3.sample()/(2*Q*(1-Q))).view(-1,1)

    return beta_sample[:,burn_in:]





