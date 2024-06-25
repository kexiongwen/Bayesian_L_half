import torch
from torch.distributions.gamma import Gamma
from MCMC.invgauss import inverse_gaussian

def shrinkage1(beta_sample,a,b):
    
    P,_=beta_sample.size()
    
    #Sample lam
    ink=beta_sample.abs().sqrt()
    g1=Gamma(2*P+a,ink.sum()+b)
    lam=g1.sample()
    ink=lam*ink
    
    #Sample V
    g2=inverse_gaussian(1/ink)
    v=2/g2.sample()
    
    #Sample tau
    g3=inverse_gaussian(v/ink.pow(2))
    tau=v/g3.sample().sqrt()
    
    if torch.any(torch.isinf(tau)):

        tau[torch.isinf(tau)] = 300
    
    if torch.any(torch.isnan(tau)):
        
        tau[torch.isnan(tau)] = 300
    
    D=tau/lam**2
    
    return D

def shrinkage2(beta_sample,a,b):
    
    P,_=beta_sample.size()
    
    #Sample lam
    ink=beta_sample.abs()**0.25
    g1=Gamma(4*P+a,ink.sum()+b)
    lam=g1.sample()
    
    ink1=(lam*ink).pow(4)
    ink2=ink1.sqrt()/lam
    
    # Sampling v2
    g2=inverse_gaussian(1/(lam*ink))
    v2_sample=2/g2.sample()
    
    # Sampling v1
    g3=inverse_gaussian(v2_sample/ink2)
    v1_sample=2*v2_sample.pow(2)/g3.sample()
    
    # Sampling tau
    g4=inverse_gaussian(v1_sample/ink1)
    tau_sample=v1_sample/g4.sample().sqrt()     
        
    if torch.any(torch.isinf(tau_sample)):
        tau_sample[torch.isinf(tau_sample)] = 2000
        
    if torch.any(torch.isnan(tau_sample)):
        tau_sample[torch.isnan(tau_sample)] = 2000
        
    D=tau_sample/lam**4
    
    return D

