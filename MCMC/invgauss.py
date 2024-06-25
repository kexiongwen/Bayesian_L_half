import torch

class inverse_gaussian:
        
    def __init__(self,mu):
        self.mu=mu
        self.device=mu.get_device()
        
        if self.device==-1:
            
            self.device='cpu'
        
    def sample(self):
        
        ink=self.mu*torch.randn_like(self.mu,device=self.device).pow(2)
        a=1+0.5*(ink-((ink+2).pow(2)-4).sqrt())
        indicator=(1/(1+a))>=torch.rand_like(self.mu,device=self.device) 
        draws=torch.ones_like(self.mu)
        draws[indicator]=self.mu[indicator]*a[indicator]
        draws[~indicator]=self.mu[~indicator]/a[~indicator]
        
        return draws
    
