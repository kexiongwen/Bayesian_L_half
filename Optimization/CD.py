import torch
import numpy as np
import math
    
def SR(Y,X,C=1.5,s=1,path=False):
    
    if path==True:
        
        beta_path=[]
        
    X=X.to(torch.float32)
    Y=Y.to(torch.float32)
    
    device=X.get_device()
    
    if device==-1:
        
        device='cpu'
    
    
    XTX=X.T@X
    N,P=X.size()
    b=C*math.log(P)/P
    beta=torch.zeros_like(X.T[:,0:1])
    if path==True:
        beta_path.append(torch.clone(beta).cpu().numpy())       
    beta_previous=torch.ones_like(X.T[:,0:1])
    Z=torch.ones_like(X.T[:,0:1]).ravel()
    a=0.5
    C1=P+a/(2**s)
    iteration=1
    power=1/(2-0.5**s)

    ink1=X[:,1:]@beta[1:,:]
    ink2=torch.zeros_like(Y)
    
    for t in range(10):
        
        while (torch.linalg.norm(beta-beta_previous)>1e-4 and iteration<20):

            beta_previous=torch.clone(beta)
            iteration=iteration+1

            for j in range(0,P):

                if (j!=0) & (j!=P-1):
                    Z[j]=X[:,j:j+1].T@(Y-ink2-ink1)/XTX[j,j]
                elif j==0:
                    Z[0]=X[:,0:1].T@(Y-ink1)/XTX[0,0]
                else:
                    Z[P-1]=X[:,P-1:P].T@(Y-ink2)/XTX[P-1,P-1]
                
                C2=1/b+torch.sum(torch.abs(beta)**(0.5**s))-torch.abs(beta[j,:])**(0.5**s)
                
                if torch.abs(Z[j])<2*((C1/(C2+torch.abs(Z[j])**(0.5**s)))*0.5/XTX[j,j])**power:
                    beta[j,:]=0
                else:

                    beta_old=torch.abs(Z[j])
                    beta_new=torch.tensor(1000,device=device)
                    k=1
                    
                    while (torch.abs(beta_old-beta_new)>1e-4 and k<=20 and beta_new>=0):

                        beta_old=torch.clone(beta_new)
                        beta_new=torch.abs(Z[j])-C1/XTX[j,j]/(beta_old+C2*beta_old**(1-0.5**s))
                        k=k+1

                    if k>20 or beta_new<0:
                        beta[j,:]=0 
                    else:
                        beta[j,:]=beta_new*torch.sign(Z[j])
                
                if (j!=0) & (j!=P-1):
                    ink1=ink1-X[:,j+1:j+2]@beta[j+1:j+2,:]
                    ink2=ink2+X[:,j:j+1]@beta[j:j+1,:]
                elif j==0:
                    ink1=ink1-X[:,1:2]@beta[1:2,:]
                    ink2=X[:,0:1]@beta[0:1,:]
                else:
                    ink1=ink2+X[:,j:j+1]@beta[j:j+1,:]-X[:,0:1]@beta[0:1,:]
                    ink2=torch.zeros_like(Y)
            
            if path==True:
            
                beta_path.append(torch.clone(beta).cpu().numpy())
        
        if t==1:
            C1=P+a/(2**s)
        else: 
            rs=Y-X@beta
            sigma2=rs.T@rs/(N+2)
            C1=sigma2*(P+a/(2**s))

    sparsity=torch.count_nonzero(beta!=0)
    sigma2_estimator=(Y-X@beta).T@(Y-X@beta)/(N-sparsity)
    sigma2_estimator=sigma2
    
    if path==True:
        
        return beta,sigma2_estimator,np.asarray(beta_path).reshape(-1,P)
    
    else:

        return beta,sigma2_estimator
    

def SQR(Y,X,Q=0.5,C=0.5,s=1,path=False):
    
    if path==True:
        
        beta_path=[]
    
    X=X.to(torch.float32)
    Y=Y.to(torch.float32)
    
    device=X.get_device()
    
    if device==-1:
        
        device='cpu'

    iteration=10
    N,P=X.size()
    b=C*math.log(P)/P
    a=0.5
    C1=(P+a/(2**s))
    power=1/(2-0.5**s)
    beta=torch.zeros_like(X.T[:,0:1])
    
    if path==True:
        beta_path.append(torch.clone(beta).cpu().numpy())   
        
    Z=torch.ones_like(X.T[:,0:1]).ravel()

    for t in range(0,iteration):

        L1_error=torch.maximum(torch.abs(Y-X@beta),torch.tensor(0.1,device=device))
        T=Y-(1-2*Q)*L1_error

        W=0.25/L1_error    

        XTW=X.T*W.T
        XTWX=(XTW*X.T).sum(-1)

        iter=1

        ink1=X[:,1:]@beta[1:,:]
        ink2=torch.zeros_like(Y)
        beta_previous=torch.ones_like(X.T[:,0:1])
        
        while (torch.linalg.norm(beta-beta_previous)>1e-4 and iter<20):

            beta_previous=torch.clone(beta)
            iter=iter+1
            
            for j in range(0,P):

                C2=1/b+torch.sum(torch.abs(beta)**(0.5**s))-torch.abs(beta[j])**(0.5**s)

                if (j!=0) & (j!=P-1):
                    Z[j]=XTW[j:j+1,:]@(T-ink1-ink2)/XTWX[j]
                elif j==0:
                    Z[0]=XTW[0:1,:]@(T-ink1)/XTWX[j]
                else:
                    Z[P-1]=XTW[P-1:P,:]@(T-ink2)/XTWX[j]

                if torch.abs(Z[j])<=2*(C1/(2*C2+2*torch.abs(Z[j])**0.5)/XTWX[j])**power:

                    beta[j]=0

                else:

                    beta_old=torch.abs(Z[j])
                    beta_new=torch.tensor(1000,device=device)
                    k=1

                    while(torch.abs(beta_old-beta_new)>1e-4 and k<20 and beta_new>=0):
                        
                        beta_old=torch.clone(beta_new)
                        beta_new=torch.abs(Z[j])-C1/XTWX[j]/(beta_old+C2*beta_old**(1-0.5**s))
                        k=k+1
                        
                    if k>=20 or beta_new<0:
                        beta[j]=0 
                    else:
                        beta[j]=beta_new*torch.sign(Z[j])

                if (j!=0) & (j!=P-1):
                    ink1=ink1-X[:,j+1:j+2]@beta[j+1:j+2,:]
                    ink2=ink2+X[:,j:j+1]@beta[j:j+1,:]
                elif j==0:
                    ink1=ink1-X[:,1:2]@beta[1:2,:]
                    ink2=X[:,0:1]@beta[0:1,:]
                else:
                    ink1=ink2+X[:,j:j+1]@beta[j:j+1,:]-X[:,0:1]@beta[0:1,:]
                    ink2=torch.zeros_like(Y)
            
            if path==True:
                beta_path.append(torch.clone(beta).cpu().numpy())
    
    if path==True:
        
        return beta,np.asarray(beta_path).reshape(-1,P)
    else:
        
        return beta


def SLR(Y,X,C=0.5,s=3,path=False):
    
    if path==True:
        
        beta_path=[]
    
    X=X.to(torch.float32)
    Y=Y.to(torch.float32)
    
    device=X.get_device()
    
    if device==-1:
        
        device='cpu'

    iteration=20
    N,P=X.size()
    b=C*math.log(P)/P
    a=0.5
    C1=(P+a/(2**s))
    power=1/(2-0.5**s)
    beta=torch.zeros_like(X.T[:,0:1])
    if path==True:
        beta_path.append(torch.clone(beta).cpu().numpy())   
    Z=torch.ones_like(X.T[:,0:1]).ravel()

    for t in range(0,iteration):

        Pro=torch.exp(X@beta)/(1+torch.exp(X@beta))

        W=(Pro/(1+torch.exp(X@beta)))

        T=X@beta+(Y-Pro)/W

        XTW=X.T*W.T
        XTWX=(XTW*X.T).sum(-1)

        iter=1

        ink1=X[:,1:]@beta[1:,:]
        ink2=torch.zeros_like(Y)
        beta_previous=torch.ones_like(X.T[:,0:1])
        
        while (torch.linalg.norm(beta-beta_previous)>1e-4 and iter<20):

            beta_previous=torch.clone(beta)
            iter=iter+1
            
            for j in range(0,P):

                C2=1/b+torch.sum(torch.abs(beta)**(0.5**s))-torch.abs(beta[j])**(0.5**s)

                if (j!=0) & (j!=P-1):
                    Z[j]=XTW[j:j+1,:]@(T-ink1-ink2)/XTWX[j]
                elif j==0:
                    Z[0]=XTW[0:1,:]@(T-ink1)/XTWX[j]
                else:
                    Z[P-1]=XTW[P-1:P,:]@(T-ink2)/XTWX[j]

                if torch.abs(Z[j])<=2*(C1/(2*C2+2*torch.abs(Z[j])**0.5)/XTWX[j])**power:

                    beta[j]=0

                else:

                    beta_old=torch.abs(Z[j])
                    beta_new=torch.tensor(1000,device=device)
                    k=1

                    while(torch.abs(beta_old-beta_new)>1e-4 and k<20 and beta_new>=0):
                        
                        beta_old=torch.clone(beta_new)
                        beta_new=torch.abs(Z[j])-C1/XTWX[j]/(beta_old+C2*beta_old**(1-0.5**s))
                        k=k+1
                        
                    if k>=20 or beta_new<0:
                        beta[j]=0 
                    else:
                        beta[j]=beta_new*torch.sign(Z[j])

                if (j!=0) & (j!=P-1):
                    ink1=ink1-X[:,j+1:j+2]@beta[j+1:j+2,:]
                    ink2=ink2+X[:,j:j+1]@beta[j:j+1,:]
                elif j==0:
                    ink1=ink1-X[:,1:2]@beta[1:2,:]
                    ink2=X[:,0:1]@beta[0:1,:]
                else:
                    ink1=ink2+X[:,j:j+1]@beta[j:j+1,:]-X[:,0:1]@beta[0:1,:]
                    ink2=torch.zeros_like(Y)
            
            if path==True:
    
                beta_path.append(torch.clone(beta).cpu().numpy())
            
    if path==True:
        
        return beta.ravel(),np.asarray(beta_path).reshape(-1,P)
    
    else:        
            
        return beta.ravel()


