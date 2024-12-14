import torch 

def metric(prediction, Y):
    
    Pre_zero = (prediction == 0)
    Pre_nonzero = (prediction == 1)
    Y_zero = (Y == 0)
    Y_nonzero = (Y == 1)
    
    TP = (Y_nonzero * Pre_nonzero).sum()
    TN = (Y_zero * Pre_zero).sum()
    FP = (Pre_nonzero).sum() - TP
    FN = (Pre_zero).sum() - TN
    
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    J = Sensitivity + Specificity - 1
    Accuracy = (prediction == Y).sum() / torch.numel(Y)
    
    result = (Accuracy, Sensitivity, Specificity, J, MCC)
    
    return result

def post_process(X, beta, method):
    
    Z = X @ beta
    probability = torch.exp(Z) / (torch.exp(Z) + 1)
    
    if method == 'MCMC':
        
        predictive = torch.mean(probability, axis = 1) > 0.5
        
    elif method == 'Optimization':
        
        predictive = probability > 0.5
        
    return predictive
    
    
    
    

    
    
    
    

    