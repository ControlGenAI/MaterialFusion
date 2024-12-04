import torch


def proximal_operator(score_delta, quantile, prox='l0'):
    score_delta = score_delta.clone()
    if quantile > 0:            
        threshold = score_delta.abs().quantile(quantile)
    else:
        threshold = -quantile  # if quantile is negative, use it as a fixed threshold
    
    if prox == 'l1':
        score_delta -= score_delta.clamp(-threshold, threshold)
        score_delta = torch.where(score_delta > 0, score_delta - threshold, score_delta)
        score_delta = torch.where(score_delta < 0, score_delta + threshold, score_delta)
    elif prox == 'l0':
        # old 
        # score_delta -= score_delta.clamp(-threshold, threshold)
        
        # new
        score_delta = torch.where(score_delta.abs() > torch.sqrt(2 * threshold), score_delta, score_delta * 0.)
    else:
        raise ValueError('configure prox argument')
    
    return score_delta
