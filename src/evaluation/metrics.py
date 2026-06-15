import torch

def compute_multi_instance_recall(sim_matrix, verb_classes, noun_classes, log_prefix="", k_vals=[1, 5, 10]):
    recalls = {}
    
    verb_mask = (verb_classes.unsqueeze(1) == verb_classes.unsqueeze(0))
    noun_mask = (noun_classes.unsqueeze(1) == noun_classes.unsqueeze(0))
    
    semantic_match_mask = (verb_mask & noun_mask).to(sim_matrix.device)
    
    for k in k_vals:
        top_k_indices = torch.topk(sim_matrix, k, dim=1).indices
        hits = torch.gather(semantic_match_mask, 1, top_k_indices) 
        correct = hits.any(dim=1)
        
        recalls[f"{log_prefix}R@{k}"] = correct.float().mean().item() * 100

    return recalls