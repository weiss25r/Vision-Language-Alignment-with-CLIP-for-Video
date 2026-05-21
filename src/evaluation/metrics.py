import torch

def compute_recall(sim_matrix, len_text_embeddings, log_prefix="", k_vals=[1, 5, 10]):
    recalls = {}

    for k in k_vals:
        top_k_indices = torch.topk(sim_matrix, k, dim=1).indices

        ground_truth = torch.arange(len_text_embeddings).unsqueeze(1).to(top_k_indices.device)
        correct = (top_k_indices == ground_truth).any(dim=1)

        recalls[f"{log_prefix}R@{k}"] = correct.float().mean().item() * 100

    return recalls
