import torch
import torch.nn.functional as F

def weighted_adj_from_pyg(edge_index, edge_weight, num_nodes, directed=True):
    """
    Build dense weighted adjacency A (N x N) from PyG edge_index and scalar edge_weight.
    If edge_weight is None, uses 1.0 for all edges.
    """
    device = edge_index.device
    A = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float32)

    if edge_weight is None:
        w = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
    else:
        w = edge_weight.view(-1).to(dtype=torch.float32)

    src, dst = edge_index[0], edge_index[1]
    A[src, dst] = A[src, dst] + w

    if not directed:
        A[dst, src] = A[dst, src] + w

    return A

def sinkhorn(log_alpha, n_iters=30, eps=1e-9):
    """
    Sinkhorn normalization in log-space for numerical stability.
    Produces an approximately doubly-stochastic matrix.
    log_alpha: (n1 x n2)
    """
    for _ in range(n_iters):
        # normalize rows
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        # normalize cols
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return torch.exp(log_alpha).clamp_min(eps)

def node_cost_matrix(x1, x2, mode="l2"):
    """
    x1: (n1 x d), x2: (n2 x d)
    returns C: (n1 x n2)
    """
    if x1 is None or x2 is None:
        raise ValueError("Need node features data.x for node_cost_matrix; or implement your own cost.")
    if mode == "l2":
        # ||x_i - y_j||_2
        return torch.cdist(x1.float(), x2.float(), p=2)
    elif mode == "cosine":
        x1n = F.normalize(x1.float(), dim=-1)
        x2n = F.normalize(x2.float(), dim=-1)
        return 1.0 - (x1n @ x2n.t())
    else:
        raise ValueError(f"Unknown mode: {mode}")

def soft_ged_distance(
    data1,
    data2,
    *,
    directed=True,
    node_cost="l2",
    tau=0.1,
    sinkhorn_iters=40,
    lambda_node=1.0,
    lambda_edge=1.0,
    # optional “mass mismatch” (insertion/deletion) penalty
    lambda_size=0.0,
):
    """
    Differentiable GED-like distance between two PyG Data graphs with weighted edges.

    Requirements:
      - data.x exists on both graphs (for node substitution cost)
      - edge weights in data.edge_attr as scalar per edge (or None -> weight=1)

    Returns:
      dist (scalar tensor), P (n1 x n2 soft assignment)
    """
    device = data1.edge_index.device
    n1 = int(data1.num_nodes)
    n2 = int(data2.num_nodes)

    # Node substitution cost
    C = node_cost_matrix(data1.x, data2.x, mode=node_cost)  # (n1 x n2)

    # Soft assignment: P ~ argmin <P, C>
    # convert costs to logits (lower cost -> higher logit)
    log_alpha = (-C / max(tau, 1e-8)).to(device)
    P = sinkhorn(log_alpha, n_iters=sinkhorn_iters)  # (n1 x n2)

    node_term = (P * C).sum()

    # Edge term via weighted adjacency
    w1 = None if getattr(data1, "edge_attr", None) is None else data1.edge_attr[:, 0]
    w2 = None if getattr(data2, "edge_attr", None) is None else data2.edge_attr[:, 0]

    A1 = weighted_adj_from_pyg(data1.edge_index, w1, n1, directed=directed)
    A2 = weighted_adj_from_pyg(data2.edge_index, w2, n2, directed=directed)

    # Transport A2 into G1's node space
    A2_mapped = P @ A2 @ P.t()   # (n1 x n1)

    # L1 difference captures deletions/insertions/substitutions of weighted edges
    edge_term = torch.abs(A1 - A2_mapped).sum()

    # Optional size penalty (rough insertion/deletion proxy)
    size_term = torch.tensor(0.0, device=device)
    if lambda_size != 0.0:
        size_term = lambda_size * torch.abs(torch.tensor(float(n1 - n2), device=device))

    dist = lambda_node * node_term + lambda_edge * edge_term + size_term
    return dist, P


dist, P = soft_ged_distance(g1, g2, directed=True, tau=0.2, lambda_node=1.0, lambda_edge=0.5)
loss = dist
loss.backward()
