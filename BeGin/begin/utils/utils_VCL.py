import copy
import torch
from torch_geometric.utils.mask import mask_to_index, index_to_mask
from dgl.data.utils import split_dataset

def coreset_rand(data, coreset_size, task="task", domain="node"):
    if task in ["task", "class"]:
        if domain == "node":
            mask_size = torch.sum(data.ndata['train_mask'])
            total_size = data.ndata['train_mask'].shape[0]
            assert coreset_size < mask_size
    
            coreset = copy.deepcopy(data)
            
            mask = mask_to_index(coreset.ndata['train_mask'])
            mask = mask[torch.randperm(mask.shape[0])]
            
            coreset_mask, data_mask = torch.split(mask, [coreset_size, mask_size-coreset_size])
            coreset.ndata['train_mask'] = index_to_mask(coreset_mask, total_size)
            data.ndata['train_mask'] = index_to_mask(data_mask, total_size)
            
            # print(f"Created coreset with {coreset_size/mask_size} of training data.")
            
        elif domain == "graph":
            train_size = len(data['train'])
            assert train_size > coreset_size
            
            train_data, coreset_data, _ = split_dataset(
                data['train'],
                (
                    (train_size-coreset_size) / train_size,
                    coreset_size / train_size,
                    0
                ),
                shuffle=True,
            )
            
            data['train'] = train_data
            coreset = coreset_data
            
        else:
            raise NotImplementedError()
        
    else:
        raise NotImplementedError()
    
    return coreset, data

# def KMeans(x, K=10, Niter=10, verbose=True):
#     """Implements Lloyd's algorithm for the Euclidean metric."""

#     start = time.time()
#     N, D = x.shape  # Number of samples, dimension of the ambient space

#     c = x[:K, :].clone()  # Simplistic initialization for the centroids

#     x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
#     c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

#     # K-means loop:
#     # - x  is the (N, D) point cloud,
#     # - cl is the (N,) vector of class labels
#     # - c  is the (K, D) cloud of cluster centroids
#     for i in range(Niter):
#         # E step: assign points to the closest cluster -------------------------
#         D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
#         cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

#         # M step: update the centroids to the normalized cluster average: ------
#         # Compute the sum of points per cluster:
#         c.zero_()
#         c.scatter_add_(0, cl[:, None].repeat(1, D), x)

#         # Divide by the number of points per cluster:
#         Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
#         c /= Ncl  # in-place division to compute the average

#     if verbose:  # Fancy display -----------------------------------------------
#         if use_cuda:
#             torch.cuda.synchronize()
#         end = time.time()
#         print(
#             f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
#         )
#         print(
#             "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
#                 Niter, end - start, Niter, (end - start) / Niter
#             )
#         )

#     return cl, c

# def coreset_cluster(data, coreset_size, task="task", domain="node"):
#     mask_size = torch.sum(data.ndata['train_mask'])
#     total_size = data.ndata['train_mask'].shape[0]
#     assert coreset_size < mask_size
    
#     if task == "task":
#         if domain == "node":
#             coreset = copy.deepcopy(data)
            
#             mask = mask_to_index(coreset.ndata['train_mask'])
#             mask = mask[torch.randperm(mask.shape[0])]
            
#             coreset_mask, data_mask = torch.split(mask, [coreset_size, mask_size-coreset_size])
#             coreset.ndata['train_mask'] = index_to_mask(coreset_mask, total_size)
#             data.ndata['train_mask'] = index_to_mask(data_mask, total_size)
            
#             print(f"Created coreset with {coreset_size/mask_size} of training data.")
            
#         else:
#             raise NotImplementedError()
        
#     else:
#         raise NotImplementedError()
    
#     return coreset, data