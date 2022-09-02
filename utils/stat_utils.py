import torch
from torch.distributions import Normal, Independent


def merge_normal_dist(dist_1, dist_2, n1=0.5, n2=0.5):
    mean_1, mean_2 = dist_1.mean, dist_2.mean
    var_1, var_2 = dist_1.variance, dist_2.variance

    new_mean = (n1 * mean_1 + n2 * mean_2) / (n1 + n2)
    new_variance = (n1**2) / ((n1 + n2) ** 2) * var_1 + (n2**2) / (
        (n1 + n2) ** 2
    ) * var_2

    return Independent(Normal(new_mean, torch.sqrt(new_variance)), 1)
