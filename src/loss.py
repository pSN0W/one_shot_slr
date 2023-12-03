import torch
import einops


def TripletLossFn(v1, v2, device, margin=0.25):
    """Custom Loss function.

    Args:
        v1 (numpy.ndarray): Array with dimension (batch_size, model_dimension) associated to Q1.
        v2 (numpy.ndarray): Array with dimension (batch_size, model_dimension) associated to Q2.
        margin (float, optional): Desired margin. Defaults to 0.25.

    Returns:
        jax.interpreters.xla.DeviceArray: Triplet Loss.
    """

    # We compute pairwise cosine sum for each example
    scores = einops.einsum(v1, v2, "b_v1 num_dim, b_v2 num_dim -> b_v1 b_v2")

    # calculate new batch size
    batch_size = len(scores)

    # get all the diagonals to know the positive score
    positive = torch.diag(scores)

    # subtract a big number from positive score to remove them from negative
    negative_without_positive = scores - 2.0 * torch.eye(batch_size, device=device)

    # Take max for hard negative mining
    closest_negative = einops.reduce(negative_without_positive, "i j -> i", "max")

    # Mask the positive examples
    negative_zero_on_duplicate = scores * (1.0 - torch.eye(batch_size, device=device))
    # get mean for each of the pairs
    mean_negative = einops.reduce(negative_zero_on_duplicate, "i j -> i", "sum") / (
        batch_size - 1
    )

    # compute the triplet loss
    triplet_loss1 = torch.maximum(
        torch.tensor(0.0), margin - positive + closest_negative
    )
    triplet_loss2 = torch.maximum(torch.tensor(0., device=device), margin - positive + mean_negative)

    # add the two losses together and take the `fastnp.mean` of it
    triplet_loss = torch.mean(triplet_loss1 + triplet_loss2)

    ## Do you want to focus more on the current example?

    return triplet_loss
