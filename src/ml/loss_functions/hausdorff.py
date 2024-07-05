import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

class HausdorffLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, weight_fn: float = 2.0):
        """
        Initializes a new HausdorffLoss instance.

        Args:
            alpha (float, optional): The power parameter for averaging the distances. Defaults to 2.0.
            weight_fn (float, optional): The weight for penalizing false negatives. Defaults to 2.0.
        """
        super(HausdorffLoss, self).__init__()
        self.alpha = alpha
        self.weight_fn = weight_fn

    def forward(self, probability_map: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hausdorff loss between the predicted segmentation and the ground truth.

        Args:
            probability_map (torch.Tensor): The predicted segmentation of the model of shape (batch_size, height, width).
            target_mask (torch.Tensor): The ground truth of shape (batch_size, height, width).

        Returns:
            torch.Tensor: A scalar tensor containing the computed loss.
        """
        assert probability_map.shape == target_mask.shape, f"Input tensors must have the same shape, got {probability_map.shape} and {target_mask.shape}"

        # Apply sigmoid to the predicted values to get probabilities
        probability_map = torch.sigmoid(probability_map)

        # Thresholding the probability map to get binary predictions
        predicted_labels = (probability_map > 0.5).float()
        
        # Convert tensors to numpy arrays for distance transform
        batch_size = target_mask.size(0)
        distance_map_pred = torch.zeros_like(predicted_labels)
        distance_map_gt = torch.zeros_like(target_mask)

        for i in range(batch_size):
            pred_np = predicted_labels[i].cpu().numpy()
            gt_np = target_mask[i].cpu().numpy()

            distance_map_pred[i] = torch.from_numpy(distance_transform_edt(pred_np == 0)).to(predicted_labels.device)
            distance_map_gt[i] = torch.from_numpy(distance_transform_edt(gt_np == 0)).to(target_mask.device)

        # Compute the average Hausdorff distance
        dist1 = (distance_map_pred * target_mask).pow(self.alpha).mean()
        dist2 = (distance_map_gt * predicted_labels).pow(self.alpha).mean()

        # Apply weighting to penalize false negatives more
        weighted_dist1 = dist1 * self.weight_fn

        hausdorff_distance = weighted_dist1 + dist2

        hausdorff_distance = torch.tensor(hausdorff_distance, requires_grad=True)

        return hausdorff_distance
