import torch


class DiceLoss(torch.nn.Module):
    """Computes the dice loss."""
    
    def __init__(self, eps: float = 1e-9):
        """Initializes a new DiceLoss instance.

        Args:
            eps (float, optional):Constant for numerical stability. Defaults to 1e-9.
        """
        
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self, probability_map: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """Computes the dice loss between the predicted segmentation and the ground truth.

        Args:
            probability_map (torch.Tensor): The predicted segmentation of the model of shape (batch_size, height, width, depth).
            target_mask (torch.Tensor): The ground truth of shape (batch_size, height, width, depth).
        Returns:
            torch.Tensor: A scalar tensor containing the computed loss.
        """
        
        assert(probability_map.shape == target_mask.shape), f"Input tensors must have the same shape, got {probability_map.shape} and {target_mask.shape}"
        
        batch_size = target_mask.size(0)
        probability_vectorized = probability_map.view(batch_size, -1)
        targets_vectorized = target_mask.view(batch_size, -1)
       
        
        intersection = 2.0 * (probability_vectorized * targets_vectorized).sum()
        union = probability_vectorized.sum() + targets_vectorized.sum()
        dice_score = (intersection + self.eps) / union
    
        return 1.0 - dice_score
