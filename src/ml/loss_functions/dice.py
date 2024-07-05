import torch

from src.ml.metrics.dice import DiceCoefficient

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
        
        dice_score = DiceCoefficient.compute_dice(probability_map, target_mask)

        dice_score = torch.tensor(dice_score, requires_grad=True)
    
        return 1.0 - dice_score
