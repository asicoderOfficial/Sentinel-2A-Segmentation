import torch

from src.ml.utils import predictions_to_binary


class DiceCoefficient:
    @staticmethod
    def compute_dice(predicted: torch.Tensor, target: torch.Tensor) -> float:
        """ Compute the Dice coefficient overall.
        The Dice coefficient is a metric used to measure the similarity or overlap between two sets, very similar to the Jaccard coefficient.

        Dice = (2 * |X & Y|) / (|X| + |Y|)

        Args:
            predicted (torch.Tensor): The predicted segmentation by the model.
            target (torch.Tensor): The ground truth segmentation.

        Returns:
            float: The Dice coefficient.
        """        
        predicted_labels = predictions_to_binary(predicted)

        intersection = torch.eq(predicted_labels, target).sum().float()
        union = predicted_labels.numel()

        dice = (intersection / union).item() if union != 0 else 0.0

        # For floating point errors
        if dice > 1.0:
            dice = 1.0
        if dice < 0.0:
            dice = 0.0

        return dice
        