import torch



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
        # Apply sigmoid to the predicted values
        predicted = torch.sigmoid(predicted)

        # If a pixel has a probability greater than 0.5, it is considered as a building (label 1), if not, it is considered as background (label 0)
        predicted_labels = predicted > 0.5

        intersection = torch.sum(predicted_labels * target).float()
        union = torch.sum(predicted_labels).float() + torch.sum(target).float()

        dice = (2.0 * intersection.item()) / (union.item())

        # For floating point errors
        if dice > 1.0:
            dice = 1.0
        if dice < 0.0:
            dice = 0.0

        return dice
    