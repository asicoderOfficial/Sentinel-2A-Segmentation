import torch


def model_n_parameters(model: torch.nn.Module) -> int:
    """ Returns the number of parameters of a model.

    Args:
        model (torch.nn.Module): Model.

    Returns:
        int: Number of parameters of the model.
    """    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predictions_to_binary(predicted: torch.Tensor, threshold:float=0.5) -> torch.Tensor:
    """ Convert the predicted values to binary values.

    Args:
        predicted (torch.Tensor): Predicted values.

    Returns:
        torch.Tensor: Binary values (0 or 1).
    """    
    # Apply sigmoid to the predicted values
    predicted = torch.sigmoid(predicted)

    # If a pixel has a probability greater than 0.5, it is considered as a building (label 1), if not, it is considered as background (label 0)
    predicted_labels = predicted > threshold
    predicted_labels = predicted_labels.int()

    return predicted_labels
