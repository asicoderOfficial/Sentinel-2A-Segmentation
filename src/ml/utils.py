import torch


def model_n_parameters(model: torch.nn.Module) -> int:
    """ Returns the number of parameters of a model.

    Args:
        model (torch.nn.Module): Model.

    Returns:
        int: Number of parameters of the model.
    """    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
