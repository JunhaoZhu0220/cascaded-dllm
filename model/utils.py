from numpy import outer
import torch
import torch.nn.functional as F


def get_model_fn(model, train=False, return_mask=False):
    # TBD: write a comment for the new arg return_mask
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability
        return_mask: 

    Returns:
        A model function.
    """

    def model_fn(x, sigma):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()

        if return_mask:
            return model(x, sigma, return_mask_logits=True)
        else:
            # otherwise output the raw values (we handle mlm training in losses.py)
            return model(x, sigma)

    return model_fn


def get_score_fn(model, train=False, sampling=False, return_mask=False):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train, return_mask=return_mask)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def score_fn(x, sigma):
            sigma = sigma.reshape(-1)
            out = model_fn(x, sigma)
            if return_mask:
                log_score, mask_logits = out
            else:
                log_score = out
            
            if sampling:
                # when sampling return true score (not log used for training)
                log_score = log_score.exp()
                
            return (log_score, mask_logits) if return_mask else log_score

    return score_fn
