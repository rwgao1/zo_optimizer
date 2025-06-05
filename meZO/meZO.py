import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.checkpoint import checkpoint

from functools import partial

from zo_optimizer import *

from typing import Iterator, Callable

# def _model_output_hook(module, input, output: torch.Tensor, k):
#     """A hook to add the ranking hook to the model output"""
#     # output.requires_grad = True  # Ensure the output has gradients
#     output.register_hook(partial(_top_k_grad_hook, k=k))

# def _top_k_grad_hook(grad: torch.Tensor, k) -> torch.Tensor:
#     """A hook to mask all but top-k values of dL/dy"""

#     # Get the top-k indices
#     top_k_indices = torch.topk(grad, k, dim=0).indices
#     mask = torch.zeros_like(grad, dtype=torch.bool)
#     mask[top_k_indices] = True
    
#     # Apply the mask to the gradient
#     grad[~mask] = 0
#     return grad

class meZOOptimizer(ZOOptimizer):
    def __init__(self, model: nn.Module, compute_loss: Callable, q: int=1, mu: float=1e-2, k=None) -> None:
        """Initialize a forward difference zeroth-order optimizer. This class only populates the gradients of the parameters, it does not perform an optimization step.
        This optimizer can be used with any optimizer that takes in parameters and gradients, such as SGD or Adam.

        Args:
            model: The model to optimize.
            compute_loss: A function that computes the loss. It should use the same model as the optimizer
            q: The number of perturbations to average over.
            mu: The step size for the perturbations.
        """
        super().__init__(model, compute_loss, q, mu)
        # self.k = k

        # self.model.register_forward_hook(partial(_model_output_hook, k=k))


    @torch.no_grad()
    def _get_grad_mask(self) -> torch.Tensor:
        vec = []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            # Ensure the parameters are located in the same device
            vec.append(param.grad.view(-1))

        vec = torch.cat(vec)
        vec = (vec / vec).type(torch.int) # Create a mask of 1s where gradients are present
        # print(vec)
        # print number of ones in mask
        # print(f"Number of unmasked: {(vec != 0).sum().item()}")
        return vec
    
    
    def loss_and_grad(self, *loss_args, **loss_kwargs):
        """Fill the gradients of the model parameters using zeroth-order optimization.

        Args:
            loss_args: Positional arguments to pass to the compute_loss function.
            loss_kwargs: Keyword arguments to pass to the compute_loss function.
        Returns:
            loss: The computed loss value.
        """        
        # loss1 = checkpoint(self.compute_loss, *loss_args, **loss_kwargs, use_reentrant=False)
        loss1 = self.compute_loss(*loss_args, **loss_kwargs)
        loss1.backward()
        grad = torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        mask = self._get_grad_mask()
        self._zero_grad()

        with torch.no_grad():

            for _ in range(self.q):
                z = torch.normal(mean=0, std=1, size=(self.dim,), device=self.device, dtype=self.dtype)
                self._perturb_params(z, self.mu, mask)
                loss2 = self.compute_loss(*loss_args, **loss_kwargs)
                self._perturb_params(z, -self.mu, mask)
                grad += (loss2 - loss1) * z
            grad /= self.q * self.mu
            #TODO: potentially apply mask to grad
            self._vector_to_grad(grad * mask)
            return loss1
        
    
