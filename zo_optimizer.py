import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from numpy import random

from typing import Iterator, Callable

class ZOOptimizer():
    def __init__(self, model: nn.Module, compute_loss: Callable, q: int=1, mu: float=1e-2) -> None:
        """Initialize a forward difference zeroth-order optimizer. This class only populates the gradients of the parameters, it does not perform an optimization step.
        This optimizer can be used with any optimizer that takes in parameters and gradients, such as SGD or Adam.

        Args:
            model: The model to optimize.
            compute_loss: A function that computes the loss. It should use the same model as the optimizer
            q: The number of perturbations to average over.
            mu: The step size for the perturbations.
            sides: The number of sides to perturb the parameters. 1 for one-sided, 2 for two-sided.
        """
        assert q >= 1, "q must be greater than or equal to 1"
        # assert sides == 1 or sides == 2, "sides must be 1 or 2" # 1 for one-sided, 2 for two-sided perturbation
        self.model = model
        self.dim = parameters_to_vector(model.parameters()).numel()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.compute_loss = compute_loss # take in model and inputs, return loss
        self.mu = mu
        self.q = q

        self.rng = random.default_rng()


    @torch.no_grad()
    def loss_and_grad(self, *loss_args, **loss_kwargs) -> None:
        """Fill the gradients of the model parameters using zeroth-order optimization.

        Args:
            loss_args: Positional arguments to pass to the compute_loss function.
            loss_kwargs: Keyword arguments to pass to the compute_loss function.
        Returns:
            loss: The computed loss value.
        """        
        loss1 = self.compute_loss(*loss_args, **loss_kwargs) # unperturbed loss
        grad = torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        for _ in range(self.q):
            z = torch.normal(mean=0, std=1, size=(self.dim,), device=self.device, dtype=self.dtype)
            self._perturb_params(z, self.mu)
            loss2 = self.compute_loss(*loss_args, **loss_kwargs)
            self._perturb_params(z, -self.mu)
            grad += (loss2 - loss1) * z
        grad /= self.q * self.mu
        self._vector_to_grad(grad)

        return loss1


    @torch.no_grad()
    def _perturb_params(self, z, scale, mask=None) -> None:
        """Perturb the parameters of the model.

        Args:
            scale: The scale of the perturbation.
            mask: A mask to apply to the parameters. If None, all parameters are perturbed.
        """
        flat_params = parameters_to_vector(self.model.parameters())
        if mask is not None:
            z *= mask

        flat_params += scale * z
        vector_to_parameters(flat_params, self.model.parameters())


    @torch.no_grad()
    def _vector_to_grad(self, flat_grad: torch.Tensor) -> None:
        """Fill .grad attribute of the model with the flat gradient vector.

        Args:
            flat_grad: The flat gradient vector.
        """

        # Pointer for slicing the vector for each parameter
        pointer = 0
        for param in self.model.parameters():
            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            param.grad = flat_grad[pointer : pointer + num_param].view_as(param).data

            # Increment the pointer
            pointer += num_param


    @torch.no_grad()
    def _zero_grad(self) -> None:
        """Zero the gradients of the model parameters."""
        for param in self.model.parameters():
            if param.requires_grad:
                param.grad = None