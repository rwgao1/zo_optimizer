import torch
import torch.nn
from torch.nn import Parameter

from numpy import random

from typing import Iterator, Callable

class ZOOptimizer():
    def __init__(self, optimizer, params: Iterator[Parameter], compute_loss: Callable, q: int=1, mu: float=1e-2, sides: int=1) -> None:
        """Initialize a forward difference zeroth-order optimizer.

        Args:
            optimizer: The optimizer to use for updating the parameters.
            params: An iterator of torch.nn.Parameter objects to optimize.
            compute_loss: A function that takes in a model, inputs, and target, and returns the loss.
            q: The number of perturbations to average over.
            mu: The step size for the perturbations.
            sides: The number of sides to perturb the parameters. 1 for one-sided, 2 for two-sided.
        """
        assert q >= 1, "q must be greater than or equal to 1"
        assert sides == 1 or sides == 2, "sides must be 1 or 2" # 1 for one-sided, 2 for two-sided perturbation
        
        self.params_to_opt = []
        for param in params:
            if param.requires_grad:
                self.params_to_opt.append(param)
                param.grad = None # ensure that grad is empty

        self.optimizer = optimizer(self.params_to_opt)
        self.compute_loss = compute_loss # take in model and inputs, return loss
        self.mu = mu
        self.q = q
        self.sides = sides

        self.rng = random.default_rng()


    @torch.no_grad()
    def step(self, model, inputs, target) -> None:
        """Perform a step of the zeroth-order optimizer.

        Args:
            model: The model to evaluate the loss on.
            inputs: The input data to the model.
            target: The target data to the model.
        """        
        loss1 = self.compute_loss(model, inputs, target) # unperturbed loss

        for _ in range(self.q):
            self.seed = self.rng.integers(0, 100) # sample new seed
            if self.sides == 1: # one-sided perturbation
                self._perturb_params(scale=1)
                loss2 = self.compute_loss(model, inputs, target)
                self.grad_scale = ((loss2 - loss1) / self.mu / self.q).item()
                self._perturb_params(scale=-1) # reset to original parameters

            # TODO: implement two-sided perturbation

            torch.manual_seed(self.seed) # reset the seed to match the perturbations

            for param in self.params_to_opt: # update the parameters
                z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
                if param.grad is None:
                    param.grad = z * self.grad_scale
                else:
                    param.grad += z * self.grad_scale
    
        self.optimizer.step()
        self.optimizer.zero_grad()

        
    @torch.no_grad()
    def _perturb_params(self, scale) -> None:
        """Perturb the parameters of the model.

        Args:
            scale: The scale of the perturbation.
        """
        torch.manual_seed(self.seed) # set the seed so we don't have to store perturbation in memory

        for param in self.params_to_opt:

            z = torch.normal(mean=0, std=1, size=param.size(), device=param.device, dtype=param.dtype)
            param.data = param.data + scale * self.mu * z
            param.z = z