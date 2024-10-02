import torch
import torch.nn
from torch.nn import Parameter

from functools import reduce
from operator import mul

class ZOSVRGOptimizer():
    def __init__(self, model, loss, dataset, m, b, q, mu):
    
        self.model = model
        self.loss = loss
        self.dataset = dataset
        self.m = m
        self.b = b
        self.q = q
        self.mu = mu

        self.num_params = sum([reduce(mul, param.size(), 1) for param in model.parameters()])


    @torch.no_grad()
    def _rge(self, data, target):
        flat_params = self._get_flat_params()
        self._set_flat_params(flat_params + self.mu * self.m)

        loss1 = self.loss(self.model(data), target)
        flat_grad = torch.zeros_like(flat_params)
        for _ in range(self.q):
            u = torch.randn_like(flat_params)
            self._set_flat_params(flat_params + self.mu * u)
            loss2 = self.loss(self.model(data), target)
            flat_grad += (loss2 - loss1) / self.mu / self.q * u
            self._set_flat_params(flat_params)

        return flat_grad


    def _get_flat_params(self):
        params = []
        for module in self.model.modules():
            if len(module._parameters) != 0:
                for key in module._parameters.keys():
                    params.append(module._parameters[key].view(-1))

        return torch.cat(params)


    def _set_flat_params(self, flat_params):
        offset = 0
        for module in self.model.modules():
            if len(module._parameters) != 0:
                for key in module._parameters.keys():
                    param_shape = module._parameters[key].size()
                    param_flat_size = reduce(mul, param_shape, 1)
                    module._parameters[key] = flat_params[
                                               offset:offset + param_flat_size].view(*param_shape)
                    offset += param_flat_size


