#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn.functional as F


class Constraint(torch.nn.Module):
    def __init__(self, bound, relation, name=None, multiplier_act=F.softplus, alpha=0., start_val=0.):
        """
        Adds a constraint to a loss function by turning the loss into a lagrangian.

        Alpha is used for a moving average as described in [1]. 
        Note that this is similar as using an optimizer with momentum.

        [1] Rezende, Danilo Jimenez, and Fabio Viola.
            "Taming vaes." arXiv preprint arXiv:1810.00597 (2018).

        Args:
            bound: Constraint bound.
            relation (str): relation of constraint,
                using naming convention from operator module (eq, le, ge).
                Defaults to 'ge'.
            name (str, optional): Constraint name
            multiplier_act (optional): When using inequality relations,
                an activation function is used to force the multiplier to be positive.
                I've experimented with ReLU, abs and softplus, softplus seems the most stable.
                Defaults to F.softplus.
            alpha (float, optional): alpha of moving average, as in [1].
                If alpha=0, no moving average is used.
            start_val (float, optional): Start value of multiplier. If an activation function
                is used the true start value might be different, because this is pre-activation.
        """
        super().__init__()
        self.name = name
        if isinstance(bound, (int, float)):
            self.bound = torch.Tensor([bound])
        elif isinstance(bound, list):
            self.bound = torch.Tensor(bound)
        else:
            self.bound = bound

        if relation in {'ge', 'le', 'eq'}:
            self.relation = relation
        else:
            raise ValueError('Unknown relation: {}'.format(relation))

        if self.relation == 'eq' and multiplier_act is not None:
            sys.stderr.write(
                "WARNING using an activation that maps to R+ with an equality \
                 constraint turns it into an inequality constraint"
            )

        self._multiplier = torch.nn.Parameter(
            torch.full((len(self.bound),), start_val))
        self._act = multiplier_act

        self.alpha = alpha
        self.avg_value = None

    @property
    def multiplier(self):
        if self._act is not None:
            return self._act(self._multiplier)
        return self._multiplier

    def forward(self, value):
        # Apply moving average, defined in [1]
        if self.alpha > 0:
            if self.avg_value is None:
                self.avg_value = value.detach().mean(0)
            else:
                self.avg_value = (self.avg_value * self.alpha + value.detach() * (1-self.alpha)).mean(0)
            value = value + (self.avg_value.unsqueeze(0) - value).detach()
        if self.relation in {'ge', 'eq'}:
            loss = self.bound.to(value.device) - value
        elif self.relation == 'le':
            loss = value - self.bound.to(value.device)
        return loss * self.multiplier


class Wrapper:
    """
    Simple class wrapper around  obj = obj_type(*args, **kwargs).
    Overwrites methods from obj with methods defined in Wrapper,
    else uses method from obj.
    """
    def __init__(self, obj_type, *args, **kwargs):
        self.obj = obj_type(*args, **kwargs)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.obj, attr)

    def __repr__(self):
        return 'Wrapped(' + self.obj.__repr__() + ')'


class ConstraintOptimizer(Wrapper):
    """
    Pytorch Optimizers only do gradient descent, but lagrangian needs
    gradient ascent for the multipliers. ConstraintOptimizer changes
    step() method of optimizer to do ascent instead of descent.

    I've gotten the best results using RMSprop with lr
    around 1e-3 and Constraint alpha=0.5.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, *args, **kwargs):
        super().__init__(optimizer, *args, **kwargs)

    def step(self, *args, **kwargs):
        # Maximize instead of minimize.
        for group in self.obj.param_groups:
            for p in group['params']:
                p.grad = -p.grad
        self.obj.step(*args, **kwargs)

    def __repr__(self):
        return 'ConstraintOptimizer (' + self.obj.__repr__() + ')'