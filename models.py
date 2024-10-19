import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
import stats

from typing import Callable, List
from dataclasses import dataclass


class StochasticActor(eqx.Module):
    """
    Stochastic actor with outputs of NN as noise scales (as opposed to a vec of scales)
    """
    layers: List[eqx.nn.Linear]
    activation: Callable
    layer_sizes: List[int]
    action_distribution: dataclass = eqx.field(static=True)

    def __init__(self, key, layer_sizes, activation=jax.nn.silu, action_distribution=stats.NormalTanhDistribution()):
        jax.debug.print("INFO: initializing PPOActorStochasticMLP, ensure final layer size is 2x action dim for mean and std")
        keys = jr.split(key, num=len(layer_sizes))
        self.layers = [
            eqx.nn.Linear(in_features, out_features, key=keys[i])
            for i, (in_features, out_features) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]))
        ]
        self.activation = activation
        self.layer_sizes = layer_sizes
        self.action_distribution = action_distribution

    def __call__(self, key, x):
        logits = self.mlp_forward(x)
        mean, raw_std = jnp.split(logits, 2)
        action = self.action_distribution.sample(key, mean, raw_std)
        return action, mean

    def mlp_forward(self, x):
        for linear in self.layers[:-1]:
            x = self.activation(linear(x))
        logits = self.layers[-1](x)  # Output mean
        return logits





