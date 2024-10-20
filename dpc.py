import jax

seed = 0
parent_key = jax.random.PRNGKey(seed)

from jax import config
# config.update("jax_enable_x64", True)
# config.update('jax_platform_name', 'cpu')

import functools as ft
import numpy as np
from scipy.spatial.transform import Rotation # for initial dataset
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
import stats
import optax
from typing import Callable, List
from dataclasses import dataclass
from eqx_utils import filter_scan

from dynamics import f
from models import StochasticActor

def sample_P_x0(n_envs, qp):
    p0 = np.random.uniform(low=-3.0, high=3.0, size=[n_envs,3])
    euler0 = np.random.uniform(low=0, high=2*np.pi, size=[n_envs,3])
    q0 = Rotation.from_euler('xyz', euler0, degrees=False).as_quat()
    v0 = np.random.uniform(low=-1.0, high=1.0, size=[n_envs,3])
    omega0 = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=[n_envs,3])
    n0 = np.random.uniform(low=qp["minWmotor"], high=qp["maxWmotor"], size=[n_envs,4])
    pt_int0 = np.zeros([n_envs,3])
    x0 = np.hstack([p0,q0,v0,omega0,n0,pt_int0])
    return x0

def compute_loss(xs):
    # Example loss: mean squared error to some target state
    target_state = jnp.zeros(xs.shape[-1])
    loss = jnp.mean(jnp.sum((xs - target_state) ** 2, axis=-1))
    return loss

# we still go through all the data - but update before going through all of it - after every "minibatch"
def minibatch_step(key, actor, x0):
    # Sample random environment index and random start time t0
    env_idx_key, t0_key, key = jr.split(key, 3)
    env_idx = jr.randint(env_idx_key, shape=(), minval=0, maxval=n_envs)
    t0 = jr.randint(t0_key, shape=(), minval=0, maxval=unroll_len - bptt_hzn)
    
    # Get initial state for the selected environment
    x0_env = x0[env_idx]

    # Simulate dynamics up to time t0 without tracking gradients
    def body_fn(i, carry):
        x, key = carry
        actor_key, key = jr.split(key)
        u = actor(actor_key, x)
        x_next = f(x, u)
        x_next = jax.lax.stop_gradient(x_next)  # Prevent gradient tracking
        return x_next
    x_t0 = jax.lax.fori_loop(0, t0, body_fn, x0_env)
    
    # Simulate dynamics from x_t0 for bptt_hzn steps with gradient tracking
    def step_fn(state, _):
        x = state
        u = actor(x)
        x_next = f(x, u)
        return x_next, x  # Store state x for loss computation
    x_final, xs = jax.lax.scan(step_fn, x_t0, xs=None, length=bptt_hzn)
    
    # Compute loss over the trajectory segment xs
    loss = compute_loss(xs)
    
    # Compute gradients with respect to the actor's parameters
    grads = jax.grad(lambda params: compute_loss(xs))(actor.parameters)
    
    # Update the actor's parameters
    actor = update_parameters(actor, grads, lr)
    
    return actor

if __name__ == "__main__":

    from params import quad_params as qp, fl_params as fp
    from stats import RunningMeanStd

    key = jr.PRNGKey(seed=0)
    unroll_len = 5000
    bptt_hzn = 20
    n_envs = 3333
    n_mb = 5
    n_e = 400
    lr = 5e-3
    nx, nu = 20, 9

    # actor
    actor_key, key = jr.split(key)
    actor = StochasticActor(actor_key, [nx, 32, 32, 32, 32, int(nu*2)])

    # dynamics
    f = ft.partial(f, qp=qp, fp=fp)

    # 
    obs_rms = RunningMeanStd(mean=jnp.zeros(nx), var=jnp.ones(nx), count=jnp.array(0))

    # 
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(actor, eqx.is_array))

    #
    x0 = sample_P_x0(n_envs, qp)

    def loss(key, x0, actor, Q=5.0, R=0.1):
        _key, key = jr.split(key)

        def body(carry, xs):
            key, xk, idx, actor = carry
            _key, key = jr.split(key)
            yk = xk # obs_rms.normalize(xk)
            uk, _ = jax.vmap(actor)(jr.split(_key, n_envs), yk)
            xkp1 = jax.vmap(f)(x=xk, u=uk)
            stage_loss = R * jnp.sum(uk**2) + Q * jnp.sum(xkp1**2) # + penalty functions
            idx += 1

            # decide on gradient truncation
            trunc_grad_mask = idx % bptt_hzn
            def branch_continue():
                return xkp1
            def branch_trunc_grad():
                return jax.lax.stop_gradient(xkp1)
            xkp1 = jax.lax.cond(trunc_grad_mask, branch_continue, branch_trunc_grad)
            return (key, xkp1, idx, actor), stage_loss

        (_, xN, _, _), stage_losses = filter_scan(body, (_key, x0, jnp.array(0), actor), [], length=unroll_len)

        print('fin')

    loss(key, x0, actor)

    # 
    mb_key, key = jr.split(key)
    minibatch_step(mb_key, actor, x0)