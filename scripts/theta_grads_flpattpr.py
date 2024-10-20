import functools as ft
import numpy as np
from scipy.spatial.transform import Rotation # for initial dataset
import os
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import equinox as eqx
from tqdm import tqdm

from eqx_utils import filter_scan
from models import StochasticActor
from stats import RunningMeanStd
from dynamics import f_fl as f
from params import quad_params as qp, ctrl_params as cp, sim_params as sp

key = jr.PRNGKey(seed=0)
_key, key = jr.split(key)

f = ft.partial(f, d=[0.]*3, qp=qp, cp=cp)
actor = StochasticActor(_key, [20, 32, 32, 32, 32, 18])
actor_shadow = StochasticActor(_key, [20, 32, 32, 32, 32, 18])

n_envs = 1
R = 1.0  # Coefficient for action cost
Q = 1.0  # Coefficient for state cost
bptt_hzn = 100  # Truncation horizon
unroll_len = 1000  # Total unroll length

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


# Initialize variables
x0 = sample_P_x0(n_envs, qp)

# Modified body function
def body(carry, _):
    key, xk, idx = carry
    key, subkey = jr.split(key)
    yk = xk  # If normalization is needed, apply here
    uk, _ = jax.vmap(actor)(jr.split(subkey, n_envs), yk)
    xkp1 = xk + jax.vmap(f)(x=xk, u=uk) * sp["Ts"]
    stage_loss = R * jnp.sum(uk**2) + Q * jnp.sum(xkp1**2)
    idx += 1

    # Decide on gradient truncation
    trunc_grad_mask = idx % bptt_hzn
    xkp1_sg = jax.lax.cond(trunc_grad_mask != 0,
                           lambda: xkp1,
                           lambda: jax.lax.stop_gradient(xkp1))

    # Prepare outputs to collect actions and states
    outputs = {
        'uk': uk,
        'xk': xk,
        'stage_loss': stage_loss
    }

    # Update the carry
    carry = (key, xkp1_sg, idx)
    return carry, outputs

# Run the scan
init_carry = (_key, x0, 0)
(carry_final, outputs) = jax.lax.scan(body, init_carry, None, length=unroll_len)

# Extract actions, states, and losses
actions = outputs['uk']  # Shape: (unroll_len, n_envs, state_dim)
states = outputs['xk']   # Shape: (unroll_len, n_envs, state_dim)
stage_losses = outputs['stage_loss']  # Shape: (unroll_len,)

# Reshape actions and states into truncated sections
num_sections = (unroll_len + bptt_hzn - 1) // bptt_hzn  # Should be 1 in this case
padded_len = num_sections * bptt_hzn

# Pad actions and states if necessary
# actions_padded = jnp.pad(actions, ((0, padded_len - unroll_len), (0, 0)))
# states_padded = jnp.pad(states, ((0, padded_len - unroll_len), (0, 0)))

# Reshape into sections
actions_sections = actions.reshape(num_sections, bptt_hzn, -1)
states_sections = states.reshape(num_sections, bptt_hzn, -1)

# Compute gradients for each truncated section
grad_magnitudes = []
grad_variances = []
for idx in range(num_sections):
    actions_sec = actions_sections[idx]  # Shape: (bptt_hzn, n_envs * state_dim)
    states_sec = states_sections[idx]    # Shape: (bptt_hzn, n_envs * state_dim)

    grad_norms = []
    grad_vars = []
    for t in tqdm(range(bptt_hzn)):
        # Define the loss function from time t onwards
        def loss_from_t_fn(actor):
            state = states_sec[t]
            total_loss = 0.0
            for k in range(t, bptt_hzn):
                # if k == t:
                #     uk, _ = actor(_key, state)
                # else:
                #     uk, _ = actor_shadow(_key, state)
                uk, _ = actor(_key, state)
                state = state + f(x=state, u=uk) * sp["Ts"]
                loss = R * jnp.sum(uk**2) + Q * jnp.sum(state**2)
                total_loss += loss
            return total_loss

        # Compute gradients w.r.t actor parameters
        grad_fn = eqx.filter_grad(loss_from_t_fn)
        grads = grad_fn(actor)

        # Compute the magnitude of gradients
        grads = eqx.filter(grads, eqx.is_array)
        flattened_leaves = [jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(grads)]
        all_params_vector = jnp.concatenate(flattened_leaves)
        grad_norm = jnp.linalg.norm(all_params_vector)
        grad_var = all_params_vector.var()
        grad_norms.append(grad_norm)
        grad_vars.append(grad_var)

    grad_magnitudes.append(grad_norms)
    grad_variances.append(grad_vars)

# # Plot the gradient magnitudes

# Create two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# First subplot: Gradient Norms
ax1.set_title('Gradient Norms over Time in Truncated Sections')
for idx, grad_norms in enumerate(grad_magnitudes):
    ax1.plot(grad_norms, label=f"Truncated Section {idx+1}")
ax1.set_xlabel('Time Step within Truncated Section')
ax1.set_ylabel('Gradient Norm')
ax1.set_yscale('log')  # Set y-axis to log scale
ax1.set_xticks(range(len(grad_magnitudes[0])))
# ax1.legend()

# Second subplot: Gradient Variances
ax2.set_title('Gradient Variances over Time in Truncated Sections')
for idx, grad_vars in enumerate(grad_variances):
    ax2.plot(grad_vars, label=f"Truncated Section {idx+1}")
ax2.set_xlabel('Time Step within Truncated Section')
ax2.set_ylabel('Gradient Variance')
ax2.set_yscale('log')  # Set y-axis to log scale
ax2.set_xticks(range(len(grad_variances[0])))
# ax2.legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("test.png", dpi=500)
