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

from eqx_utils import filter_scan
from models import StochasticActor
from stats import RunningMeanStd
from dynamics import f_fl_patt_pr as f
from params import quad_params as qp, ctrl_params as cp, sim_params as sp

key = jr.PRNGKey(seed=0)
_key, key = jr.split(key)

f = ft.partial(f, d=[0.]*3, qp=qp, cp=cp)
actor = StochasticActor(_key, [20, 32, 32, 32, 32, 18])

n_envs = 1
R = 1.0  # Coefficient for action cost
Q = 1.0  # Coefficient for state cost
bptt_hzn = 100  # Truncation horizon
unroll_len = 100  # Total unroll length

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
actions = outputs['uk']  # Shape: (unroll_len, n_envs)
states = outputs['xk']   # Shape: (unroll_len, n_envs)
stage_losses = outputs['stage_loss']  # Shape: (unroll_len,)

# Reshape actions and states into truncated sections
num_sections = (unroll_len + bptt_hzn - 1) // bptt_hzn
padded_len = num_sections * bptt_hzn

# Reshape into sections
actions_sections = actions.reshape(num_sections, bptt_hzn, -1)
states_sections = states.reshape(num_sections, bptt_hzn, -1)

# Compute gradients for each truncated section
grad_magnitudes = []
for idx in range(num_sections):
    actions_sec = actions_sections[idx]  # Shape: (bptt_hzn, n_envs)
    actions_sec_shadow = jnp.copy(actions_sections[idx])  # Shape: (bptt_hzn, n_envs)
    states_sec = states_sections[idx]    # Shape: (bptt_hzn, n_envs)
    initial_state = states_sec[0]

    # Define the loss function for the section
    def section_loss_fn(initial_state, actions_sec):
        state = initial_state
        total_loss = 0.0
        for i, uk in enumerate(actions_sec):
            # if i == 0:
            #     state = state + f(x=state, u=uk) * sp["Ts"]
            # else:
            #     state = state + f(x=state, u=actions_sec_shadow[i]) * sp["Ts"]
            state = state + f(x=state, u=uk) * sp["Ts"]

            loss = R * jnp.sum(uk**2) + Q * jnp.sum(state**2)
            total_loss += loss
        return total_loss

    # Compute gradients w.r.t initial state and actions
    grad_fn = jax.grad(section_loss_fn, argnums=(0, 1))
    grads = grad_fn(initial_state, actions_sec)

    # Compute the magnitude of gradients
    grad_mag_initial_state = jnp.linalg.norm(grads[0])
    grad_mag_actions = jnp.linalg.norm(grads[1], axis=1)  # Per time step

    grad_magnitudes.append({
        'initial_state': grad_mag_initial_state,
        'actions': grad_mag_actions
    })

    # Print the JAXpr for the first truncated section
    if idx == 0:
        jaxpr = jax.make_jaxpr(section_loss_fn)(initial_state, actions_sec)
        print("JAXpr of the first truncated section:")
        print(jaxpr)

# Plot the gradient magnitudes
plt.figure()
plt.title(f"Truncated Sections dL w.r.t dAction")
for idx, grad_mag in enumerate(grad_magnitudes):
    plt.plot(grad_mag['actions'], label=f"Truncated Section {idx+1}")
plt.xlabel('Step within Truncated Section')
plt.ylabel('Gradient Magnitude')
plt.yscale('log')  # Set y-axis to log scale

plt.xticks(ticks=range(len(grad_mag['actions'])))
plt.legend()
plt.savefig("test.png", dpi=500)