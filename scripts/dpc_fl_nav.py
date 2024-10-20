import functools as ft
import numpy as np
# from scipy.spatial.transform import Rotation # for initial dataset
from jax.scipy.spatial.transform import Rotation
import os
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import equinox as eqx
import optax
jax.config.update("jax_debug_nans", False)
from eqx_utils import filter_scan
from models import DeterministicTanhActor as Actor
from stats import RunningMeanStd
from dynamics import f_fl_patt_pr_step as f
from params import quad_params as qp, ctrl_params as cp, sim_params as sp
from plotting import Animator

key = jr.PRNGKey(seed=0)
_key, key = jr.split(key)

f = ft.partial(f, d=[0.]*3, qp=qp, cp=cp, Ts=sp["Ts"])
actor = Actor(_key, [12, 32, 32, 32, 32, int(9*2)])
obs_rms = RunningMeanStd(jnp.zeros(12), jnp.ones(12), jnp.array(0))

n_envs = 2
n_sgd_steps_per_epoch = 1
n_epochs = 1000 # 400 # 2 # 400
lr = 1e-3
R = 1.0  # Coefficient for action cost
Q = 1.0  # Coefficient for state cost
bptt_hzn = 10 # 20 # Truncation horizon
unroll_len = 100 # 100 # Total unroll length

@eqx.filter_jit
def sample_P_x0(key, n_envs, qp):
    _key, key = jr.split(key)
    p0 = jr.uniform(_key, minval=-3.0, maxval=3.0, shape=[n_envs,3]); _key, key = jr.split(key)
    euler0 = jr.uniform(_key, minval=0, maxval=2*jnp.pi, shape=[n_envs,3]); _key, key = jr.split(key)
    q0 = Rotation.from_euler('xyz', euler0, degrees=False).as_quat()
    v0 = jr.uniform(_key, minval=-1.0, maxval=1.0, shape=[n_envs,3]); _key, key = jr.split(key)
    omega0 = jr.uniform(_key, minval=-jnp.pi/2, maxval=jnp.pi/2, shape=[n_envs,3]); _key, key = jr.split(key)
    n0 = jr.uniform(_key, minval=qp["minWmotor"], maxval=qp["maxWmotor"], shape=[n_envs,4]); _key, key = jr.split(key)
    pt_int0 = jnp.zeros([n_envs,3])
    x0 = jnp.hstack([p0,q0,v0,omega0,n0,pt_int0])
    return x0

# Initialize variables

opt = optax.adam(lr)
opt_state = opt.init(eqx.filter(actor, eqx.is_array))

def get_obs(x):
    return jnp.hstack([x[0:3], x[7:13], x[17:20]])

def loss_function(actor, key, x0, obs_rms):

    _key, key = jr.split(key)

    # Modified body function
    def scan_rollout(carry, _):
        key, xk, idx = carry
        key, subkey = jr.split(key)
        _key, key = jr.split(key)
        yk = obs_rms.normalize(jax.vmap(get_obs)(xk)) # jnp.hstack([xk[:,0:3], xk[:,7:13], xk[:,17:20]]) # xk  # If normalization is needed, apply here
        uk, _ = jax.vmap(actor)(jr.split(subkey, n_envs), yk)
        xkp1 = jax.vmap(f)(x=xk, u=uk)
        
        # ignore attitude error and rotor speed
        xke = jax.vmap(get_obs)(xkp1) # jnp.hstack([xkp1[:,0:3], xkp1[:,7:13], xkp1[:,17:20]])

        stage_loss = R * jnp.sum(uk**2) + Q * jnp.sum(xke**2)
        idx += 1

        # Decide on state reset upon flight envelope departure
        x_fl_lb = jnp.hstack([qp["x_lb"], cp["tilde_p_lb"]])
        x_fl_ub = jnp.hstack([qp["x_ub"], cp["tilde_p_ub"]])
        termination_mask = jnp.logical_or(xkp1 < x_fl_lb, xkp1 > x_fl_ub).any(axis=1)[:,None]
        # jax.debug.print("termination mask: {}", termination_mask)

        
        xkp1 = xkp1 * ~termination_mask + sample_P_x0(_key, n_envs, qp) * termination_mask
        # test_mask = jnp.logical_or(xkp1 < x_fl_lb, xkp1 > x_fl_ub).any(axis=1)[:,None]
        # jax.debug.print("post_reset termination mask: {}", test_mask)

        # jax.debug.breakpoint()

        # xkp1_rs = jax.lax.cond(termination_mask != 0 ,lambda: sample_P_x0(1, qp) ,lambda: xkp1)

        # Decide on gradient truncation
        trunc_grad_mask = idx % bptt_hzn
        xkp1_sg = jax.lax.cond(trunc_grad_mask != 0,
                            lambda: xkp1,
                            lambda: jax.lax.stop_gradient(xkp1))

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
    (carry_final, outputs) = jax.lax.scan(scan_rollout, init_carry, None, length=unroll_len)


    return jnp.sum(outputs["stage_loss"]) / n_envs / unroll_len, outputs 

def sgd_step(carry, _):

    key, actor, opt_state, obs_rms = carry
    _key, key = jr.split(key)

    # sample initial state distribution
    x0 = sample_P_x0(_key, n_envs, qp)

    _ = loss_function(actor, _key, x0, obs_rms)

    (loss, outputs), grads = eqx.filter_value_and_grad(loss_function, has_aux=True)(actor, _key, x0, obs_rms)

    nan_to_num = lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    grads = jax.tree.map(nan_to_num, grads)

    new_obs_rms = obs_rms.update(jax.vmap(jax.vmap(get_obs))(outputs["xk"]))

    # jax.debug.breakpoint()
    jax.debug.print("loss: {}", loss)
    # jax.debug.print("obs_rms.mean: {}", obs_rms.mean)
    # jax.debug.print("obs_rms.var: {}", obs_rms.var)

    updates, new_opt_state = opt.update(
        grads, opt_state, actor
    )

    new_actor = eqx.apply_updates(actor, updates)

    return (key, new_actor, new_opt_state, new_obs_rms), loss

best_loss = jnp.inf
best_actor = jax.tree.map(lambda x: x, actor)
for i in range(n_epochs):
    # perform an epoch
    carry = (_key, actor, opt_state, obs_rms)
    carry, loss = filter_scan(sgd_step, carry, None, length=n_sgd_steps_per_epoch)
    _, actor, opt_state, obs_rms = carry

    if loss < best_loss:
        print("saving new best actor")
        best_actor = jax.tree.map(lambda x: x, actor)
        best_loss = loss

# test the result

# plot an inferenced trajectory with start end points
n_test_envs = 30

_key, key = jr.split(key)

s = sample_P_x0(_key, n_test_envs, qp)

s_hist, a_hist = [], []
# pol_s = get_params(best_opt_s)
roll_len = 1000
for i, t in enumerate(range(roll_len)):
    _key, key = jr.split(key)
    y = obs_rms.normalize(jax.vmap(get_obs)(s))
    a, m = jax.vmap(best_actor)(jr.split(_key, n_test_envs), y)
    print(f"state: {s[0]}")
    print(f"action: {m[0]}")

    s = jax.vmap(f)(s, m)

    # resets failures
    x_fl_lb = jnp.hstack([qp["x_lb"], cp["tilde_p_lb"]])
    x_fl_ub = jnp.hstack([qp["x_ub"], cp["tilde_p_ub"]])
    termination_mask = jnp.logical_or(s < x_fl_lb, s > x_fl_ub).any(axis=1)[:,None]
    s = s * ~termination_mask + sample_P_x0(_key, n_test_envs, qp) * termination_mask
    s_hist.append(s); a_hist.append(a)

s_hist = np.stack(s_hist)
a_hist = np.stack(a_hist)
fig, ax = plt.subplots()
for i in range(n_test_envs):
    ax.plot(s_hist[:,i,0], s_hist[:,i,1])
# ax.add_patch(Circle(cs[0,:2], cs[0,2]))
ax.set_aspect('equal')
ax.set_ylim(-5, 5)
ax.set_xlim(-5, 5)
plt.savefig("test_dpc_fl_cyl.png", dpi=500)

animator = Animator(qp, s_hist[:,0], np.arange(roll_len), a_hist[:,0,0:3])
animator.animate()

print('fin')