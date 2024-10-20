import functools as ft
import numpy as np
import jax

jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

jax.config.update("jax_debug_nans", True)

import os
import sys

here = os.path.dirname(__file__)

sys.path.append(os.path.join(here, '..'))

import geometry

from params import quad_params as qp, ctrl_params as cp, sim_params as sp
from dynamics import f_fl_patt_pr_step as f
from plotting import Animator

# Initialize state and reference
xk = jnp.array([
    0, 0, 0,                # p  (position, x, y, z)
    1, 0, 0, 0,             # q  (quaternion angle, q0, q1, q2, q3)
    0, 0, 0,                # pd (velocity, xd, yd, zd)
    0, 0, 0,                # omega (body rotation rate p,q,r)
    *[522.9847140714692]*4, # n  (rotational velocity of rotors)
    0, 0, 0                 # tilde_p_sum (feedback linearization integration state x, y, z)
])
rk = jnp.copy(xk)

t = jnp.arange(sp["Ti"], sp["Tf"], sp["Ts"])
x = [jnp.copy(xk)]
r = [jnp.copy(rk)]

f = ft.partial(f, d=[0.]*3, qp=qp, cp=cp, Ts=sp["Ts"])

p_d_fn = jax.jit(lambda tk: jnp.array([jnp.sin(tk), jnp.cos(tk), jnp.sin(2*tk)]))
dp_d_fn = jax.jit(jax.jacobian(p_d_fn))
ddp_d_fn = jax.jit(jax.jacobian(dp_d_fn))

for tk in tqdm(t):

    # Desired Trajectory:
    # -------------------

    # pringle

    p_d = p_d_fn(tk)
    dp_d = dp_d_fn(tk)
    ddp_d = ddp_d_fn(tk)

    # p_d =   jnp.array([jnp.sin(tk), jnp.cos(tk), jnp.sin(2*tk)])
    # dp_d =  jnp.array([jnp.cos(tk), -jnp.sin(tk), 2*jnp.cos(2*tk)])
    # ddp_d = jnp.array([-jnp.sin(tk), -jnp.cos(tk), -4*jnp.sin(2*tk)])

    # cicular
    # p_d =   jnp.array([jnp.sin(tk), jnp.cos(tk), 0])
    # dp_d =  jnp.array([jnp.cos(tk), -jnp.sin(tk), 0])
    # ddp_d = jnp.array([-jnp.sin(tk), -jnp.cos(tk), 0])

    uk = jnp.hstack([p_d, dp_d, ddp_d])

    rk = rk.at[0:3].set(p_d)  # Desired position

    # up and down
    # hz = 1
    # p_d = np.array([0,0,-np.sin(tk * hz)])
    # dp_d = np.array([0,0,-np.cos(tk * hz)])
    # ddp_d = np.array([0,0,np.sin(tk * hz)])

    # stand still
    # p_d = np.array([0,0,0])
    # dp_d = np.array([0,0,0])
    # ddp_d = np.array([0,0,0])

    # step the system
    xk = f(xk, uk)

    # Clip the rotor speeds within limits
    # xk = xk.at[13:17].set(jnp.clip(xk[13:17], qp["x_lb"][13:17], qp["x_ub"][13:17]))

    # save
    # ----
    x.append(jnp.copy(xk))
    r.append(jnp.copy(rk))

x = np.asarray(jnp.stack(x))
r = np.asarray(jnp.stack(r))
t = np.asarray(t)

animator = Animator(qp, x, t, r)
animator.animate()

plt.close()
plt.plot(x[:,0], x[:,1], label="xy")
plt.plot(x[:,0], x[:,2], label="xz")
plt.legend()
plt.savefig('data/images/test.png', dpi=500)
