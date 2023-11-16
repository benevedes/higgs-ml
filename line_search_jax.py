import jax
import jax.numpy as jnp

def potential(phi, m, beta):
    return -m**2/4*phi**2 + beta/4*jnp.log(phi)*phi**4