from mercurydpm2jax import mdpm_jax

import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

for name, target in mdpm_jax.registrations().items():
    jax.ffi.register_ffi_target(name, target)


@jax.jit
def run_dpm():
    num_particles = 2
    dim = 3

    # call MDPM, returns flattened array (x0,y0,z0,x1,y1,z1)
    out_type = jax.ShapeDtypeStruct((num_particles * dim,), jax.numpy.float32)
    positions = jax.ffi.ffi_call("run_dpm", out_type)()
    return positions.reshape(-1, 3)  # rehape


for i in range(10000):
    k = run_dpm()
    print(k)
