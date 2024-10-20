import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
import equinox.internal as eqxi

@eqx.filter_jit
def filter_scan(f, init, xs, length=None, reverse=False, unroll=1):
    # Partition the initial carry and sequence inputs into dynamic and static parts
    init_dynamic, init_static = eqx.partition(init, eqx.is_array)
    xs_dynamic, xs_static = eqx.partition(xs, eqx.is_array)

    # Define the scanned function, handling the combination and partitioning
    def scanned_fn(carry_dynamic, x_dynamic):
        # Combine dynamic and static parts for the carry and input
        carry = eqx.combine(carry_dynamic, init_static)
        x = eqx.combine(x_dynamic, xs_static)

        # Apply the original function
        out_carry, out_y = f(carry, x)

        # Partition the outputs into dynamic and static parts
        out_carry_dynamic, out_carry_static = eqx.partition(out_carry, eqx.is_array)
        out_y_dynamic, out_y_static = eqx.partition(out_y, eqx.is_array)

        # Return dynamic outputs and wrap static outputs using Static to prevent tracing
        return out_carry_dynamic, (out_y_dynamic, eqxi.Static((out_carry_static, out_y_static)))

    # Use lax.scan with the modified scanned function
    final_carry_dynamic, (ys_dynamic, static_out) = lax.scan(
        scanned_fn, init_dynamic, xs_dynamic, length=length, reverse=reverse, unroll=unroll
    )

    # Extract static outputs
    out_carry_static, ys_static = static_out.value

    # Combine dynamic and static parts of the outputs
    final_carry = eqx.combine(final_carry_dynamic, out_carry_static)
    ys = eqx.combine(ys_dynamic, ys_static)

    return final_carry, ys


def vectorize_pytree(tree):
    """Flatten a PyTree and return a single vector of all the leaves (parameters)."""
    # Flatten the PyTree into leaves and a structure (treedef)
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    
    # Flatten each leaf into a 1D vector and concatenate them
    flat_vector = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
    
    return flat_vector, treedef, leaves  # Return the vector and the tree structure


def devectorize_pytree(vector, treedef, leaves):
    """Reconstruct the PyTree from a vector and the tree definition (treedef)."""
    # Compute the size of each flattened leaf
    sizes = [leaf.size for leaf in leaves]
    
    # Split the vector back into individual arrays corresponding to the original leaves
    split_leaves = jnp.split(vector, jnp.cumsum(sizes)[:-1])
    
    # Reshape each leaf back to its original shape
    reshaped_leaves = [leaf.reshape(original.shape) for leaf, original in zip(split_leaves, leaves)]
    
    # Reconstruct the PyTree from the reshaped leaves and the original tree structure
    return jax.tree_util.tree_unflatten(treedef, reshaped_leaves)