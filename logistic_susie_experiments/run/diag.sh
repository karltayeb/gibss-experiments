#!/bin/bash
# Diagnose the XLA:CPU AOT cache mismatch on a COMPUTE node. Run:
#   srun --account=pi-mstephens --partition=caslake --pty bash run/diag.sh
# or as a 1-task job:  sbatch --account=pi-mstephens --partition=caslake \
#   --wrap 'bash run/diag.sh'  (from logistic_susie_experiments/)
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== host CPU =="; lscpu | grep -iE 'model name|flags' | head -2 | cut -c1-200
echo; echo "== JAX/XLA env BEFORE our exports =="; env | grep -iE 'jax|xla|cache' || echo "(none)"

echo; echo "== apply the sbatch's cache disables =="
export GIBSS_NO_JAX_CACHE=1
unset JAX_COMPILATION_CACHE_DIR
export JAX_ENABLE_COMPILATION_CACHE=false

echo "== what jax actually sees + a trivial compile =="
uv run python - <<'PY'
import jax, jax.numpy as jnp, jaxlib
print("jax", jax.__version__, "| jaxlib", jaxlib.__version__)
print("cache_dir =", jax.config.jax_compilation_cache_dir)
print("enable    =", getattr(jax.config, "jax_enable_compilation_cache", "n/a"))
print("backend   =", jax.default_backend())
# a real (jit) compile+run — if the AOT warning fires HERE, it's the persistent cache;
# if it SIGILLs, that's the crash killing the jobs.
x = jnp.arange(1000.0)
print("trivial jit result:", float(jax.jit(lambda z: (z*z).sum())(x)))
PY
echo "== exit $? =="
