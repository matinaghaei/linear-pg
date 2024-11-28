import jax
import jax.numpy as jnp
# opt in early to change in JAX's RNG generation
# https://github.com/google/jax/discussions/18480
jax.config.update("jax_threefry_partitionable", True)
# allow use of 64-bit floats
jax.config.update("jax_enable_x64", True)
# force all operations on cpu (faster for bandit experiments)
jax.config.update("jax_platform_name", "cpu")

from bandit_environments import Bandit, FixedBandit
from experiment import run_experiment

T = 1_000_000
TIME_TO_LOG = T // 100
NUM_ARMS = 3
LOG_DIR = f"logs"
EXP_NAME = f"linear_pg"
INTIAL_POLICY = "uniform"
NUM_INSTANCES = 1
ENV_SEED = 1337
EXP_SEED = 1337 + 42

environment_definitions = [
    # {
    #     "Bandit": Bandit,
    #     "bandit_kwargs": {"d": 2},
    #     "environment_name": "Random",
    # },
    # {
    #     "Bandit": Bandit,
    #     "bandit_kwargs": {"d": 3},
    #     "environment_name": "Random",
    # },
    # {
    #     "Bandit": Bandit,
    #     "bandit_kwargs": {"d": 4},
    #     "environment_name": "Random",
    # },
    # {
    #     "Bandit": Bandit,
    #     "bandit_kwargs": {"d": 5},
    #     "environment_name": "Random",
    # },
    # {
    #     "Bandit": Bandit,
    #     "bandit_kwargs": {"d": 6},
    #     "environment_name": "Random",
    # },
    {
        "Bandit": FixedBandit,
        "environment_name": "Deterministic",
        "bandit_kwargs": {
            "K": NUM_ARMS,
            "d": 2, 
            "features": jnp.array([[0.0, -1.0], [0.6, 0.6], [1.0, 0.0]]), 
            "mean_r": jnp.array([1.0, -1.8, -2.0])
        }
    },
]

algos = [
    {
        "algo_name": "linear_pg",
        "algo_kwargs": {},
        "theta_0": jnp.array([2., 2.])
    },
    # {
    #     "algo_name": "linear_pg_eta=0.01",
    #     "algo_kwargs": {"eta": 0.01},
    # },
    # {
    #     "algo_name": "linear_pg_eta=0.03",
    #     "algo_kwargs": {"eta": 0.03},
    # },
    # {
    #     "algo_name": "linear_pg_eta=0.1",
    #     "algo_kwargs": {"eta": 0.1},
    # },
    # {
    #     "algo_name": "linear_pg_eta=0.3",
    #     "algo_kwargs": {"eta": 0.3},
    # },
    # {
    #     "algo_name": "linear_pg_eta=1.0",
    #     "algo_kwargs": {"eta": 1.0},
    # },
]

run_experiment(
    environment_definitions,
    algos,
    T=T,
    environment_seed=ENV_SEED,
    experiment_seed=EXP_SEED,
    num_instances=NUM_INSTANCES,
    runs_per_instance=1,
    time_to_log=TIME_TO_LOG,
    log_dir=LOG_DIR,
    exp_name=EXP_NAME,
    intial_policy=INTIAL_POLICY,
)
