import jax
from absl import app, flags


# opt in early to change in JAX's RNG generation
# https://github.com/google/jax/discussions/18480
jax.config.update("jax_threefry_partitionable", True)
# allow use of 64-bit floats
jax.config.update("jax_enable_x64", True)
# force all operations on cpu (faster for bandit experiments)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from bandit_environments import BerBandit, BetaBandit, FixedBandit, GaussBandit
from experiment import run_experiment

FLAGS = flags.FLAGS

flags.DEFINE_integer("env_number", -1, "Environment number to run (-1 for all)")
flags.DEFINE_integer("algo_number", -1, "Environment number to run (-1 for all)")
flags.DEFINE_string("initial_policy", "uniform", "Initial policy to use {uniform, bad}")

flags.DEFINE_integer("t", 1000_000, "Number of iterations")
flags.DEFINE_string("exp_name", "linear_spg", "Experiment Name")
flags.DEFINE_string("save_dir", "./logs/", "Log directory")
flags.DEFINE_integer("runs_per_instance", 5, "Runs per instance")
flags.DEFINE_integer("num_instances", 25, "Number of instance")
flags.DEFINE_integer("env_seed", 1337, "Environment Seed")
flags.DEFINE_integer("exp_seed", 100, "Experiment Seed")

NUM_ARMS = 6


environment_definitions = [
    {
        "Bandit": BetaBandit,
        "bandit_kwargs": {"a_plus_b": 4, "K": NUM_ARMS, "d": 3},
        "max_reward_gap": 0.5,
        "environment_name": "Beta (easy)",
    },
    {
        "Bandit": BetaBandit,
        "bandit_kwargs": {"a_plus_b": 4, "K": NUM_ARMS, "d": 3},
        "max_reward_gap": 0.1,
        "environment_name": "Beta (hard)",
    },
    {
        "Bandit": GaussBandit,
        "bandit_kwargs": {"sigma": 0.1, "K": NUM_ARMS, "d": 3},
        "max_reward_gap": 0.5,
        "environment_name": "Gaussian (easy)",
    },
    {
        "Bandit": GaussBandit,
        "bandit_kwargs": {"sigma": 0.1, "K": NUM_ARMS, "d": 3},
        "max_reward_gap": 0.1,
        "environment_name": "Gaussian (hard)",
    },
    {
        "Bandit": BerBandit,
        "bandit_kwargs": {"K": NUM_ARMS, "d": 3},
        "max_reward_gap": 0.5,
        "environment_name": "Bernoulli (easy)",
    },
    {
        "Bandit": BerBandit,
        "bandit_kwargs": {"K": NUM_ARMS, "d": 3},
        "max_reward_gap": 0.1,
        "environment_name": "Bernoulli (hard)",
    },
]


def main(_):
    envs = []
    if FLAGS.env_number == -1:
        envs = environment_definitions
    else:
        envs.append(environment_definitions[FLAGS.env_number])
    print(envs)

    ALGOS = [
        # {
        #     "algo_name": f"linear_spg",
        #     "algo_kwargs": {},
        # },
        {
            "algo_name": "linear_spg_eta=0.01",
            "algo_kwargs": {"eta": 0.01},
        },
        {
            "algo_name": "linear_spg_eta=0.03",
            "algo_kwargs": {"eta": 0.03},
        },
        {
            "algo_name": "linear_spg_eta=0.1",
            "algo_kwargs": {"eta": 0.1},
        },
        {
            "algo_name": "linear_spg_eta=0.3",
            "algo_kwargs": {"eta": 0.3},
        },
        {
            "algo_name": "linear_spg_eta=1.0",
            "algo_kwargs": {"eta": 1.0},
        },
        {
            "algo_name": "linear_spg_eta=3.0",
            "algo_kwargs": {"eta": 3.0},
        },
        {
            "algo_name": "linear_spg_eta=10.0",
            "algo_kwargs": {"eta": 10.0},
        },
        {
            "algo_name": "linear_spg_eta=30.0",
            "algo_kwargs": {"eta": 30.0},
        },
        {
            "algo_name": "linear_spg_eta=100.0",
            "algo_kwargs": {"eta": 100.0},
        },
        {
            "algo_name": "linear_spg_eta=1000.0",
            "algo_kwargs": {"eta": 1000.0},
        },
    ]

    algos = []
    if FLAGS.algo_number == -1:
        algos = ALGOS
    else:
        algos.append(ALGOS[FLAGS.algo_number])
    print(algos)

    run_experiment(
        envs,
        algos,
        T=FLAGS.t,
        environment_seed=FLAGS.env_seed,
        experiment_seed=FLAGS.exp_seed,
        num_instances=FLAGS.num_instances,
        runs_per_instance=FLAGS.runs_per_instance,
        time_to_log=FLAGS.t // 100,
        log_dir=FLAGS.save_dir,
        exp_name=FLAGS.exp_name,
        intial_policy=FLAGS.initial_policy,
    )


if __name__ in "__main__":
    app.run(main)
