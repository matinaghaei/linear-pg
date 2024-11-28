from flax.struct import dataclass
import jax
import jax.numpy as jnp


@dataclass
class Bandit:
    mean_r: jnp.array
    K: int = -1
    best_arm: int = 0
    instance_number: int = 0
    name: str = "Bandit"
    d: int = None
    features: jnp.array = None

    @classmethod
    def create(cls, instance_number, name, **kwargs):
        best_arm = jnp.argmax(kwargs["mean_r"])
        return cls(
            instance_number=instance_number,
            best_arm=best_arm,
            name=name,
            **kwargs,
        )

    def randomize(self, key):
        return self.mean_r


@dataclass
class BerBandit(Bandit):
    def randomize(self, key):
        rt = jax.random.uniform(key, shape=(self.K,))
        return jnp.array(rt < self.mean_r).astype(jnp.float32)


@dataclass
class GaussBandit(Bandit):
    sigma: float = 0.1
    name = "Gaussian Bandit"

    def randomize(self, key):
        rt = jax.random.normal(key, shape=(self.K,))
        return jnp.clip(self.mean_r + self.sigma * rt, 0, 1)


@dataclass
class BetaBandit(Bandit):
    a_plus_b: float = 0
    name = "Beta Bandit"

    def randomize(self, key):
        return jax.random.beta(
            key, self.a_plus_b * self.mean_r, self.a_plus_b * (1 - self.mean_r)
        )


@dataclass
class FixedBandit(Bandit):
    name = "Fixed Bandit"


def generate_random_features(key, K, d):
    theta_key, features_key, reward_key = jax.random.split(key, 3)
    X = jax.random.uniform(features_key, (K, d))
    mu = jax.random.uniform(reward_key, (K,)).sort(descending=True)
    return X, mu


def generate_realizable_rewards(key, K, d):
    theta_key, features_key, reward_key = jax.random.split(key, 3)
    X = jax.random.uniform(features_key, (K, d))
    theta = jax.random.uniform(theta_key, (d,))
    mu = X @ theta
    return X, mu


def check_reward_realizability(X, r):
    proj = X @ jnp.linalg.inv(X.T @ X) @ X.T
    return jnp.allclose(r, proj @ r)


def check_reward_ordering(X, r):
    proj = X @ jnp.linalg.inv(X.T @ X) @ X.T
    r_ = proj @ r
    return len(jnp.unique(r_)) == len(r_) and jnp.array_equal(r.argsort(), r_.argsort())


def check_non_domination(X):
    matrix = X @ X.T
    p = matrix.shape[0]
    flag = True
    for i in range(p):
        for j in range(p):
            if i != j and matrix[i, j] >= matrix[i, i]:
                flag = False
    return flag


def check_3_arm_det_feature_ordering(X, r):
    order = (-r).argsort()
    return (X[order[1]] - X[order[2]]) @ (X[order[0]] - X[order[2]]) > 0


def check_3_arm_sto_feature_ordering(X, r):
    order = (-r).argsort()
    return (X[order[0]] - X[order[1]]) @ (X[order[1]] - X[order[2]]) > 0


def check_multi_arm_feature_ordering(X, r):
    order = (-r).argsort()
    K = X.shape[0]
    for i in range(1, K):
        for j in range(i+1, K):
            for k in range(i+1, K):
                if (X[order[i]] - X[order[j]]) @ (X[order[0]] - X[order[k]]) <= 0:
                    return False
    return True


def make_bandit(
    env_key,
    instance_number,
    bandit_class,
    bandit_kwargs,
    environment_name,
    min_reward_gap=None,
    max_reward_gap=0.5,
):
    """
    Generates a bandit environment with K arms, with a random mean reward vector in [0, 1]^K

    To vary the difficutly of the bandit, the reward vectors are sampled in the range of `[0.5 - max_reward_gap/2, 0.5 + max_reward_gap/2]`

    To ensure that the bandit is not too easy/hard, we subtract the `min_reward_gap`, then clip the rewards to be in [0, 1]^K

    Note: if `min_reward_gap` is not None, then we cannot guarentee that the mean reward are in the range of `[0.5 - max_reward_gap/2, 0.5 + max_reward_gap/2]`
    """

    if "d" in bandit_kwargs:

        d = bandit_kwargs["d"]
        K = bandit_kwargs["K"]
        X, mean_reward = generate_realizable_rewards(env_key, K, d)

        if not check_3_arm_det_feature_ordering(X, mean_reward):
            return None
        
        bandit_kwargs["features"] = X

    else:
        
        mean_reward = max_reward_gap * jax.random.uniform(
            env_key, (bandit_kwargs["K"],)
        ) + (0.5 - max_reward_gap / 2)
        mean_reward = mean_reward.sort()
        if min_reward_gap is not None:
            mean_reward = mean_reward.at[:-1].add(-min_reward_gap)
            mean_reward = jnp.clip(mean_reward, 0.0, 1.0)
        print(f"reward gap: {mean_reward[-1] - mean_reward[0]}")
    
    bandit = bandit_class.create(
        instance_number, environment_name, **bandit_kwargs
    )
    return bandit


def make_envs(env_def, num_instances, key):
    envs = []
    created = 0
    while created < num_instances:
        key, env_key = jax.random.split(key)

        if env_def["Bandit"] is FixedBandit:
            env = env_def["Bandit"].create(
                created,
                env_def["environment_name"],
                **env_def["bandit_kwargs"],
            )
        else:
            env = make_bandit(
                env_key,
                created,
                env_def["Bandit"],
                env_def["bandit_kwargs"],
                env_def["environment_name"],
                min_reward_gap=env_def.get("min_reward_gap", None),
                max_reward_gap=env_def.get("max_reward_gap", 0.5),
            )
        if env is not None:
            envs.append(env)
            created += 1
    return envs
