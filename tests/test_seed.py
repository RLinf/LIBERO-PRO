from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from liberopro.liberopro.envs.env_wrapper import ControlEnv


def test_seed_resets_both_rngs():
    wrapper = ControlEnv.__new__(ControlEnv)
    wrapper.env = SimpleNamespace(seed=None)
    state = np.random.get_state()
    try:
        wrapper.seed(23)
        expected_global = np.random.uniform(size=16)
        expected_local = wrapper.env.rng.uniform(size=16)

        np.random.seed(987)
        np.random.uniform(size=100)
        wrapper.seed(23)
        np.testing.assert_array_equal(np.random.uniform(size=16), expected_global)
        np.testing.assert_array_equal(wrapper.env.rng.uniform(size=16), expected_local)
        assert wrapper.env.seed == 23

        wrapper.seed(24)
        assert not np.array_equal(np.random.uniform(size=16), expected_global)
        assert not np.array_equal(wrapper.env.rng.uniform(size=16), expected_local)
    finally:
        np.random.set_state(state)


def test_seed_preserves_callable_branch():
    wrapper = ControlEnv.__new__(ControlEnv)
    wrapper.env = SimpleNamespace(seed=Mock())
    state = np.random.get_state()
    wrapper.seed(23)
    wrapper.env.seed.assert_called_once_with(23)
    assert not hasattr(wrapper.env, "rng")
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
