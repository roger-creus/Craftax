import jax
from jax import lax
from functools import partial
from gymnax.environments import spaces, environment
from typing import Optional

from craftax.craftax.envs.common import log_achievements_to_info
from craftax.environment_base.environment_bases import EnvironmentNoAutoReset
from craftax.craftax.constants import *
from craftax.craftax.game_logic import craftax_step, is_game_over
from craftax.craftax.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax.craftax.ascii_renderer import (
    render_craftax_ascii_grid_local,
    render_craftax_ascii_grid_full,
    create_discovered_map,
    ascii_grid_to_string,
)
from craftax.craftax.world_gen.world_gen import generate_world


class CraftaxAsciiEnvNoAutoReset(EnvironmentNoAutoReset):
    """Craftax environment returning ASCII string observations.

    Game logic (step, world gen) and grid computation (char/color IDs) are
    JIT-compiled.  Only the final string formatting runs in plain Python
    (outside JIT), since JAX cannot trace string operations.

    Because of this, ``step`` and ``reset`` are overridden *without* the
    base-class ``@jax.jit`` decorator.  Callers should use ``step()`` and
    ``reset()`` directly; ``step_env`` / ``reset_env`` are intentionally
    not implemented.

    Args:
        static_env_params: optional StaticEnvParams override
        view_mode: "local" for 9x11 window, "full" for 48x48 with fog of war
        include_legend: whether to include character legend in output
        use_color: whether to include ANSI color codes
    """

    def __init__(
        self,
        static_env_params: Optional[StaticEnvParams] = None,
        view_mode: str = "local",
        include_legend: bool = True,
        use_color: bool = True,
    ):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params
        self.view_mode = view_mode
        self.include_legend = include_legend
        self.use_color = use_color
        self.discovered_map = None

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def _jit_step(self, rng, state, action, params):
        """JIT-compiled game step (no string formatting)."""
        state, reward = craftax_step(rng, state, action, params, self.static_env_params)
        done = is_game_over(state, params, self.static_env_params)
        info = log_achievements_to_info(state, done)
        info["discount"] = jax.lax.select(done, 0.0, 1.0)
        return lax.stop_gradient(state), reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _jit_reset(self, rng, params):
        """JIT-compiled world generation."""
        rng, _rng = jax.random.split(rng)
        return generate_world(_rng, params, self.static_env_params)

    def step(self, key, state, action, params=None):
        """Step the environment. Game logic is JIT; string formatting is not."""
        if params is None:
            params = self.default_params
        state, reward, done, info = self._jit_step(key, state, action, params)
        obs = self.get_obs(state)
        return obs, state, reward, done, info

    def reset(self, key, params=None):
        """Reset the environment. World gen is JIT; string formatting is not."""
        if params is None:
            params = self.default_params
        state = self._jit_reset(key, params)
        self.discovered_map = create_discovered_map()
        obs = self.get_obs(state)
        return obs, state

    def step_env(self, rng, state, action, params):
        """Not used — step() overrides the base-class JIT boundary."""
        raise NotImplementedError("Use step() directly")

    def reset_env(self, rng, params):
        """Not used — reset() overrides the base-class JIT boundary."""
        raise NotImplementedError("Use reset() directly")

    def get_obs(self, state: EnvState) -> str:
        """Return ASCII string observation."""
        if self.view_mode == "full":
            if self.discovered_map is None:
                self.discovered_map = create_discovered_map()
            char_grid, color_grid, self.discovered_map, status_data = (
                render_craftax_ascii_grid_full(state, self.discovered_map)
            )
        else:
            char_grid, color_grid, status_data = render_craftax_ascii_grid_local(state)

        return ascii_grid_to_string(
            char_grid,
            color_grid,
            status_data,
            include_legend=self.include_legend,
            use_color=self.use_color,
        )

    def get_obs_grid(self, state: EnvState):
        """Return raw JAX char/color grids (for programmatic use)."""
        if self.view_mode == "full":
            if self.discovered_map is None:
                self.discovered_map = create_discovered_map()
            return render_craftax_ascii_grid_full(state, self.discovered_map)
        else:
            return render_craftax_ascii_grid_local(state)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params, self.static_env_params)

    @property
    def name(self) -> str:
        return "Craftax-ASCII-NoAutoReset-v1"

    @property
    def num_actions(self) -> int:
        return len(Action)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Shape of the underlying char/color grids (get_obs returns a string)."""
        if self.view_mode == "full":
            h, w = 48, 48
        else:
            h, w = OBS_DIM
        return spaces.Box(0, 255, (h, w, 2), dtype=jnp.int32)


class CraftaxAsciiEnv(environment.Environment):
    """Auto-reset version of the ASCII environment.

    See CraftaxAsciiEnvNoAutoReset for design notes on the JIT boundary.
    """

    def __init__(
        self,
        static_env_params: Optional[StaticEnvParams] = None,
        view_mode: str = "local",
        include_legend: bool = True,
        use_color: bool = True,
    ):
        super().__init__()

        if static_env_params is None:
            static_env_params = self.default_static_params()
        self.static_env_params = static_env_params
        self.view_mode = view_mode
        self.include_legend = include_legend
        self.use_color = use_color
        self.discovered_map = None

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def _jit_step(self, rng, state, action, params):
        """JIT-compiled game step with auto-reset."""
        state, reward = craftax_step(rng, state, action, params, self.static_env_params)
        done = is_game_over(state, params, self.static_env_params)
        info = log_achievements_to_info(state, done)
        info["discount"] = jax.lax.select(done, 0.0, 1.0)

        # Auto-reset logic
        rng, _rng = jax.random.split(rng)
        reset_state = generate_world(_rng, params, self.static_env_params)
        state = jax.tree_util.tree_map(
            lambda old, new: jax.lax.select(done, new, old),
            state,
            reset_state,
        )
        return lax.stop_gradient(state), reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _jit_reset(self, rng, params):
        """JIT-compiled world generation."""
        rng, _rng = jax.random.split(rng)
        return generate_world(_rng, params, self.static_env_params)

    def step(self, key, state, action, params=None):
        if params is None:
            params = self.default_params
        state, reward, done, info = self._jit_step(key, state, action, params)
        if bool(done):
            self.discovered_map = create_discovered_map()
        obs = self.get_obs(state)
        return obs, state, reward, done, info

    def reset(self, key, params=None):
        if params is None:
            params = self.default_params
        state = self._jit_reset(key, params)
        self.discovered_map = create_discovered_map()
        obs = self.get_obs(state)
        return obs, state

    def step_env(self, rng, state, action, params):
        """Not used — step() overrides the base-class JIT boundary."""
        raise NotImplementedError("Use step() directly")

    def reset_env(self, rng, params):
        """Not used — reset() overrides the base-class JIT boundary."""
        raise NotImplementedError("Use reset() directly")

    def get_obs(self, state: EnvState) -> str:
        if self.view_mode == "full":
            if self.discovered_map is None:
                self.discovered_map = create_discovered_map()
            char_grid, color_grid, self.discovered_map, status_data = (
                render_craftax_ascii_grid_full(state, self.discovered_map)
            )
        else:
            char_grid, color_grid, status_data = render_craftax_ascii_grid_local(state)

        return ascii_grid_to_string(
            char_grid,
            color_grid,
            status_data,
            include_legend=self.include_legend,
            use_color=self.use_color,
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        return is_game_over(state, params, self.static_env_params)

    @property
    def name(self) -> str:
        return "Craftax-ASCII-v1"

    @property
    def num_actions(self) -> int:
        return len(Action)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Shape of the underlying char/color grids (get_obs returns a string)."""
        if self.view_mode == "full":
            h, w = 48, 48
        else:
            h, w = OBS_DIM
        return spaces.Box(0, 255, (h, w, 2), dtype=jnp.int32)
