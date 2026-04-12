#!/usr/bin/env python3
"""Demo script for the ASCII renderer — random agent with terminal rendering.

Runs a random agent in Craftax and renders the game state as a NetHack-style
ASCII grid in the terminal.  Supports both local (9x11) and full (48x48 with
fog of war) view modes.

Usage:
    python examples/demo_ascii_renderer.py
    python examples/demo_ascii_renderer.py --view-mode full --max-steps 500
    python examples/demo_ascii_renderer.py --no-color --view-mode local
"""
import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.ascii_renderer import render_craftax_ascii, create_discovered_map
from craftax.craftax.constants import Action


def parse_args():
    parser = argparse.ArgumentParser(description="Craftax ASCII renderer demo")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--view-mode",
        choices=["local", "full"],
        default="local",
        help="local = 9x11 window, full = 48x48 with fog of war",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable ANSI color codes"
    )
    parser.add_argument(
        "--render-interval",
        type=int,
        default=1,
        help="Render to terminal every N steps",
    )
    return parser.parse_args()


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def main():
    args = parse_args()
    use_color = not args.no_color

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    params = env.default_params
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng, params)

    discovered_map = create_discovered_map() if args.view_mode == "full" else None

    num_actions = len(Action)
    action_names = {a.value: a.name for a in Action}
    total_reward = 0.0
    achievements = set()

    for step_i in range(args.max_steps):
        # Random action
        rng, _rng = jax.random.split(rng)
        action = int(jax.random.randint(_rng, (), 0, num_actions))

        # Step
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = step_fn(_rng, state, action, params)
        total_reward += float(reward)

        # Track achievements
        for i, v in enumerate(np.array(state.achievements)):
            if v:
                achievements.add(i)

        # Render
        if step_i % args.render_interval == 0:
            ascii_str, discovered_map = render_craftax_ascii(
                state,
                discovered_map=discovered_map,
                view_mode=args.view_mode,
                include_legend=(step_i == 0),
                use_color=use_color,
            )

            clear_screen()
            print(
                f"Step {step_i + 1}/{args.max_steps} | "
                f"Action: {action_names.get(action, '?')} | "
                f"Reward: {float(reward):.2f} | "
                f"Total: {total_reward:.2f} | "
                f"Achievements: {len(achievements)}"
            )
            print()
            print(ascii_str)

        if bool(done):
            print(f"\nGame over at step {step_i + 1}")
            break

    print(
        f"\nFinal: {step_i + 1} steps, reward={total_reward:.2f}, "
        f"achievements={len(achievements)}"
    )


if __name__ == "__main__":
    main()
