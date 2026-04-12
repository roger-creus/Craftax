#!/usr/bin/env python3
"""Random agent for Craftax with real-time ASCII terminal rendering and GIF recording."""
import argparse
import os
import sys
import time
import json

import jax
import jax.numpy as jnp
import numpy as np
import imageio

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.ascii_renderer import render_craftax_ascii, create_discovered_map
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG, Action


def parse_args():
    parser = argparse.ArgumentParser(description="Random agent for Craftax")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view-mode", choices=["local", "full"], default="local")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--save-gif", action="store_true", default=True)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--gif-fps", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results_random")
    parser.add_argument(
        "--render-interval", type=int, default=1, help="Render every N steps"
    )
    return parser.parse_args()


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def main():
    args = parse_args()
    use_color = not args.no_color
    save_gif = args.save_gif and not args.no_gif

    os.makedirs(args.output_dir, exist_ok=True)

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    params = env.default_params
    step_fn = jax.jit(env.step)
    render_pixels_fn = jax.jit(render_craftax_pixels, static_argnums=(1,))

    rng = jax.random.PRNGKey(args.seed)
    action_names = {a.value: a.name for a in Action}
    num_actions = len(Action)

    all_episode_metrics = []

    for ep in range(args.num_episodes):
        rng, _rng = jax.random.split(rng)
        obs, state = env.reset(_rng, params)

        discovered_map = create_discovered_map() if args.view_mode == "full" else None

        frames = []
        total_reward = 0.0
        achievements_unlocked = set()
        step_times = []

        for step_i in range(args.max_steps):
            t0 = time.time()

            rng, _rng = jax.random.split(rng)
            action = int(jax.random.randint(_rng, (), 0, num_actions))

            rng, _rng = jax.random.split(rng)
            obs, state, reward, done, info = step_fn(_rng, state, action, params)
            total_reward += float(reward)
            step_times.append(time.time() - t0)

            for i, v in enumerate(np.array(state.achievements)):
                if v:
                    achievements_unlocked.add(i)

            if step_i % args.render_interval == 0:
                ascii_str, discovered_map = render_craftax_ascii(
                    state,
                    discovered_map=discovered_map,
                    view_mode=args.view_mode,
                    include_legend=False,
                    use_color=use_color,
                )

                clear_screen()
                print(
                    f"=== RANDOM AGENT | Episode {ep+1}/{args.num_episodes} | "
                    f"Step {step_i+1}/{args.max_steps} ==="
                )
                print(
                    f"Action: {action_names.get(action, '?')} | "
                    f"Reward: {float(reward):.2f} | "
                    f"Total: {total_reward:.2f} | "
                    f"Achievements: {len(achievements_unlocked)}"
                )
                print()
                print(ascii_str)

            if save_gif:
                pixels = render_pixels_fn(state, BLOCK_PIXEL_SIZE_IMG)
                frames.append(np.array(pixels, dtype=np.uint8))

            if bool(done):
                print(f"\n--- Episode {ep+1} ended at step {step_i+1} ---")
                break

        avg_step_time = np.mean(step_times) if step_times else 0
        metrics = {
            "episode": ep + 1,
            "steps": step_i + 1,
            "total_reward": round(total_reward, 4),
            "achievements": len(achievements_unlocked),
            "avg_step_time_ms": round(avg_step_time * 1000, 2),
            "steps_per_second": round(1.0 / avg_step_time, 1)
            if avg_step_time > 0
            else 0,
        }
        all_episode_metrics.append(metrics)

        print(f"\n{'='*60}")
        print(f"Episode {ep+1} Summary:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")

        if save_gif and frames:
            gif_path = os.path.join(args.output_dir, f"random_ep{ep+1}.gif")
            imageio.mimsave(gif_path, frames, fps=args.gif_fps, loop=0)
            print(f"  GIF saved: {gif_path} ({len(frames)} frames)")

    metrics_path = os.path.join(args.output_dir, "random_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_episode_metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    print(f"\n{'='*60}")
    print("RANDOM AGENT SUMMARY")
    print(f"{'='*60}")
    rewards = [m["total_reward"] for m in all_episode_metrics]
    achievs = [m["achievements"] for m in all_episode_metrics]
    print(f"  Episodes: {len(all_episode_metrics)}")
    print(f"  Avg reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
    print(f"  Avg achievements: {np.mean(achievs):.1f} +/- {np.std(achievs):.1f}")
    print(f"  Best reward: {max(rewards):.4f}")
    print(f"  Best achievements: {max(achievs)}")


if __name__ == "__main__":
    main()
