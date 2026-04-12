#!/usr/bin/env python3
"""LLM agent for Craftax using vLLM offline inference with ASCII observations.

Each step is fully Markovian: the model receives the complete game state
(ASCII observation + legend + action space + floor-specific tutorial guidance)
with NO history aggregation.  The model is prompted to reason before acting.

Usage:
    python run_llm_agent.py
    python run_llm_agent.py --model Qwen/Qwen3.5-9B --max-steps 500
    python run_llm_agent.py --view-mode full --no-render
"""
import argparse
import os
import sys
import time
import json
import re

import jax
import jax.numpy as jnp
import numpy as np
import imageio

from vllm import LLM, SamplingParams

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.ascii_renderer import render_craftax_ascii, create_discovered_map
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG, Action


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------
ACTION_NAMES = {a.name: a.value for a in Action}
ACTION_BY_VALUE = {a.value: a.name for a in Action}
NUM_ACTIONS = len(Action)

ACTION_FUZZY = {}
for name, val in ACTION_NAMES.items():
    ACTION_FUZZY[name.lower()] = val
    ACTION_FUZZY[name.lower().replace("_", " ")] = val
    ACTION_FUZZY[name.lower().replace("_", "")] = val
ACTION_FUZZY.update({
    "move left": 1, "move right": 2, "move up": 3, "move down": 4,
    "go left": 1, "go right": 2, "go up": 3, "go down": 4,
    "west": 1, "east": 2, "north": 3, "south": 4,
    "left": 1, "right": 2, "up": 3, "down": 4,
    "interact": 5, "attack": 5, "mine": 5, "use": 5,
    "sleep": 6, "rest": 17, "wait": 0, "nothing": 0, "noop": 0,
    "descend": 18, "ascend": 19,
    "shoot": 24, "shoot arrow": 24, "fireball": 26, "iceball": 27,
    "place torch": 28, "torch": 28,
})


from agent_memory import (
    SpatialMemory,
    parse_agent_response,
    EMPTY_SCRATCHPAD,
    MEMORY_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Tutorial text split by floor
# ---------------------------------------------------------------------------

GENERAL_GUIDE = """\
# Craftax Game Guide

Craftax is a roguelike survival game with 9 floors (0-8).  You control a character (shown as ^v<> indicating your facing direction) on an ASCII map.

## Core Mechanics
- You have 5 intrinsics: HP, Food, Drink, Energy, Mana.
- Food, Drink, Energy decay over time. Replenish by eating (kill cows/eat ripe plants), drinking (face water + DO), sleeping (SLEEP restores HP when food/drink/energy > 0).
- HP recovers when food, drink AND energy are all > 0.  If HP reaches 0 you die.
- Mana recovers naturally; used for spells and enchanting.
- To interact with the block you are FACING, use DO.  You must be adjacent and facing toward it.
- To face a direction, move in that direction first (LEFT/RIGHT/UP/DOWN).  If blocked, you turn without moving.
- Your absolute position Pos:(row,col) is shown in the status bar. Use it to navigate and record locations.
- The map is 48x48 (rows 0-47, cols 0-47). Blank spaces at the edges are out of bounds — you cannot go there. Turn around!
- When eating or drinking, repeat DO until the stat is FULL. Leaving with partial food/drink is risky.

## Progression
- Mine trees (face tree + DO) to get wood.  Place a crafting table (PLACE_TABLE), stand adjacent, craft tools.
- Craft pickaxe to mine stone/ores.  Place furnace (PLACE_FURNACE) to smelt iron.
- Kill 8 monsters per floor to unlock the ladder down (>).  Stand on > and DESCEND.
- Each new floor awards XP to spend on Dexterity, Strength, or Intelligence.

## Attributes
- Dexterity: increases max food/drink/energy, slows decay, increases bow damage.
- Strength: increases melee damage and max HP.
- Intelligence: increases max mana, spell/enchantment damage.

## Potions
- Found in chests. 6 colours, effects randomized each game (±health, ±mana, ±energy).
- Experiment carefully!

## Day/Night Cycle
- The overworld has a day/night cycle shown in [Status] as Day, Dusk, or Night.
- At night, more hostile mobs spawn on the overworld. Be cautious or sleep through it.
- Underground floors are always dark regardless of time — you need torches (PLACE_TORCH).
- Torches are crafted from 1 wood + 1 coal (MAKE_TORCH).

## Combat Tips
- Block yourself in with stone (PLACE_STONE) before sleeping — you are vulnerable while asleep.
- REST executes no-ops until an intrinsic hits 0, you're attacked, or you're fully healed.
- Craft arrows (MAKE_ARROW) for ranged combat with your bow (SHOOT_ARROW).

## Crafting Reference
IMPORTANT: Crafting is automatic — just use the crafting ACTION while standing adjacent to the
required station(s) with the right materials in your inventory. There is NO smelting step.
Raw ores (iron, coal, diamond, etc.) are used directly in recipes. Do NOT use DO on a crafting
table or furnace — that will DESTROY it! Only use the specific MAKE_* action keys.
- Adjacent to crafting table: MAKE_WOOD_PICKAXE (1 wood), MAKE_WOOD_SWORD (2 wood)
- MAKE_STONE_PICKAXE (1 wood + 1 stone), MAKE_STONE_SWORD (1 wood + 2 stone)
- Adjacent to crafting table AND furnace: MAKE_IRON_PICKAXE (1 wood + 1 iron + 1 coal), MAKE_IRON_SWORD (1 wood + 2 iron + 1 coal)
- MAKE_DIAMOND_PICKAXE (1 wood + 3 diamonds), MAKE_DIAMOND_SWORD (1 wood + 2 diamonds)
- MAKE_IRON_ARMOUR (3 iron + 3 coal per piece), MAKE_DIAMOND_ARMOUR (3 diamonds per piece)
- MAKE_ARROW (1 wood + 1 stone), MAKE_TORCH (1 wood + 1 coal)
- Enchanting: adjacent to enchantment table, 9 mana + sapphire (ice) or ruby (fire)
- All crafting/enchanting actions work instantly if you have materials + correct adjacency.
"""

FLOOR_GUIDES = {
    0: """\
## Floor 0: Overworld (Current Floor)
Grasslands, lakes and mountains.  Trees (T) give wood, stone (%) needs a pickaxe.
- PRIORITY: Mine trees → place crafting table → craft wood pickaxe → mine stone → craft stone tools → mine coal/iron → craft iron tools.
- Eat cows (u) for food, drink from water (~), sleep when energy is low.
- Collect saplings from grass (DO on grass) and plant them (PLACE_PLANT) for future food.
- Collect lots of wood before descending — it's scarce underground.
- Find the ladder down (>) and DESCEND when ready.
- The overworld ladder is always open (no kill requirement).""",

    1: """\
## Floor 1: Dungeon (Current Floor)
Connected rooms with paths.  Contains fountains ({) and chests ($).
- First chest gives a bow.  Craft arrows (MAKE_ARROW) at a crafting table.
- Kill 8 orcs/mages to unlock the ladder down (>).
- Snails (n) can be eaten for food.
- Drink from fountains ({) by facing them + DO.
- If low HP: block yourself in with stone, then SLEEP or REST.""",

    2: """\
## Floor 2: Gnomish Mines (Current Floor)
Dark cavern — you MUST place torches (PLACE_TORCH) to see.  Craft torches from wood + coal.
- Rich in ores: coal, iron, diamonds, sapphires, rubies along cavern edges.
- Bats (B) can be eaten.  Pools of water (~) for drinking.
- Gnomes are stronger than orcs — avoid being surrounded in open spaces.
- Diamonds: craft diamond sword (2) or pickaxe (3).  Diamond pickaxe mines sapphires/rubies.
- Craft iron armour (3 iron + 3 coal per piece) or diamond armour (3 diamonds per piece).""",

    3: """\
## Floor 3: Sewers (Current Floor)
Dungeon layout with water patches.  Fill water gaps by placing stone then mining it.
- Lizards swim through water — very dangerous!  Kobolds throw high-damage daggers.
- First chest contains a BOOK — read it (READ_BOOK) to learn fireball or iceball spell.
- Spells cost 2 mana (CAST_FIREBALL / CAST_ICEBALL).  They do fire/ice damage.
- Later enemies resist physical damage — spells become essential!
- ICE enchantment table (E) is on this floor.  Enchant items with sapphires (9 mana + sapphire).
- Enchanting sword/bow: +50% ice damage.  Enchanting armour: -20% ice damage per piece.""",

    4: """\
## Floor 4: Vaults (Current Floor)
Another dungeon level.  Contains a second book and the FIRE enchantment table (E).
- Knights and archers are armoured — physical damage is HALVED against them.
- Use enchanted weapons or spells (fire/ice) to deal effective damage.
- Enchant with rubies at the fire enchantment table (9 mana + ruby).""",

    5: """\
## Floor 5: Troll Mines (Current Floor)
Dark cavern, richest in ores.  Place torches to see.
- PRIORITY: Mine everything — craft full diamond armour if possible.
- Trolls are very strong.  Deep things in water are weak but have deadly ranged attacks.
- Rich in coal, iron, diamonds, sapphires, rubies.""",

    6: """\
## Floor 6: Fire Realm (Current Floor)
Islands separated by lava (&).  No water on this level!
- Build bridges: PLACE_STONE on lava, then mine the stone to create a path.
- Pigmen and fire elementals are immune to fire damage and resist physical.
- ICE damage is required (ice spells or ice-enchanted weapons).
- Periodically ASCEND to Troll Mines to drink water.
- Lots of coal and rubies available.""",

    7: """\
## Floor 7: Ice Realm (Current Floor)
Dark level — place torches.  Frost trolls and ice elementals are the strongest creatures.
- They are immune to ice damage — FIRE damage required.
- No food on this level — ASCEND to Fire Realm to eat.
- Lots of sapphires and rubies in the mountains.
- This is the last floor before the boss!""",

    8: """\
## Floor 8: Graveyard — BOSS LEVEL (Current Floor)
The necromancer (N) is the final boss.  No ladder out.  No food/water, but intrinsics don't decay.
- The necromancer summons waves of enemies from graves (+).
- Kill all enemies in a wave → necromancer becomes VULNERABLE (N turns yellow).
- Attack the vulnerable necromancer (face + DO) to trigger the next wave.
- Waves correspond to creatures from each floor (floor 0 through floor 7).
- After defeating the final wave (ice realm), attack the necromancer to WIN!""",
}

ACTION_SPACE_TEXT = """\
## Available Actions
Movement: LEFT, RIGHT, UP, DOWN (also turns you to face that direction)
Interact: DO (mine block / attack creature / drink water / eat plant / open chest — in your facing direction)
Rest: SLEEP (restores energy; HP recovers while asleep if food/drink/energy>0; wakes at max energy or if attacked; 2.5x damage while asleep!)
      REST (idles until HP is full, food=0, drink=0, or attacked; normal damage)
Place: PLACE_STONE, PLACE_TABLE, PLACE_FURNACE, PLACE_PLANT, PLACE_TORCH
Craft (adjacent to table): MAKE_WOOD_PICKAXE, MAKE_STONE_PICKAXE, MAKE_IRON_PICKAXE, MAKE_DIAMOND_PICKAXE
  MAKE_WOOD_SWORD, MAKE_STONE_SWORD, MAKE_IRON_SWORD, MAKE_DIAMOND_SWORD
  MAKE_IRON_ARMOUR, MAKE_DIAMOND_ARMOUR, MAKE_ARROW, MAKE_TORCH
Combat: SHOOT_ARROW, CAST_FIREBALL, CAST_ICEBALL
Spells: READ_BOOK, ENCHANT_SWORD, ENCHANT_ARMOUR, ENCHANT_BOW
Potions: DRINK_POTION_RED, DRINK_POTION_GREEN, DRINK_POTION_BLUE, DRINK_POTION_PINK, DRINK_POTION_CYAN, DRINK_POTION_YELLOW
Navigate: DESCEND (go down ladder >), ASCEND (go up ladder <)
Level up: LEVEL_UP_DEXTERITY, LEVEL_UP_STRENGTH, LEVEL_UP_INTELLIGENCE
Do nothing: NOOP"""


def build_system_prompt(floor_idx):
    """Build the full system prompt with general guide + floor-specific guidance."""
    floor_guide = FLOOR_GUIDES.get(
        int(floor_idx), f"## Floor {int(floor_idx)} (no specific guide available)"
    )
    return f"""{GENERAL_GUIDE}

{floor_guide}

{ACTION_SPACE_TEXT}

{MEMORY_INSTRUCTIONS}"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LLM agent for Craftax (Markovian)")
    # Model
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    p.add_argument("--max-model-len", type=int, default=32864)
    p.add_argument("--max-tokens", type=int, default=16384)
    # Qwen3.5 thinking mode recommended params
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    # Environment
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=9999999)
    p.add_argument("--seed", type=int, default=42)
    # Rendering
    p.add_argument("--view-mode", choices=["local", "full"], default="local")
    p.add_argument("--use-color", action="store_true", default=False,
                   help="Use ANSI colors in obs sent to LLM")
    # Output
    p.add_argument("--save-gif", action="store_true", default=True)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--output-dir", type=str, default="results_llm")
    p.add_argument("--render-terminal", action="store_true", default=True)
    p.add_argument("--no-render", action="store_true")
    return p.parse_args()


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def messages_to_text(messages, tokenizer):
    """Convert chat messages to a prompt string using the tokenizer's template."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)


def main():
    args = parse_args()
    save_gif = args.save_gif and not args.no_gif
    render_terminal = args.render_terminal and not args.no_render

    # Create timestamped run directory to avoid overwriting previous results
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    args.output_dir = run_dir
    print(f"Results will be saved to: {run_dir}")

    # --- Load LLM ---
    print(f"Loading model: {args.model}")
    print(f"  max_model_len={args.max_model_len}, max_tokens={args.max_tokens}")
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )
    print("Model loaded!")

    # --- Setup environment ---
    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    params = env.default_params
    step_fn = jax.jit(env.step)
    render_pixels_fn = jax.jit(render_craftax_pixels, static_argnums=(1,))

    rng = jax.random.PRNGKey(args.seed)
    all_episode_metrics = []
    all_interactions = []

    for ep in range(args.num_episodes):
        rng, _rng = jax.random.split(rng)
        obs, state = env.reset(_rng, params)

        discovered_map = create_discovered_map() if args.view_mode == "full" else None
        scratchpad = EMPTY_SCRATCHPAD
        spatial_mem = SpatialMemory()
        recent_actions = []  # last 5 (step, action_name, reward)
        pending_query_results = ""
        feedback_msg = ""

        frames = []
        total_reward = 0.0
        achievements_unlocked = set()
        step_times = []
        llm_times = []
        episode_interactions = []

        for step_i in range(args.max_steps):
            t0 = time.time()

            # --- Skip LLM call while sleeping/resting (actions are ignored) ---
            if bool(state.is_sleeping) or bool(state.is_resting):
                status = "sleeping" if bool(state.is_sleeping) else "resting"
                rng, _rng = jax.random.split(rng)
                obs, state, reward, done, info = step_fn(
                    _rng, state, 0, params  # NOOP
                )
                total_reward += float(reward)
                if render_terminal:
                    sys.stdout.write(
                        f"\r  [{status}... HP:{int(state.player_health)} "
                        f"Energy:{int(state.player_energy)} "
                        f"Food:{int(state.player_food)}]  "
                    )
                    sys.stdout.flush()
                if bool(done):
                    print(f"\n--- Died while {status} at step {step_i+1} ---")
                    break
                continue

            # --- Get current floor for floor-specific prompt ---
            floor_idx = int(state.player_level)

            # --- Get ASCII observation (no color for LLM, always with legend) ---
            ascii_obs, discovered_map = render_craftax_ascii(
                state,
                discovered_map=discovered_map,
                view_mode=args.view_mode,
                include_legend=True,
                use_color=args.use_color,
            )

            # --- Build prompt: system (with floor guide) + user (observation + history + scratchpad) ---
            system_prompt = build_system_prompt(floor_idx)
            # Format recent action history
            if recent_actions:
                n = len(recent_actions)
                history_lines = "\n".join(
                    f"  {n - idx} turns ago (turn {s}): {a} -> reward {r:+.1f}"
                    for idx, (s, a, r) in enumerate(reversed(recent_actions))
                )
                history_str = f"Recent actions (most recent first):\n{history_lines}"
            else:
                history_str = "Recent actions: (none yet, this is your first turn)"
            # Memory context
            mem_parts = []
            if pending_query_results:
                mem_parts.append(pending_query_results)
                pending_query_results = ""
            if spatial_mem.data.get(floor_idx):
                mem_parts.append(f"Saved locations: {spatial_mem.format_floor_summary(floor_idx)}")
            mem_str = "\n".join(mem_parts)

            feedback_prefix = ""
            if feedback_msg:
                feedback_prefix = f"[SYSTEM FEEDBACK] {feedback_msg}\n\n"
                feedback_msg = ""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        feedback_prefix
                        + f"Turn {step_i + 1}. Current observation:\n\n"
                        f"{ascii_obs}\n\n"
                        f"{history_str}\n\n"
                        + (f"{mem_str}\n\n" if mem_str else "")
                        + f"Your current scratchpad:\n{scratchpad}\n\n"
                        f"Analyze the situation, choose your next action, "
                        f"and use MAP_ADD/MAP_GET/MAP_REMOVE commands as needed.\n"
                        f"Remember: end with --- delimiter, then Action: and optional MAP_*/Scratchpad."
                    ),
                },
            ]
            prompt_text = messages_to_text(messages, tokenizer)
            input_tokens = len(tokenizer.encode(prompt_text))

            # --- LLM inference ---
            t_llm = time.time()
            outputs = llm.generate([prompt_text], sampling_params)
            llm_time = time.time() - t_llm
            llm_times.append(llm_time)

            generated_text = outputs[0].outputs[0].text
            output_tokens = len(outputs[0].outputs[0].token_ids)

            # --- Parse everything from the structured block ---
            action, parsed_name, new_scratchpad, query_results, reasoning = (
                parse_agent_response(generated_text, spatial_mem, ACTION_FUZZY)
            )
            if parsed_name == "NOOP (parse failed)" and generated_text:
                feedback_msg = (
                    "Your previous response could not be parsed into a valid action. "
                    "Make sure to end your response with:\n---\nAction: <ACTION_NAME>\n"
                    "Valid actions include: LEFT, RIGHT, UP, DOWN, DO, SLEEP, REST, "
                    "PLACE_TABLE, MAKE_WOOD_PICKAXE, etc."
                )
            if query_results:
                pending_query_results = query_results
            if new_scratchpad:
                scratchpad = new_scratchpad

            # --- Step environment ---
            rng, _rng = jax.random.split(rng)
            obs, state, reward, done, info = step_fn(_rng, state, action, params)
            total_reward += float(reward)
            step_time = time.time() - t0
            step_times.append(step_time)

            # --- Update recent action history (keep last 5) ---
            recent_actions.append((step_i + 1, parsed_name, float(reward)))
            if len(recent_actions) > 5:
                recent_actions.pop(0)

            # Track achievements
            achiev = np.array(state.achievements)
            new_achievements = []
            for i, v in enumerate(achiev):
                if v and i not in achievements_unlocked:
                    achievements_unlocked.add(i)
                    new_achievements.append(i)

            # --- Log interaction ---
            interaction = {
                "step": step_i + 1,
                "floor": floor_idx,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "llm_time_s": round(llm_time, 2),
                "action_parsed": parsed_name,
                "action_id": action,
                "reward": float(reward),
                "total_reward": round(total_reward, 4),
                "achievements": len(achievements_unlocked),
                "scratchpad": scratchpad[:1000],
                "spatial_memory_count": sum(len(v) for v in spatial_mem.data.values()),
                "llm_output": generated_text[:2000],
            }
            episode_interactions.append(interaction)

            # --- Render to terminal ---
            if render_terminal:
                display_ascii, _ = render_craftax_ascii(
                    state,
                    discovered_map=discovered_map,
                    view_mode=args.view_mode,
                    include_legend=False,
                    use_color=True,
                )

                clear_screen()
                print(
                    f"=== LLM AGENT ({args.model.split('/')[-1]}) | "
                    f"Ep {ep+1}/{args.num_episodes} | "
                    f"Step {step_i+1}/{args.max_steps} ==="
                )
                print(
                    f"Action: {parsed_name} (id={action}) | "
                    f"Reward: {float(reward):.2f} | "
                    f"Total: {total_reward:.2f} | "
                    f"Achievements: {len(achievements_unlocked)}"
                )
                if new_achievements:
                    print(f"  NEW ACHIEVEMENTS: {new_achievements}")
                print(
                    f"LLM: {input_tokens} in / {output_tokens} out / "
                    f"{llm_time:.1f}s"
                )
                print()
                print(display_ascii)
                print()
                # Show scratchpad
                print("--- Scratchpad ---")
                pad_display = scratchpad
                if len(pad_display) > 500:
                    pad_display = pad_display[:500] + "..."
                print(pad_display)
                # Show spatial memory summary
                mem_total = sum(len(v) for v in spatial_mem.data.values())
                if mem_total > 0:
                    print(f"\n--- Spatial Memory ({mem_total} entries) ---")
                    for f_idx in sorted(spatial_mem.data.keys()):
                        print(f"  {spatial_mem.format_floor_summary(f_idx)}")
                print()
                # Show truncated reasoning
                if len(reasoning) > 300:
                    reasoning = reasoning[:300] + "..."
                print(f"LLM: {reasoning}")

            # --- Record frame for GIF ---
            if save_gif:
                pixels = render_pixels_fn(state, BLOCK_PIXEL_SIZE_IMG)
                frames.append(np.array(pixels, dtype=np.uint8))

            if bool(done):
                print(f"\n--- Episode {ep+1} ended at step {step_i+1} (game over) ---")
                break

        # --- Episode metrics ---
        avg_step_time = np.mean(step_times) if step_times else 0
        avg_llm_time = np.mean(llm_times) if llm_times else 0
        metrics = {
            "episode": ep + 1,
            "steps": step_i + 1,
            "total_reward": round(total_reward, 4),
            "achievements": len(achievements_unlocked),
            "avg_step_time_s": round(avg_step_time, 2),
            "avg_llm_time_s": round(avg_llm_time, 2),
            "model": args.model,
            "view_mode": args.view_mode,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        all_episode_metrics.append(metrics)

        print(f"\n{'='*60}")
        print(f"Episode {ep+1} Summary:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")

        # Save GIF
        if save_gif and frames:
            gif_path = os.path.join(args.output_dir, f"llm_ep{ep+1}.gif")
            imageio.mimsave(gif_path, frames, fps=args.gif_fps, loop=0)
            print(f"  GIF saved: {gif_path} ({len(frames)} frames)")

        if spatial_mem.data:
            mem_path = os.path.join(args.output_dir, f"spatial_memory_ep{ep+1}.json")
            spatial_mem.save(mem_path)
            print(f"  Spatial memory saved: {mem_path}")

        all_interactions.append({
            "episode": ep + 1,
            "config": {
                "model": args.model,
                "view_mode": args.view_mode,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
            "interactions": episode_interactions,
        })

        # Save after EACH episode (so partial runs aren't lost)
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_episode_metrics, f, indent=2)
        interactions_path = os.path.join(args.output_dir, "interactions.json")
        with open(interactions_path, "w") as f:
            json.dump(all_interactions, f, indent=2)
        print(f"  Saved metrics & interactions to {args.output_dir}")
    print(f"Interactions saved: {interactions_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"LLM AGENT SUMMARY ({args.model})")
    print(f"{'='*60}")
    rewards = [m["total_reward"] for m in all_episode_metrics]
    achievs = [m["achievements"] for m in all_episode_metrics]
    print(f"  Episodes: {len(all_episode_metrics)}")
    print(f"  Avg reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
    print(f"  Avg achievements: {np.mean(achievs):.1f} +/- {np.std(achievs):.1f}")
    print(f"  Best reward: {max(rewards):.4f}")
    print(f"  Best achievements: {max(achievs)}")
    if llm_times:
        print(f"  Avg LLM time: {np.mean(llm_times):.2f}s/step")


if __name__ == "__main__":
    main()
