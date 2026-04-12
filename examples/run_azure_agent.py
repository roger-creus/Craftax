#!/usr/bin/env python3
"""Azure OpenAI (GPT) agent for Craftax.

Each step is Markovian (full state observation) with an external scratchpad
and spatial memory. The model is prompted to reason before acting.

Requires AZURE_ENDPOINT, AZURE_API_KEY, AZURE_MODEL_NAME in .env.

Usage:
    python examples/run_azure_agent.py
    python examples/run_azure_agent.py --max-steps 500 --reasoning-effort high
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

from openai import AzureOpenAI

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.ascii_renderer import render_craftax_ascii, create_discovered_map
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG, Action

from agent_memory import (
    SpatialMemory,
    StructuredScratchpad,
    parse_agent_response,
    MEMORY_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Load API key from .env
# ---------------------------------------------------------------------------
def load_env(path=".env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


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


# ---------------------------------------------------------------------------
# Tutorial text split by floor
# ---------------------------------------------------------------------------

GENERAL_GUIDE = """\
# Craftax Game Guide

## Goal
Descend through all 9 floors (0-8) and defeat the Necromancer on floor 8.

Craftax is a roguelike survival game. You control a character (shown as ^v<> indicating your facing direction) on an ASCII map.

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
- Found in chests. 6 colours, effects randomized each game (+-health, +-mana, +-energy).
- Experiment carefully!

## Day/Night Cycle
- The overworld has a day/night cycle shown in [Status] as Day, Dusk, or Night.
- At night, more hostile mobs spawn on the overworld. Be cautious or sleep through it.
- Underground floors are always dark regardless of time — you need torches (PLACE_TORCH).
- Torches are crafted from 1 wood + 1 coal (MAKE_TORCH).

## Combat Tips
- Armour significantly reduces damage taken. Iron armour (3 iron + 3 coal per piece) helps a lot;
  diamond armour (3 diamonds per piece) is even better. Worth crafting when you can.
- Diamond sword (1 wood + 2 diamonds) deals the most melee damage in the game.
- Important: you take 2.5x damage while asleep. Consider blocking entry points with
  PLACE_STONE or using natural walls/corners/tunnels before sleeping.
- Craft arrows (MAKE_ARROW) for ranged combat with your bow (SHOOT_ARROW).

## Crafting Reference
Crafting is automatic — use the crafting ACTION while standing adjacent to the required
station(s) with the right materials in your inventory. There is no smelting step.
Raw ores (iron, coal, diamond, etc.) are used directly in recipes. Do not use DO on a crafting
table or furnace — that will destroy it. Only use the specific MAKE_* action keys.
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
- PRIORITY: Mine trees -> place crafting table -> craft wood pickaxe -> mine stone -> craft stone tools -> mine coal/iron -> craft iron tools + iron armour.
- Eat cows (u) for food, drink from water (~), sleep when energy is low.
- Collect saplings from grass (DO on grass) and plant them (PLACE_PLANT) for future food.
- Collect wood before descending - it's scarce underground.
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
Dark cavern - you MUST place torches (PLACE_TORCH) to see.  Craft torches from wood + coal.
- Rich in ores: coal, iron, diamonds, sapphires, rubies along cavern edges.
- Bats (B) can be eaten.  Pools of water (~) for drinking.
- Gnomes are stronger than orcs - avoid being surrounded in open spaces.
- PRIORITY: craft diamond sword (1 wood + 2 diamonds) for massive damage, then diamond pickaxe (1 wood + 3 diamonds) to mine sapphires/rubies.
- Craft iron armour (3 iron + 3 coal per piece) or diamond armour (3 diamonds per piece). Mine aggressively here — this is the best floor for diamonds!""",

    3: """\
## Floor 3: Sewers (Current Floor)
Dungeon layout with water patches.  Fill water gaps by placing stone then mining it.
- Lizards swim through water - very dangerous!  Kobolds throw high-damage daggers.
- First chest contains a BOOK - read it (READ_BOOK) to learn fireball or iceball spell.
- Spells cost 2 mana (CAST_FIREBALL / CAST_ICEBALL).  They do fire/ice damage.
- Later enemies resist physical damage - spells become essential!
- ICE enchantment table (E) is on this floor.  Enchant items with sapphires (9 mana + sapphire).
- Enchanting sword/bow: +50% ice damage.  Enchanting armour: -20% ice damage per piece.""",

    4: """\
## Floor 4: Vaults (Current Floor)
Another dungeon level.  Contains a second book and the FIRE enchantment table (E).
- Knights and archers are armoured - physical damage is HALVED against them.
- Use enchanted weapons or spells (fire/ice) to deal effective damage.
- Enchant with rubies at the fire enchantment table (9 mana + ruby).""",

    5: """\
## Floor 5: Troll Mines (Current Floor)
Dark cavern, richest in ores.  Place torches to see.
- PRIORITY: Mine everything - craft full diamond armour if possible.
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
Dark level - place torches.  Frost trolls and ice elementals are the strongest creatures.
- They are immune to ice damage - FIRE damage required.
- No food on this level - ASCEND to Fire Realm to eat.
- Lots of sapphires and rubies in the mountains.
- This is the last floor before the boss!""",

    8: """\
## Floor 8: Graveyard - BOSS LEVEL (Current Floor)
The necromancer (N) is the final boss.  No ladder out.  No food/water, but intrinsics don't decay.
- The necromancer summons waves of enemies from graves (+).
- Kill all enemies in a wave -> necromancer becomes VULNERABLE (N turns yellow).
- Attack the vulnerable necromancer (face + DO) to trigger the next wave.
- Waves correspond to creatures from each floor (floor 0 through floor 7).
- After defeating the final wave (ice realm), attack the necromancer to WIN!""",
}

ACTION_SPACE_TEXT = """\
## Available Actions
Movement: LEFT, RIGHT, UP, DOWN (also turns you to face that direction)
Interact: DO (mine block / attack creature / drink water / eat plant / open chest - in your facing direction)
Rest: SLEEP (restores energy; HP recovers while asleep if food/drink/energy>0; wakes at max energy or if attacked; 2.5x damage while asleep)
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

{MEMORY_INSTRUCTIONS}

REMINDER: Always end your response with ---\\nAction: ACTION_NAME"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Azure GPT agent for Craftax (Markovian + scratchpad)")
    # Model
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--reasoning-effort", type=str, default="medium",
                   choices=["low", "medium", "high"],
                   help="Reasoning effort level for the model")
    # Cost per M tokens (auto-detected from model name, override with flags)
    p.add_argument("--cost-input-per-m", type=float, default=None)
    p.add_argument("--cost-output-per-m", type=float, default=None)
    # Environment
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=9999999)
    p.add_argument("--seed", type=int, default=42)
    # Rendering
    p.add_argument("--view-mode", choices=["local", "full"], default="local")
    p.add_argument("--use-color", action="store_true", default=False)
    # Output
    p.add_argument("--save-gif", action="store_true", default=True)
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--output-dir", type=str, default="results_azure")
    p.add_argument("--render-terminal", action="store_true", default=True)
    p.add_argument("--no-render", action="store_true")
    return p.parse_args()


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def main():
    args = parse_args()
    save_gif = args.save_gif and not args.no_gif
    render_terminal = args.render_terminal and not args.no_render

    # Create timestamped run directory
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    args.output_dir = run_dir
    print(f"Results will be saved to: {run_dir}")

    # --- Load API credentials ---
    load_env()
    endpoint = os.environ.get("AZURE_ENDPOINT", "")
    api_key = os.environ.get("AZURE_API_KEY", "")
    model_name = os.environ.get("AZURE_MODEL_NAME", "")
    if not all([endpoint, api_key, model_name]):
        print("ERROR: AZURE_ENDPOINT, AZURE_API_KEY, AZURE_MODEL_NAME must be set in .env")
        sys.exit(1)

    # --- Detect API type ---
    # Responses API: endpoint contains /responses
    # Chat Completions API: endpoint contains /chat/completions or neither
    use_responses_api = "/responses" in endpoint

    base_endpoint = endpoint.split("/openai/")[0] if "/openai/" in endpoint else endpoint
    api_version = "2025-04-01-preview"
    if "api-version=" in endpoint:
        api_version = endpoint.split("api-version=")[1].split("&")[0]

    # Auto-detect pricing from model name
    if args.cost_input_per_m is None:
        if "pro" in model_name.lower():
            args.cost_input_per_m = 30.00
        else:
            args.cost_input_per_m = 2.50
    if args.cost_output_per_m is None:
        if "pro" in model_name.lower():
            args.cost_output_per_m = 180.00
        else:
            args.cost_output_per_m = 15.00

    api_type = "Responses API" if use_responses_api else "Chat Completions API"
    print(f"Initializing Azure OpenAI client")
    print(f"  Endpoint: {base_endpoint}")
    print(f"  Model: {model_name}")
    print(f"  API type: {api_type}")
    print(f"  API version: {api_version}")
    print(f"  Reasoning effort: {args.reasoning_effort}")

    if use_responses_api:
        import requests as http_requests
        client = None  # use raw HTTP for Responses API
    else:
        client = AzureOpenAI(
            azure_endpoint=base_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    print("Client ready!")

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
        scratchpad = StructuredScratchpad()
        spatial_mem = SpatialMemory()
        recent_actions = []
        pending_query_results = ""
        feedback_msg = ""

        frames = []
        total_reward = 0.0
        achievements_unlocked = set()
        step_times = []
        llm_times = []
        episode_interactions = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0

        for step_i in range(args.max_steps):
            t0 = time.time()

            # --- Skip LLM call while sleeping/resting ---
            if bool(state.is_sleeping) or bool(state.is_resting):
                status = "sleeping" if bool(state.is_sleeping) else "resting"
                rng, _rng = jax.random.split(rng)
                obs, state, reward, done, info = step_fn(_rng, state, 0, params)
                total_reward += float(reward)
                if render_terminal:
                    sys.stdout.write(
                        f"\r  [{status}... HP:{int(state.player_health)} "
                        f"Energy:{int(state.player_energy)} "
                        f"Food:{int(state.player_food)}]  "
                    )
                    sys.stdout.flush()
                if save_gif:
                    pixels = render_pixels_fn(state, BLOCK_PIXEL_SIZE_IMG)
                    frames.append(np.array(pixels, dtype=np.uint8))
                if bool(done):
                    print(f"\n--- Died while {status} at step {step_i+1} ---")
                    break
                continue

            floor_idx = int(state.player_level)

            # --- ASCII observation ---
            ascii_obs, discovered_map = render_craftax_ascii(
                state,
                discovered_map=discovered_map,
                view_mode=args.view_mode,
                include_legend=True,
                use_color=args.use_color,
            )

            # --- Build prompt ---
            system_prompt = build_system_prompt(floor_idx)

            if recent_actions:
                n = len(recent_actions)
                history_lines = "\n".join(
                    f"  {n - idx} turns ago (turn {s}): {a} -> reward {r:+.1f}"
                    for idx, (s, a, r) in enumerate(reversed(recent_actions))
                )
                history_str = f"Recent actions (most recent first):\n{history_lines}"
            else:
                history_str = "Recent actions: (none yet, this is your first turn)"

            mem_parts = []
            if pending_query_results:
                mem_parts.append(pending_query_results)
                pending_query_results = ""
            if spatial_mem.data.get(floor_idx):
                mem_parts.append(
                    f"Saved locations: {spatial_mem.format_floor_summary(floor_idx)}"
                )
            mem_str = "\n".join(mem_parts)

            feedback_prefix = ""
            if feedback_msg:
                feedback_prefix = f"[SYSTEM FEEDBACK] {feedback_msg}\n\n"
                feedback_msg = ""

            user_content = (
                feedback_prefix
                + f"Turn {step_i + 1}. Current observation:\n\n"
                f"{ascii_obs}\n\n"
                f"{history_str}\n\n"
                + (f"{mem_str}\n\n" if mem_str else "")
                + f"Your current scratchpad (sections you don't mention are preserved):\n"
                f"{scratchpad.format_for_prompt()}\n\n"
                f"Write your Assessment and Decision, then end with "
                f"--- delimiter, Action:, and any section updates."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            # --- Azure API call ---
            t_llm = time.time()
            try:
                if use_responses_api:
                    # --- Responses API (for Pro models) ---
                    body = {
                        "model": model_name,
                        "instructions": messages[0]["content"],  # system
                        "input": messages[1]["content"],  # user
                        "max_output_tokens": args.max_tokens,
                    }
                    if args.reasoning_effort != "none":
                        body["reasoning"] = {"effort": args.reasoning_effort}
                    resp = http_requests.post(
                        endpoint,
                        headers={
                            "api-key": api_key,
                            "Content-Type": "application/json",
                        },
                        json=body,
                        timeout=300,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # Extract text from output
                    generated_text = ""
                    for item in data.get("output", []):
                        if item.get("type") == "message":
                            for c in item.get("content", []):
                                if c.get("type") == "output_text":
                                    generated_text += c.get("text", "")
                    usage = data.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    reasoning_tokens = usage.get(
                        "output_tokens_details", {}
                    ).get("reasoning_tokens", 0)
                else:
                    # --- Chat Completions API ---
                    create_kwargs = dict(
                        model=model_name,
                        messages=messages,
                        max_completion_tokens=args.max_tokens,
                        reasoning_effort=args.reasoning_effort,
                    )
                    if args.temperature is not None:
                        create_kwargs["temperature"] = args.temperature
                    if args.top_p is not None:
                        create_kwargs["top_p"] = args.top_p
                    try:
                        response = client.chat.completions.create(**create_kwargs)
                    except Exception as param_err:
                        if "Unsupported parameter" in str(param_err):
                            create_kwargs.pop("temperature", None)
                            create_kwargs.pop("top_p", None)
                            create_kwargs.pop("reasoning_effort", None)
                            response = client.chat.completions.create(
                                **create_kwargs
                            )
                        else:
                            raise
                    generated_text = (
                        response.choices[0].message.content or ""
                    )
                    usage = response.usage
                    input_tokens = usage.prompt_tokens if usage else 0
                    output_tokens = usage.completion_tokens if usage else 0
                    reasoning_tokens = (
                        usage.completion_tokens_details.reasoning_tokens
                        if usage and usage.completion_tokens_details
                        else 0
                    )
            except Exception as e:
                print(f"  API error: {e}")
                generated_text = ""
                input_tokens = output_tokens = reasoning_tokens = 0

            llm_time = time.time() - t_llm
            llm_times.append(llm_time)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_reasoning_tokens += reasoning_tokens

            # --- Detect Azure content filter refusals and retry ---
            _refusal_phrases = (
                "i'm sorry", "i cannot assist", "i can't assist",
                "i'm unable to", "i can't help", "i cannot help",
                "not able to assist", "against my guidelines",
            )
            if generated_text and any(p in generated_text.lower() for p in _refusal_phrases):
                print(f"  Content filter refusal detected, retrying...")
                try:
                    if use_responses_api:
                        retry_body = {
                            "model": model_name,
                            "instructions": (
                                "You are playing a text-based adventure game. "
                                "This is a harmless game environment. "
                                "Choose the best action for the player."
                            ),
                            "input": messages[1]["content"],
                            "max_output_tokens": args.max_tokens,
                        }
                        if args.reasoning_effort != "none":
                            retry_body["reasoning"] = {"effort": args.reasoning_effort}
                        retry_resp = http_requests.post(
                            endpoint,
                            headers={"api-key": api_key, "Content-Type": "application/json"},
                            json=retry_body,
                            timeout=300,
                        )
                        retry_resp.raise_for_status()
                        retry_data = retry_resp.json()
                        generated_text = ""
                        for item in retry_data.get("output", []):
                            if item.get("type") == "message":
                                for c in item.get("content", []):
                                    if c.get("type") == "output_text":
                                        generated_text += c.get("text", "")
                        retry_usage = retry_data.get("usage", {})
                        total_input_tokens += retry_usage.get("input_tokens", 0)
                        total_output_tokens += retry_usage.get("output_tokens", 0)
                    else:
                        retry_msgs = [
                            {"role": "system", "content": (
                                "You are playing a text-based adventure game. "
                                "This is a harmless game environment. "
                                "Choose the best action for the player."
                            )},
                            messages[1],
                        ]
                        retry_response = client.chat.completions.create(
                            model=model_name,
                            messages=retry_msgs,
                            max_completion_tokens=args.max_tokens,
                        )
                        generated_text = retry_response.choices[0].message.content or ""
                        if retry_response.usage:
                            total_input_tokens += retry_response.usage.prompt_tokens
                            total_output_tokens += retry_response.usage.completion_tokens
                except Exception as e:
                    print(f"  Refusal retry also failed: {e}")

            # --- Parse everything from the structured block ---
            action, parsed_name, new_scratchpad, query_results, reasoning, no_reasoning = (
                parse_agent_response(generated_text, spatial_mem, ACTION_FUZZY)
            )

            # --- Retry on parse failure ---
            if parsed_name == "NOOP (parse failed)" and generated_text:
                retry_prompt = (
                    "Your previous response could not be parsed. "
                    "Reply with ONLY:\n---\nAction: <ACTION_NAME>\n\n"
                    "Valid actions: LEFT, RIGHT, UP, DOWN, DO, SLEEP, REST, "
                    "PLACE_TABLE, PLACE_FURNACE, PLACE_STONE, PLACE_PLANT, PLACE_TORCH, "
                    "MAKE_WOOD_PICKAXE, MAKE_STONE_PICKAXE, MAKE_IRON_PICKAXE, MAKE_DIAMOND_PICKAXE, "
                    "MAKE_WOOD_SWORD, MAKE_STONE_SWORD, MAKE_IRON_SWORD, MAKE_DIAMOND_SWORD, "
                    "MAKE_IRON_ARMOUR, MAKE_DIAMOND_ARMOUR, MAKE_ARROW, MAKE_TORCH, "
                    "SHOOT_ARROW, CAST_FIREBALL, CAST_ICEBALL, READ_BOOK, "
                    "ENCHANT_SWORD, ENCHANT_ARMOUR, ENCHANT_BOW, "
                    "DESCEND, ASCEND, NOOP, "
                    "LEVEL_UP_DEXTERITY, LEVEL_UP_STRENGTH, LEVEL_UP_INTELLIGENCE, "
                    "DRINK_POTION_RED, DRINK_POTION_GREEN, DRINK_POTION_BLUE, "
                    "DRINK_POTION_PINK, DRINK_POTION_CYAN, DRINK_POTION_YELLOW"
                )
                try:
                    if use_responses_api:
                        retry_body = {
                            "model": model_name,
                            "instructions": "You are a game agent. Output only a valid action.",
                            "input": retry_prompt,
                            "max_output_tokens": 100,
                        }
                        if args.reasoning_effort != "none":
                            retry_body["reasoning"] = {"effort": "low"}
                        retry_resp = http_requests.post(
                            endpoint,
                            headers={"api-key": api_key, "Content-Type": "application/json"},
                            json=retry_body,
                            timeout=60,
                        )
                        retry_resp.raise_for_status()
                        retry_data = retry_resp.json()
                        retry_text = ""
                        for item in retry_data.get("output", []):
                            if item.get("type") == "message":
                                for c in item.get("content", []):
                                    if c.get("type") == "output_text":
                                        retry_text += c.get("text", "")
                        retry_usage = retry_data.get("usage", {})
                        total_input_tokens += retry_usage.get("input_tokens", 0)
                        total_output_tokens += retry_usage.get("output_tokens", 0)
                    else:
                        retry_response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "You are a game agent. Output only a valid action."},
                                {"role": "user", "content": retry_prompt},
                            ],
                            max_completion_tokens=100,
                        )
                        retry_text = retry_response.choices[0].message.content or ""
                        if retry_response.usage:
                            total_input_tokens += retry_response.usage.prompt_tokens
                            total_output_tokens += retry_response.usage.completion_tokens

                    if retry_text:
                        retry_action, retry_name, _, _, _, _ = parse_agent_response(
                            retry_text, spatial_mem, ACTION_FUZZY
                        )
                        if retry_name != "NOOP (parse failed)":
                            action, parsed_name = retry_action, retry_name
                            print(f"  Retry parsed: {parsed_name}")
                except Exception as retry_err:
                    print(f"  Retry failed: {retry_err}")

                if parsed_name == "NOOP (parse failed)":
                    feedback_msg = (
                        "Your previous response could not be parsed into a valid action. "
                        "Make sure to end your response with:\n---\nAction: <ACTION_NAME>"
                    )
            if not generated_text:
                feedback_msg = "API error occurred. Your action defaulted to NOOP."
            if no_reasoning and not feedback_msg:
                feedback_msg = (
                    "WARNING: Your previous response had no reasoning. "
                    "You MUST write ## Assessment and ## Decision sections "
                    "before the --- delimiter. This helps you make better decisions."
                )
            if query_results:
                pending_query_results = query_results
            if new_scratchpad:
                scratchpad.update_from_text(new_scratchpad)

            # --- Step environment ---
            rng, _rng = jax.random.split(rng)
            obs, state, reward, done, info = step_fn(_rng, state, action, params)
            total_reward += float(reward)
            step_time = time.time() - t0
            step_times.append(step_time)

            # Update recent action history (keep last 5)
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
                "reasoning_tokens": reasoning_tokens,
                "llm_time_s": round(llm_time, 2),
                "action_parsed": parsed_name,
                "action_id": action,
                "reward": float(reward),
                "total_reward": round(total_reward, 4),
                "achievements": len(achievements_unlocked),
                "scratchpad": scratchpad.format_for_prompt()[:1000],
                "spatial_memory_count": sum(
                    len(v) for v in spatial_mem.data.values()
                ),
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

                running_cost = (
                    total_input_tokens * args.cost_input_per_m / 1_000_000
                    + (total_output_tokens)
                    * args.cost_output_per_m
                    / 1_000_000
                )

                clear_screen()
                print(
                    f"=== AZURE AGENT ({model_name}) | "
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
                    f"Azure: {input_tokens} in / {output_tokens} out / "
                    f"{reasoning_tokens} reason / {llm_time:.1f}s / "
                    f"Total: {total_input_tokens+total_output_tokens} tok ({total_reasoning_tokens} reasoning) / ${running_cost:.4f}"
                )
                print()
                print(display_ascii)
                print()
                # Show scratchpad
                print("--- Scratchpad ---")
                print(str(scratchpad))
                # Show spatial memory summary
                mem_total = sum(len(v) for v in spatial_mem.data.values())
                if mem_total > 0:
                    print(f"\n--- Spatial Memory ({mem_total} entries) ---")
                    for f_idx in sorted(spatial_mem.data.keys()):
                        print(
                            f"  {spatial_mem.format_floor_summary(f_idx)}"
                        )
                print()
                if len(reasoning) > 300:
                    reasoning = reasoning[:300] + "..."
                print(f"GPT: {reasoning}")

            # --- Record frame for GIF ---
            if save_gif:
                pixels = render_pixels_fn(state, BLOCK_PIXEL_SIZE_IMG)
                frames.append(np.array(pixels, dtype=np.uint8))

            if bool(done):
                print(
                    f"\n--- Episode {ep+1} ended at step {step_i+1} (game over) ---"
                )
                break

        # --- Episode metrics ---
        avg_step_time = np.mean(step_times) if step_times else 0
        avg_llm_time = np.mean(llm_times) if llm_times else 0
        cost_input = total_input_tokens * args.cost_input_per_m / 1_000_000
        cost_output = (
            (total_output_tokens)
            * args.cost_output_per_m
            / 1_000_000
        )
        cost_total = cost_input + cost_output
        metrics = {
            "episode": ep + 1,
            "steps": step_i + 1,
            "total_reward": round(total_reward, 4),
            "achievements": len(achievements_unlocked),
            "avg_step_time_s": round(avg_step_time, 2),
            "avg_llm_time_s": round(avg_llm_time, 2),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_reasoning_tokens": total_reasoning_tokens,
            "cost_usd": round(cost_total, 4),
            "model": model_name,
            "view_mode": args.view_mode,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "temperature": args.temperature,
        }
        all_episode_metrics.append(metrics)

        print(f"\n{'='*60}")
        print(f"Episode {ep+1} Summary:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(
            f"  cost_breakdown: ${cost_input:.4f} in + ${cost_output:.4f} out = ${cost_total:.4f}"
        )
        print(f"{'='*60}")

        if save_gif and frames:
            gif_path = os.path.join(args.output_dir, f"azure_ep{ep+1}.gif")
            imageio.mimsave(gif_path, frames, fps=args.gif_fps, loop=0)
            print(f"  GIF saved: {gif_path} ({len(frames)} frames)")

        if spatial_mem.data:
            mem_path = os.path.join(
                args.output_dir, f"spatial_memory_ep{ep+1}.json"
            )
            spatial_mem.save(mem_path)
            print(f"  Spatial memory saved: {mem_path}")

        all_interactions.append({
            "episode": ep + 1,
            "config": {
                "model": model_name,
                "view_mode": args.view_mode,
                "max_tokens": args.max_tokens,
                "reasoning_effort": args.reasoning_effort,
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

    print(f"\n{'='*60}")
    print(f"AZURE AGENT SUMMARY ({model_name})")
    print(f"{'='*60}")
    rewards = [m["total_reward"] for m in all_episode_metrics]
    achievs = [m["achievements"] for m in all_episode_metrics]
    print(f"  Episodes: {len(all_episode_metrics)}")
    print(f"  Avg reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
    print(f"  Avg achievements: {np.mean(achievs):.1f} +/- {np.std(achievs):.1f}")
    print(f"  Best reward: {max(rewards):.4f}")
    print(f"  Best achievements: {max(achievs)}")
    if llm_times:
        print(f"  Avg Azure time: {np.mean(llm_times):.2f}s/step")
    total_cost = sum(m["cost_usd"] for m in all_episode_metrics)
    print(f"  Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
