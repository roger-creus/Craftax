#!/usr/bin/env python3
"""Gemini LLM agent for Craftax using the Google GenAI SDK.

Each step is Markovian (full state observation) but the agent maintains an
external scratchpad/memory that it can read and rewrite each turn to track
goals, plans, spatial notes and lessons learned.

Includes rate-limit safety for Gemini free tier (RPM, TPM, RPD).

Requires GEMINI_API_KEY in .env or as environment variable.

Usage:
    python examples/run_gemini_agent.py
    python examples/run_gemini_agent.py --model gemini-2.5-flash --max-steps 500
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

from google import genai
from google.genai import types

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.renderer import render_craftax_pixels
from craftax.craftax.ascii_renderer import render_craftax_ascii, create_discovered_map
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_IMG, Action


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
# Rate limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """Simple token-bucket rate limiter for RPM, TPM, and RPD."""

    def __init__(self, rpm=4000, tpm=4_000_000, rpd=150_000):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self._request_times = []  # timestamps of requests in current minute
        self._token_counts = []   # (timestamp, tokens) in current minute
        self._daily_requests = 0
        self._day_start = time.time()

    def wait_if_needed(self, estimated_tokens=5000):
        """Block until we can safely make another request."""
        now = time.time()

        # Reset daily counter if 24h passed
        if now - self._day_start > 86400:
            self._daily_requests = 0
            self._day_start = now

        # Check RPD
        if self._daily_requests >= self.rpd:
            print(f"  [RATE LIMIT] Daily request limit ({self.rpd}) reached. Stopping.")
            sys.exit(1)

        # Prune entries older than 60s
        cutoff = now - 60
        self._request_times = [t for t in self._request_times if t > cutoff]
        self._token_counts = [(t, n) for t, n in self._token_counts if t > cutoff]

        # Check RPM
        while len(self._request_times) >= self.rpm:
            wait = self._request_times[0] - cutoff + 0.5
            if wait > 0:
                print(f"  [RATE LIMIT] RPM limit, waiting {wait:.1f}s...")
                time.sleep(wait)
            now = time.time()
            cutoff = now - 60
            self._request_times = [t for t in self._request_times if t > cutoff]

        # Check TPM
        tokens_in_window = sum(n for _, n in self._token_counts)
        while tokens_in_window + estimated_tokens > self.tpm:
            wait = self._token_counts[0][0] - cutoff + 0.5
            if wait > 0:
                print(f"  [RATE LIMIT] TPM limit, waiting {wait:.1f}s...")
                time.sleep(wait)
            now = time.time()
            cutoff = now - 60
            self._token_counts = [(t, n) for t, n in self._token_counts if t > cutoff]
            tokens_in_window = sum(n for _, n in self._token_counts)

    def record(self, total_tokens):
        """Record a completed request."""
        now = time.time()
        self._request_times.append(now)
        self._token_counts.append((now, total_tokens))
        self._daily_requests += 1


from agent_memory import (
    SpatialMemory,
    parse_agent_response,
    EMPTY_SCRATCHPAD,
    MEMORY_INSTRUCTIONS,
)


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
- Found in chests. 6 colours, effects randomized each game (+-health, +-mana, +-energy).
- Experiment carefully!

## Day/Night Cycle
- The overworld has a day/night cycle shown in [Status] as Day, Dusk, or Night.
- At night, more hostile mobs spawn on the overworld. Be cautious or sleep through it.
- Underground floors are always dark regardless of time — you need torches (PLACE_TORCH).
- Torches are crafted from 1 wood + 1 coal (MAKE_TORCH).

## Combat Tips
- Block yourself in with stone (PLACE_STONE) before sleeping - you are vulnerable while asleep.
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
- PRIORITY: Mine trees -> place crafting table -> craft wood pickaxe -> mine stone -> craft stone tools -> mine coal/iron -> craft iron tools.
- Eat cows (u) for food, drink from water (~), sleep when energy is low.
- Collect saplings from grass (DO on grass) and plant them (PLACE_PLANT) for future food.
- Collect lots of wood before descending - it's scarce underground.
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
- Diamonds: craft diamond sword (2) or pickaxe (3).  Diamond pickaxe mines sapphires/rubies.
- Craft iron armour (3 iron + 3 coal per piece) or diamond armour (3 diamonds per piece).""",

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
    p = argparse.ArgumentParser(description="Gemini agent for Craftax (Markovian + scratchpad)")
    # Model
    p.add_argument("--model", type=str, default="gemini-3.1-flash-lite-preview")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--thinking-budget", type=int, default=8192,
                   help="Token budget for Gemini thinking (0 to disable)")
    # Rate limits (Gemini 3.1 Flash Lite free tier)
    p.add_argument("--rpm", type=int, default=4000, help="Requests per minute limit")
    p.add_argument("--tpm", type=int, default=4_000_000, help="Tokens per minute limit")
    p.add_argument("--rpd", type=int, default=150_000, help="Requests per day limit")
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
    p.add_argument("--output-dir", type=str, default="results_gemini")
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

    # Create timestamped run directory to avoid overwriting previous results
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    args.output_dir = run_dir
    print(f"Results will be saved to: {run_dir}")

    # --- Load API key ---
    load_env()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found. Set it in .env or environment.")
        sys.exit(1)

    # --- Init Gemini client + rate limiter ---
    print(f"Initializing Gemini client with model: {args.model}")
    client = genai.Client(api_key=api_key)
    rate_limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm, rpd=args.rpd)

    gen_config_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_output_tokens": args.max_tokens,
    }
    if args.thinking_budget > 0:
        gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=args.thinking_budget
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
        scratchpad = EMPTY_SCRATCHPAD
        spatial_mem = SpatialMemory()
        recent_actions = []  # last 5 (step, action_name, reward)
        pending_query_results = ""  # MAP_GET results from previous turn
        feedback_msg = ""  # error/parse feedback for next turn

        frames = []
        total_reward = 0.0
        achievements_unlocked = set()
        step_times = []
        llm_times = []
        episode_interactions = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_thinking_tokens = 0

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
            # Memory context: pending query results + floor summary
            mem_parts = []
            if pending_query_results:
                mem_parts.append(pending_query_results)
                pending_query_results = ""
            floor_summary = spatial_mem.format_floor_summary(floor_idx)
            if spatial_mem.data.get(floor_idx):
                mem_parts.append(f"Saved locations: {floor_summary}")
            mem_str = "\n".join(mem_parts)

            # Build feedback prefix (parse errors from previous turn)
            feedback_prefix = ""
            if feedback_msg:
                feedback_prefix = f"[SYSTEM FEEDBACK] {feedback_msg}\n\n"
                feedback_msg = ""

            user_msg = (
                feedback_prefix
                + f"Turn {step_i + 1}. Current observation:\n\n"
                f"{ascii_obs}\n\n"
                f"{history_str}\n\n"
                + (f"{mem_str}\n\n" if mem_str else "")
                + f"Your current scratchpad:\n{scratchpad}\n\n"
                f"Analyze the situation, choose your next action, "
                f"and use MAP_ADD/MAP_GET/MAP_REMOVE commands as needed.\n"
                f"Remember: end with --- delimiter, then Action: and optional MAP_*/Scratchpad."
            )

            # --- Rate limit check ---
            rate_limiter.wait_if_needed(estimated_tokens=5000)

            # --- Gemini API call ---
            t_llm = time.time()
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        **{k: v for k, v in gen_config_kwargs.items()},
                    ),
                )
                generated_text = response.text or ""
                usage = response.usage_metadata
                input_tokens = usage.prompt_token_count if usage else 0
                output_tokens = usage.candidates_token_count if usage else 0
                thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
                total_tokens = input_tokens + output_tokens + thinking_tokens
                rate_limiter.record(total_tokens)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print(f"  [RATE LIMIT] 429 from API, backing off 30s...")
                    time.sleep(30)
                    # Retry once
                    try:
                        response = client.models.generate_content(
                            model=args.model,
                            contents=user_msg,
                            config=types.GenerateContentConfig(
                                system_instruction=system_prompt,
                                **{k: v for k, v in gen_config_kwargs.items()},
                            ),
                        )
                        generated_text = response.text or ""
                        usage = response.usage_metadata
                        input_tokens = usage.prompt_token_count if usage else 0
                        output_tokens = usage.candidates_token_count if usage else 0
                        thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
                        rate_limiter.record(input_tokens + output_tokens + thinking_tokens)
                    except Exception as e2:
                        print(f"  API error (retry): {e2}")
                        generated_text = ""
                        input_tokens = output_tokens = thinking_tokens = 0
                else:
                    print(f"  API error: {e}")
                    generated_text = ""
                    input_tokens = output_tokens = thinking_tokens = 0

            llm_time = time.time() - t_llm
            llm_times.append(llm_time)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_thinking_tokens += thinking_tokens

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
            if not generated_text:
                feedback_msg = (
                    "API error occurred on the previous turn. Your action defaulted to NOOP."
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
                "thinking_tokens": thinking_tokens,
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
                    f"=== GEMINI AGENT ({args.model}) | "
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
                running_cost = (
                    total_input_tokens * 0.25 / 1_000_000
                    + (total_output_tokens + total_thinking_tokens) * 1.50 / 1_000_000
                )
                print(
                    f"Gemini: {input_tokens} in / {output_tokens} out / "
                    f"{thinking_tokens} think / {llm_time:.1f}s / "
                    f"Total: {total_input_tokens+total_output_tokens+total_thinking_tokens} tok / ${running_cost:.4f}"
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
                reasoning = reasoning
                if len(reasoning) > 300:
                    reasoning = reasoning[:300] + "..."
                print(f"Gemini: {reasoning}")

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
        # Cost: $0.25/M input, $1.50/M output (thinking counts as output)
        cost_input = total_input_tokens * 0.25 / 1_000_000
        cost_output = (total_output_tokens + total_thinking_tokens) * 1.50 / 1_000_000
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
            "total_thinking_tokens": total_thinking_tokens,
            "cost_usd": round(cost_total, 4),
            "model": args.model,
            "view_mode": args.view_mode,
            "max_tokens": args.max_tokens,
            "thinking_budget": args.thinking_budget,
            "temperature": args.temperature,
        }
        all_episode_metrics.append(metrics)

        print(f"\n{'='*60}")
        print(f"Episode {ep+1} Summary:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"  cost_breakdown: ${cost_input:.4f} in + ${cost_output:.4f} out = ${cost_total:.4f}")
        print(f"{'='*60}")

        if save_gif and frames:
            gif_path = os.path.join(args.output_dir, f"gemini_ep{ep+1}.gif")
            imageio.mimsave(gif_path, frames, fps=args.gif_fps, loop=0)
            print(f"  GIF saved: {gif_path} ({len(frames)} frames)")

        # Save spatial memory
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
                "thinking_budget": args.thinking_budget,
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
    print(f"GEMINI AGENT SUMMARY ({args.model})")
    print(f"{'='*60}")
    rewards = [m["total_reward"] for m in all_episode_metrics]
    achievs = [m["achievements"] for m in all_episode_metrics]
    print(f"  Episodes: {len(all_episode_metrics)}")
    print(f"  Avg reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
    print(f"  Avg achievements: {np.mean(achievs):.1f} +/- {np.std(achievs):.1f}")
    print(f"  Best reward: {max(rewards):.4f}")
    print(f"  Best achievements: {max(achievs)}")
    if llm_times:
        print(f"  Avg Gemini time: {np.mean(llm_times):.2f}s/step")


if __name__ == "__main__":
    main()
