"""Stable GlyphBench-facing API for the Craftax fork.

This module intentionally stays importable without JAX. It exposes the action
names and Unicode character mapping used by GlyphBench, while resolving Craftax
runtime enums and renderer functions lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ACTION_NAMES: tuple[str, ...] = (
    "NOOP",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "MOVE_UP",
    "MOVE_DOWN",
    "DO",
    "SLEEP",
    "PLACE_STONE",
    "PLACE_TABLE",
    "PLACE_FURNACE",
    "PLACE_PLANT",
    "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE",
    "MAKE_IRON_PICKAXE",
    "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD",
    "MAKE_IRON_SWORD",
    "REST",
    "DESCEND",
    "ASCEND",
    "MAKE_DIAMOND_PICKAXE",
    "MAKE_DIAMOND_SWORD",
    "MAKE_IRON_ARMOR",
    "MAKE_DIAMOND_ARMOR",
    "SHOOT_ARROW",
    "MAKE_ARROW",
    "CAST_FIREBALL",
    "CAST_ICEBALL",
    "PLACE_TORCH",
    "DRINK_POTION_RED",
    "DRINK_POTION_GREEN",
    "DRINK_POTION_BLUE",
    "DRINK_POTION_PINK",
    "DRINK_POTION_CYAN",
    "DRINK_POTION_YELLOW",
    "READ_BOOK",
    "ENCHANT_SWORD",
    "ENCHANT_ARMOR",
    "MAKE_TORCH",
    "LEVEL_UP_DEXTERITY",
    "LEVEL_UP_STRENGTH",
    "LEVEL_UP_INTELLIGENCE",
    "ENCHANT_BOW",
)

ACTION_DESCRIPTIONS: tuple[str, ...] = (
    "do nothing this turn",
    "move or face one cell left",
    "move or face one cell right",
    "move or face one cell up",
    "move or face one cell down",
    "interact with the cell you face (chop, mine, drink, attack, open, eat)",
    "sleep to restore energy and health while food and drink remain",
    "place a stone block in the cell you face",
    "place a crafting table in the cell you face",
    "place a furnace in the cell you face",
    "plant a sapling in the cell you face",
    "craft a wood pickaxe near a crafting table",
    "craft a stone pickaxe near a crafting table",
    "craft an iron pickaxe near a crafting table and furnace",
    "craft a wood sword near a crafting table",
    "craft a stone sword near a crafting table",
    "craft an iron sword near a crafting table and furnace",
    "wait in place until healed, attacked, or an intrinsic is depleted",
    "descend the ladder under you to the next floor",
    "ascend the ladder under you to the previous floor",
    "craft a diamond pickaxe near the required station",
    "craft a diamond sword near the required station",
    "craft one iron armor piece near a table and furnace",
    "craft one diamond armor piece near the required station",
    "shoot an arrow in your facing direction",
    "craft arrows near a crafting table",
    "cast a fireball in your facing direction",
    "cast an iceball in your facing direction",
    "place a torch in the cell you face",
    "drink a red potion",
    "drink a green potion",
    "drink a blue potion",
    "drink a pink potion",
    "drink a cyan potion",
    "drink a yellow potion",
    "read a book to learn a spell",
    "enchant your sword near the matching enchantment table",
    "enchant one armor piece near the matching enchantment table",
    "craft torches from wood and coal",
    "spend XP to raise dexterity",
    "spend XP to raise strength",
    "spend XP to raise intelligence",
    "enchant your bow near the matching enchantment table",
)

ACTION_ALIASES: dict[str, str] = {
    "LEFT": "MOVE_LEFT",
    "RIGHT": "MOVE_RIGHT",
    "UP": "MOVE_UP",
    "DOWN": "MOVE_DOWN",
    "MAKE_IRON_ARMOUR": "MAKE_IRON_ARMOR",
    "MAKE_DIAMOND_ARMOUR": "MAKE_DIAMOND_ARMOR",
    "ENCHANT_ARMOUR": "ENCHANT_ARMOR",
    "ENCHANT_WEAPON": "ENCHANT_SWORD",
}

UPSTREAM_ACTION_NAMES: tuple[str, ...] = (
    "NOOP",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "DO",
    "SLEEP",
    "PLACE_STONE",
    "PLACE_TABLE",
    "PLACE_FURNACE",
    "PLACE_PLANT",
    "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE",
    "MAKE_IRON_PICKAXE",
    "MAKE_WOOD_SWORD",
    "MAKE_STONE_SWORD",
    "MAKE_IRON_SWORD",
    "REST",
    "DESCEND",
    "ASCEND",
    "MAKE_DIAMOND_PICKAXE",
    "MAKE_DIAMOND_SWORD",
    "MAKE_IRON_ARMOUR",
    "MAKE_DIAMOND_ARMOUR",
    "SHOOT_ARROW",
    "MAKE_ARROW",
    "CAST_FIREBALL",
    "CAST_ICEBALL",
    "PLACE_TORCH",
    "DRINK_POTION_RED",
    "DRINK_POTION_GREEN",
    "DRINK_POTION_BLUE",
    "DRINK_POTION_PINK",
    "DRINK_POTION_CYAN",
    "DRINK_POTION_YELLOW",
    "READ_BOOK",
    "ENCHANT_SWORD",
    "ENCHANT_ARMOUR",
    "MAKE_TORCH",
    "LEVEL_UP_DEXTERITY",
    "LEVEL_UP_STRENGTH",
    "LEVEL_UP_INTELLIGENCE",
    "ENCHANT_BOW",
)

# Fork renderer char IDs mapped to the closest GlyphBench/Craftax glyphs.
# Values must stay single-codepoint: the GlyphBench observation contract checks this.
UNICODE_CHAR_TABLE: tuple[str, ...] = (
    " ",  # 0 space / unseen darkness
    "·",  # 1 grass/path
    "≈",  # 2 water
    "S",  # 3 stone
    "♣",  # 4 tree
    "=",  # 5 wood / placed block
    "░",  # 6 sand
    "♨",  # 7 lava
    "+",  # 8 young plant
    "*",  # 9 ripe plant
    "█",  # 10 wall
    "^",  # 11 stalagmite
    "C",  # 12 coal
    "I",  # 13 iron
    "D",  # 14 diamond
    "♦",  # 15 sapphire
    "▲",  # 16 ruby
    "t",  # 17 crafting table
    "f",  # 18 furnace
    "$",  # 19 chest
    "⊙",  # 20 fountain
    ";",  # 21 fire/ice grass
    ":",  # 22 gravel
    "❄",  # 23 ice shrub
    "Ⓔ",  # 24 enchantment table
    "N",  # 25 necromancer
    "⚰",  # 26 grave marker
    "?",  # 27 invalid/unknown
    "@",  # 28 you, no direction
    "*",  # 29 torch / projectile impact
    "⇣",  # 30 ladder down
    "⇡",  # 31 ladder up
    "-",  # 32 arrow projectile
    "†",  # 33 dagger projectile
    "z",  # 34 zombie
    "G",  # 35 gnome warrior
    "O",  # 36 orc soldier
    "L",  # 37 lizard
    "K",  # 38 knight
    "T",  # 39 troll
    "p",  # 40 pigman
    "r",  # 41 frost troll
    "c",  # 42 cow
    "b",  # 43 bat
    "s",  # 44 snail
    "k",  # 45 skeleton
    "g",  # 46 gnome archer
    "W",  # 47 orc mage
    "q",  # 48 kobold
    "a",  # 49 archer
    "d",  # 50 deep thing
    "F",  # 51 fire elemental
    "i",  # 52 ice elemental
    "_",  # 53 dim/fog placeholder
    "←",  # 54 you, facing left
    "→",  # 55 you, facing right
    "↑",  # 56 you, facing up
    "↓",  # 57 you, facing down
    "█",  # 58 map edge / out of bounds
)

CHAR_ID_MEANING: tuple[str, ...] = (
    "unseen or dark tile",
    "grass or path",
    "water",
    "stone",
    "tree",
    "wood or placed block",
    "sand",
    "lava",
    "young plant",
    "ripe plant or projectile effect",
    "wall",
    "stalagmite",
    "coal ore",
    "iron ore",
    "diamond ore",
    "sapphire ore",
    "ruby ore",
    "crafting table",
    "furnace",
    "chest",
    "fountain",
    "fire or ice grass",
    "gravel",
    "ice shrub",
    "enchantment table",
    "necromancer",
    "grave marker",
    "invalid or unknown tile",
    "you, no facing direction",
    "torch or projectile effect",
    "ladder down",
    "ladder up",
    "arrow projectile",
    "dagger projectile",
    "zombie",
    "gnome warrior",
    "orc soldier",
    "lizard",
    "knight",
    "troll",
    "pigman",
    "frost troll",
    "cow",
    "bat",
    "snail",
    "skeleton",
    "gnome archer",
    "orc mage",
    "kobold",
    "archer",
    "deep thing",
    "fire elemental",
    "ice elemental",
    "fogged discovered tile",
    "you, facing left",
    "you, facing right",
    "you, facing up",
    "you, facing down",
    "map edge or out of bounds",
)


@dataclass(frozen=True)
class UnicodeCraftaxRender:
    grid: str
    legend: str
    char_grid: Any
    color_grid: Any
    status_data: Any


def action_values() -> tuple[int, ...]:
    """Return Craftax ``Action`` enum values aligned with ``ACTION_NAMES``."""
    from craftax.craftax.constants import Action

    return tuple(int(getattr(Action, name).value) for name in UPSTREAM_ACTION_NAMES)


def unicode_grid_to_string(char_grid: Any) -> str:
    """Render a Craftax char-ID grid with the stable Unicode mapping."""
    char_np = np.asarray(char_grid, dtype=np.int32)
    rows: list[str] = []
    for row in char_np:
        rows.append("".join(UNICODE_CHAR_TABLE[int(ch)] for ch in row))
    return "\n".join(rows)


def unicode_legend_from_char_grid(char_grid: Any) -> str:
    """Build a compact legend for the visible Unicode symbols."""
    char_np = np.asarray(char_grid, dtype=np.int32)
    meanings_by_symbol: dict[str, list[str]] = {}
    for char_id in sorted(int(x) for x in np.unique(char_np)):
        sym = UNICODE_CHAR_TABLE[char_id]
        meaning = CHAR_ID_MEANING[char_id]
        if sym.isspace():
            continue
        meanings_by_symbol.setdefault(sym, [])
        if meaning not in meanings_by_symbol[sym]:
            meanings_by_symbol[sym].append(meaning)

    return "\n".join(
        f"{sym} — {' / '.join(meanings_by_symbol[sym])}"
        for sym in sorted(meanings_by_symbol)
    )


def render_craftax_unicode_grid_local(state: Any) -> UnicodeCraftaxRender:
    """Render the local 9x11 Craftax view as stable Unicode glyphs."""
    from craftax.craftax import ascii_renderer

    char_grid, color_grid, status_data = ascii_renderer.render_craftax_ascii_grid_local(
        state
    )
    return UnicodeCraftaxRender(
        grid=unicode_grid_to_string(char_grid),
        legend=unicode_legend_from_char_grid(char_grid),
        char_grid=char_grid,
        color_grid=color_grid,
        status_data=status_data,
    )


assert len(ACTION_NAMES) == len(ACTION_DESCRIPTIONS) == len(UPSTREAM_ACTION_NAMES)
assert len(UNICODE_CHAR_TABLE) == len(CHAR_ID_MEANING) == 59
assert all(len(ch) == 1 for ch in UNICODE_CHAR_TABLE)
