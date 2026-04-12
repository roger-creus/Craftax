"""NetHack-style ASCII renderer for Craftax.

Architecture:
  - render_craftax_ascii_grid(state): JIT-compiled JAX function returning
    integer char_grid, color_grid, and status_data arrays.
  - ascii_grid_to_string(...): Pure-Python formatter converting those arrays
    to an ANSI-colored string for display or LLM consumption.
  - render_craftax_ascii(state, ...): Convenience wrapper calling both.
"""

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax.constants import (
    BlockType,
    ItemType,
    OBS_DIM,
    MAX_OBS_DIM,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
)
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.util.game_logic_utils import (
    is_boss_vulnerable,
    get_max_health,
    get_max_food,
    get_max_drink,
    get_max_energy,
    get_max_mana,
)

# ---------------------------------------------------------------------------
# Character ID table  (JAX operates on integer IDs; Python maps IDs to chars)
# ---------------------------------------------------------------------------
# fmt: off
# ID:  0     1     2     3     4     5     6     7     8     9
#      space dot   tilde pct   T     eq    comma amp   p     P
# ID: 10    11    12    13    14    15    16    17    18    19
#      hash  caret c     i     d     s     r     C     F     dollar
# ID: 20    21    22    23    24    25    26    27    28    29
#      lbrace semi  colon t_lc  E     N     plus  qmark at    star
# ID: 30    31    32    33    34    35    36    37    38    39
#      gt    lt    dash  slash Z     G     O     L     K     R
# ID: 40    41    42    43    44    45    46    47    48    49
#      H     Y     u_lc  B     n_lc  S     g_lc  W     k_lc  A
# ID: 50    51    52    53
#      D     X     J     underscore(dim placeholder)

CHAR_TABLE = list(
    " .~%T=,&pP#^cidsrCF${;:tEN+?@*><-/ZGOLKRHYuBnSgWkADXJ_"
) + ["\u2190", "\u2192", "\u2191", "\u2193", "\u2588"]
# 54 base + 4 arrows (← → ↑ ↓) + 1 solid block (█ for out-of-bounds)
assert len(CHAR_TABLE) == 59, f"CHAR_TABLE has {len(CHAR_TABLE)} entries"

# Character IDs (constants for indexing)
C_SPACE = 0
C_DOT = 1
C_TILDE = 2
C_PCT = 3
C_T = 4
C_EQ = 5
C_COMMA = 6
C_AMP = 7
C_p = 8
C_P = 9
C_HASH = 10
C_CARET = 11
C_c = 12
C_i = 13
C_d = 14
C_s = 15
C_r = 16
C_C = 17
C_F = 18
C_DOLLAR = 19
C_LBRACE = 20
C_SEMI = 21
C_COLON = 22
C_t = 23
C_E = 24
C_N = 25
C_PLUS = 26
C_QMARK = 27
C_AT = 28
C_STAR = 29
C_GT = 30
C_LT = 31
C_DASH = 32
C_SLASH = 33
C_Z = 34
C_G = 35
C_O = 36
C_L = 37
C_K = 38
C_R = 39
C_H = 40
C_Y = 41
C_u = 42
C_B = 43
C_n = 44
C_S = 45
C_g = 46
C_W = 47
C_k = 48
C_A = 49
C_D = 50
C_X = 51
C_J = 52
C_UNDER = 53  # dim/fog placeholder
C_ARROW_LEFT = 54   # ←  player facing left
C_ARROW_RIGHT = 55  # →  player facing right
C_ARROW_UP = 56     # ↑  player facing up
C_ARROW_DOWN = 57   # ↓  player facing down
C_OOB = 58          # █  out of bounds

# ---------------------------------------------------------------------------
# ANSI color ID table  (16-color SGR codes)
# ---------------------------------------------------------------------------
# ID -> ANSI SGR code string
# 0 = no color (reset)
ANSI_TABLE = [
    "",           # 0: no color
    "\033[32m",   # 1: green
    "\033[1;34m", # 2: bright blue  (note: using bold+blue)
    "\033[37m",   # 3: white
    "\033[1;32m", # 4: bright green
    "\033[33m",   # 5: yellow
    "\033[1;33m", # 6: bright yellow
    "\033[1;31m", # 7: bright red
    "\033[90m",   # 8: dark gray
    "\033[1;36m", # 9: bright cyan
    "\033[31m",   # 10: red
    "\033[1;37m", # 11: bright white
    "\033[1;35m", # 12: bright magenta
    "\033[2m",    # 13: dim (for fog of war)
    "\033[36m",   # 14: cyan
    "\033[34m",   # 15: blue
    "\033[35m",   # 16: magenta
]
ANSI_RESET = "\033[0m"

# Color IDs (constants)
CLR_NONE = 0
CLR_GREEN = 1
CLR_BRIGHT_BLUE = 2
CLR_WHITE = 3
CLR_BRIGHT_GREEN = 4
CLR_YELLOW = 5
CLR_BRIGHT_YELLOW = 6
CLR_BRIGHT_RED = 7
CLR_DARK_GRAY = 8
CLR_BRIGHT_CYAN = 9
CLR_RED = 10
CLR_BRIGHT_WHITE = 11
CLR_BRIGHT_MAGENTA = 12
CLR_DIM = 13
CLR_CYAN = 14
CLR_BLUE = 15
CLR_MAGENTA = 16

# ---------------------------------------------------------------------------
# Block type -> (char_id, color_id)  [indexed by BlockType.value 0-36]
# ---------------------------------------------------------------------------
BLOCK_CHAR_IDS = jnp.array([
    C_QMARK,   # 0  INVALID
    C_OOB,     # 1  OUT_OF_BOUNDS
    C_DOT,     # 2  GRASS
    C_TILDE,   # 3  WATER
    C_PCT,     # 4  STONE
    C_T,       # 5  TREE
    C_EQ,      # 6  WOOD
    C_DOT,     # 7  PATH
    C_c,       # 8  COAL
    C_i,       # 9  IRON
    C_d,       # 10 DIAMOND
    C_C,       # 11 CRAFTING_TABLE
    C_F,       # 12 FURNACE
    C_COMMA,   # 13 SAND
    C_AMP,     # 14 LAVA
    C_p,       # 15 PLANT
    C_P,       # 16 RIPE_PLANT
    C_HASH,    # 17 WALL
    C_SPACE,   # 18 DARKNESS
    C_HASH,    # 19 WALL_MOSS
    C_CARET,   # 20 STALAGMITE
    C_s,       # 21 SAPPHIRE
    C_r,       # 22 RUBY
    C_DOLLAR,  # 23 CHEST
    C_LBRACE,  # 24 FOUNTAIN
    C_SEMI,    # 25 FIRE_GRASS
    C_SEMI,    # 26 ICE_GRASS
    C_COLON,   # 27 GRAVEL
    C_T,       # 28 FIRE_TREE
    C_t,       # 29 ICE_SHRUB
    C_E,       # 30 ENCHANTMENT_TABLE_FIRE
    C_E,       # 31 ENCHANTMENT_TABLE_ICE
    C_N,       # 32 NECROMANCER
    C_PLUS,    # 33 GRAVE
    C_PLUS,    # 34 GRAVE2
    C_PLUS,    # 35 GRAVE3
    C_N,       # 36 NECROMANCER_VULNERABLE
], dtype=jnp.int32)

BLOCK_COLOR_IDS = jnp.array([
    CLR_BRIGHT_RED,     # 0  INVALID
    CLR_DARK_GRAY,      # 1  OUT_OF_BOUNDS
    CLR_GREEN,          # 2  GRASS
    CLR_BRIGHT_BLUE,    # 3  WATER
    CLR_WHITE,          # 4  STONE
    CLR_BRIGHT_GREEN,   # 5  TREE
    CLR_YELLOW,         # 6  WOOD
    CLR_YELLOW,         # 7  PATH
    CLR_DARK_GRAY,      # 8  COAL
    CLR_BRIGHT_WHITE,   # 9  IRON
    CLR_BRIGHT_CYAN,    # 10 DIAMOND
    CLR_YELLOW,         # 11 CRAFTING_TABLE
    CLR_BRIGHT_RED,     # 12 FURNACE
    CLR_BRIGHT_YELLOW,  # 13 SAND
    CLR_BRIGHT_RED,     # 14 LAVA
    CLR_GREEN,          # 15 PLANT
    CLR_BRIGHT_GREEN,   # 16 RIPE_PLANT
    CLR_WHITE,          # 17 WALL
    CLR_NONE,           # 18 DARKNESS
    CLR_GREEN,          # 19 WALL_MOSS
    CLR_WHITE,          # 20 STALAGMITE
    CLR_BRIGHT_BLUE,    # 21 SAPPHIRE
    CLR_BRIGHT_RED,     # 22 RUBY
    CLR_BRIGHT_YELLOW,  # 23 CHEST
    CLR_BRIGHT_CYAN,    # 24 FOUNTAIN
    CLR_RED,            # 25 FIRE_GRASS
    CLR_BRIGHT_CYAN,    # 26 ICE_GRASS
    CLR_WHITE,          # 27 GRAVEL
    CLR_BRIGHT_RED,     # 28 FIRE_TREE
    CLR_BRIGHT_CYAN,    # 29 ICE_SHRUB
    CLR_BRIGHT_RED,     # 30 ENCHANTMENT_TABLE_FIRE
    CLR_BRIGHT_CYAN,    # 31 ENCHANTMENT_TABLE_ICE
    CLR_BRIGHT_MAGENTA, # 32 NECROMANCER
    CLR_WHITE,          # 33 GRAVE
    CLR_DARK_GRAY,      # 34 GRAVE2
    CLR_WHITE,          # 35 GRAVE3
    CLR_BRIGHT_YELLOW,  # 36 NECROMANCER_VULNERABLE
], dtype=jnp.int32)

# ---------------------------------------------------------------------------
# Item type -> (char_id, color_id)  [indexed by ItemType.value 0-4]
# ---------------------------------------------------------------------------
ITEM_CHAR_IDS = jnp.array([
    C_SPACE,  # 0 NONE (should never be used as overlay)
    C_STAR,   # 1 TORCH
    C_GT,     # 2 LADDER_DOWN
    C_LT,     # 3 LADDER_UP
    C_GT,     # 4 LADDER_DOWN_BLOCKED
], dtype=jnp.int32)

ITEM_COLOR_IDS = jnp.array([
    CLR_NONE,          # 0 NONE
    CLR_BRIGHT_YELLOW, # 1 TORCH
    CLR_BRIGHT_WHITE,  # 2 LADDER_DOWN
    CLR_BRIGHT_WHITE,  # 3 LADDER_UP
    CLR_RED,           # 4 LADDER_DOWN_BLOCKED
], dtype=jnp.int32)

# ---------------------------------------------------------------------------
# Mob type -> (char_id, color_id)  [indexed by (class, type_id)]
# Shape: (5 classes, 8 types)
# Classes: 0=melee, 1=passive, 2=ranged, 3=mob_projectile, 4=player_projectile
# ---------------------------------------------------------------------------
MOB_CHAR_IDS = jnp.array([
    # Melee: Zombie, GnomeWarrior, OrcSoldier, Lizard, Knight, Troll, Pigman, FrostTroll
    [C_Z, C_G, C_O, C_L, C_K, C_R, C_H, C_Y],
    # Passive: Cow, Bat, Snail, (unused x5)
    [C_u, C_B, C_n, C_SPACE, C_SPACE, C_SPACE, C_SPACE, C_SPACE],
    # Ranged: Skeleton, GnomeArcher, OrcMage, Kobold, Archer, DeepThing, FireElem, IceElem
    [C_S, C_g, C_W, C_k, C_A, C_D, C_X, C_J],
    # Mob projectiles: Arrow, Dagger, Fireball, Iceball, Arrow2, Slimeball, Fireball2, Iceball2
    [C_DASH, C_SLASH, C_STAR, C_STAR, C_DASH, C_STAR, C_STAR, C_STAR],
    # Player projectiles: same chars
    [C_DASH, C_SLASH, C_STAR, C_STAR, C_DASH, C_STAR, C_STAR, C_STAR],
], dtype=jnp.int32)

MOB_COLOR_IDS = jnp.array([
    # Melee
    [CLR_GREEN, CLR_YELLOW, CLR_GREEN, CLR_BRIGHT_GREEN, CLR_BRIGHT_WHITE,
     CLR_GREEN, CLR_BRIGHT_RED, CLR_BRIGHT_CYAN],
    # Passive
    [CLR_YELLOW, CLR_DARK_GRAY, CLR_YELLOW, CLR_NONE, CLR_NONE,
     CLR_NONE, CLR_NONE, CLR_NONE],
    # Ranged
    [CLR_BRIGHT_WHITE, CLR_YELLOW, CLR_BRIGHT_MAGENTA, CLR_RED, CLR_BRIGHT_WHITE,
     CLR_BRIGHT_BLUE, CLR_BRIGHT_RED, CLR_BRIGHT_CYAN],
    # Mob projectiles
    [CLR_WHITE, CLR_WHITE, CLR_BRIGHT_RED, CLR_BRIGHT_CYAN, CLR_BRIGHT_WHITE,
     CLR_BRIGHT_GREEN, CLR_RED, CLR_CYAN],
    # Player projectiles
    [CLR_BRIGHT_YELLOW, CLR_BRIGHT_YELLOW, CLR_BRIGHT_RED, CLR_BRIGHT_CYAN,
     CLR_BRIGHT_YELLOW, CLR_BRIGHT_GREEN, CLR_RED, CLR_CYAN],
], dtype=jnp.int32)

# Player direction -> char: 0=@ (noop), 1=← (left), 2=→ (right), 3=↑ (up), 4=↓ (down)
# Uses Unicode arrows to avoid collisions with < (ladder_up), > (ladder_down), ^ (stalagmite)
PLAYER_DIR_CHARS = jnp.array(
    [C_AT, C_ARROW_LEFT, C_ARROW_RIGHT, C_ARROW_UP, C_ARROW_DOWN], dtype=jnp.int32
)
PLAYER_COLOR_NORMAL = CLR_BRIGHT_WHITE
PLAYER_COLOR_SLEEPING = CLR_DARK_GRAY

# ---------------------------------------------------------------------------
# Mob name table  (Python-side only, for legend and status)
# ---------------------------------------------------------------------------
MOB_NAMES = [
    # Melee (class 0)
    ["Zombie", "Gnome Warrior", "Orc Soldier", "Lizard",
     "Knight", "Troll", "Pigman", "Frost Troll"],
    # Passive (class 1)
    ["Cow", "Bat", "Snail", "", "", "", "", ""],
    # Ranged (class 2)
    ["Skeleton", "Gnome Archer", "Orc Mage", "Kobold",
     "Archer", "Deep Thing", "Fire Elemental", "Ice Elemental"],
    # Mob projectiles (class 3)
    ["Arrow", "Dagger", "Fireball", "Iceball",
     "Arrow", "Slimeball", "Fireball", "Iceball"],
    # Player projectiles (class 4)
    ["Arrow(P)", "Dagger(P)", "Fireball(P)", "Iceball(P)",
     "Arrow(P)", "Slimeball(P)", "Fireball(P)", "Iceball(P)"],
]

FLOOR_NAMES = [
    "Overworld", "Dungeon", "Gnomish Mines", "Sewers",
    "Vaults", "Troll Mines", "Fire Realm", "Ice Realm", "Graveyard",
]

# ---------------------------------------------------------------------------
# Status data indices  (for the flat status_data array)
# ---------------------------------------------------------------------------
# These constants define the layout of the status_data array returned by
# render_craftax_ascii_grid.  Total length = 49.
S_HEALTH = 0
S_MAX_HEALTH = 1
S_FOOD = 2
S_MAX_FOOD = 3
S_DRINK = 4
S_MAX_DRINK = 5
S_ENERGY = 6
S_MAX_ENERGY = 7
S_MANA = 8
S_MAX_MANA = 9
S_XP = 10
S_DEX = 11
S_STR = 12
S_INT = 13
S_FLOOR = 14
S_DIRECTION = 15
S_TIMESTEP = 16
S_LIGHT_LEVEL = 17
S_IS_SLEEPING = 18
S_IS_RESTING = 19
S_LEARNED_FIRE = 20
S_LEARNED_ICE = 21
S_MONSTERS_KILLED = 22
S_MONSTERS_REQUIRED = 23
S_BOSS_VULNERABLE = 24
# Inventory starts at 25
S_INV_WOOD = 25
S_INV_STONE = 26
S_INV_COAL = 27
S_INV_IRON = 28
S_INV_DIAMOND = 29
S_INV_SAPPHIRE = 30
S_INV_RUBY = 31
S_INV_SAPLING = 32
S_INV_TORCHES = 33
S_INV_ARROWS = 34
S_INV_BOOKS = 35
S_INV_PICKAXE = 36
S_INV_SWORD = 37
S_SWORD_ENCH = 38
S_INV_BOW = 39
S_BOW_ENCH = 40
# Potions 41-46
S_POTION_START = 41
# Armor 47-50
S_ARMOR_START = 47
# Armor enchantments 51-54
S_ARMOR_ENCH_START = 51
# Player position
S_POS_ROW = 55
S_POS_COL = 56
STATUS_DATA_LEN = 57


# ===================================================================
# JIT-compiled grid renderer
# ===================================================================

def _build_status_data(state):
    """Build flat array of all HUD values from state."""
    return jnp.array([
        state.player_health,
        get_max_health(state),
        state.player_food,
        get_max_food(state),
        state.player_drink,
        get_max_drink(state),
        state.player_energy,
        get_max_energy(state),
        state.player_mana,
        get_max_mana(state),
        state.player_xp,
        state.player_dexterity,
        state.player_strength,
        state.player_intelligence,
        state.player_level,
        state.player_direction,
        state.timestep,
        state.light_level,
        state.is_sleeping,
        state.is_resting,
        state.learned_spells[0],
        state.learned_spells[1],
        state.monsters_killed[state.player_level],
        MONSTERS_KILLED_TO_CLEAR_LEVEL,
        is_boss_vulnerable(state),
        # Inventory
        state.inventory.wood,
        state.inventory.stone,
        state.inventory.coal,
        state.inventory.iron,
        state.inventory.diamond,
        state.inventory.sapphire,
        state.inventory.ruby,
        state.inventory.sapling,
        state.inventory.torches,
        state.inventory.arrows,
        state.inventory.books,
        state.inventory.pickaxe,
        state.inventory.sword,
        state.sword_enchantment,
        state.inventory.bow,
        state.bow_enchantment,
        # Potions (6)
        state.inventory.potions[0],
        state.inventory.potions[1],
        state.inventory.potions[2],
        state.inventory.potions[3],
        state.inventory.potions[4],
        state.inventory.potions[5],
        # Armor (4)
        state.inventory.armour[0],
        state.inventory.armour[1],
        state.inventory.armour[2],
        state.inventory.armour[3],
        # Armor enchantments (4)
        state.armour_enchantments[0],
        state.armour_enchantments[1],
        state.armour_enchantments[2],
        state.armour_enchantments[3],
        # Player position
        state.player_position[0],
        state.player_position[1],
    ], dtype=jnp.float32)


def _add_mob_to_grid(carry, mob_index):
    """Scan function: place one mob's char/color onto the grids."""
    char_grid, color_grid, mobs, mob_class_index, view_size, player_pos = carry

    local_position = (
        mobs.position[mob_index]
        - player_pos
        + jnp.array([view_size[0], view_size[1]]) // 2
    )
    on_screen = jnp.logical_and(
        local_position >= 0,
        local_position < jnp.array([view_size[0], view_size[1]]),
    ).all()
    on_screen = jnp.logical_and(on_screen, mobs.mask[mob_index])

    char_id = MOB_CHAR_IDS[mob_class_index, mobs.type_id[mob_index]]
    color_id = MOB_COLOR_IDS[mob_class_index, mobs.type_id[mob_index]]

    # Clamp to valid range for safe indexing (even if off-screen)
    row = jnp.clip(local_position[0], 0, view_size[0] - 1)
    col = jnp.clip(local_position[1], 0, view_size[1] - 1)

    char_grid = jnp.where(
        on_screen,
        char_grid.at[row, col].set(char_id),
        char_grid,
    )
    color_grid = jnp.where(
        on_screen,
        color_grid.at[row, col].set(color_id),
        color_grid,
    )

    return (char_grid, color_grid, mobs, mob_class_index, view_size, player_pos), None


def _scan_mob_class(char_grid, color_grid, mobs, mob_class_index, view_size, player_pos):
    """Scan over all mobs in a class and overlay them onto the grids."""
    (char_grid, color_grid, _, _, _, _), _ = jax.lax.scan(
        _add_mob_to_grid,
        (char_grid, color_grid, mobs, mob_class_index, view_size, player_pos),
        jnp.arange(mobs.mask.shape[0]),
    )
    return char_grid, color_grid


@jax.jit
def render_craftax_ascii_grid_local(state):
    """JIT-compiled: render the 9x11 local view as char/color ID grids.

    Returns:
        char_grid: jnp.int32 array of shape (9, 11)
        color_grid: jnp.int32 array of shape (9, 11)
        status_data: jnp.float32 array of shape (STATUS_DATA_LEN,)
    """
    level = state.player_level
    map_data = state.map[level]
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # --- Pad and slice map (same as render_craftax_symbolic) ---
    padded_grid = jnp.pad(
        map_data,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )
    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    # --- Handle boss vulnerability ---
    vulnerable = is_boss_vulnerable(state)
    map_view = jnp.where(
        jnp.logical_and(map_view == BlockType.NECROMANCER.value, vulnerable),
        BlockType.NECROMANCER_VULNERABLE.value,
        map_view,
    )

    # --- Block -> char/color ---
    char_grid = BLOCK_CHAR_IDS[map_view]
    color_grid = BLOCK_COLOR_IDS[map_view]

    # --- Item overlay ---
    padded_items = jnp.pad(
        state.item_map[level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )
    item_view = jax.lax.dynamic_slice(padded_items, tl_corner, OBS_DIM)

    # Handle ladder blocking
    ladder_blocked = state.monsters_killed[level] < MONSTERS_KILLED_TO_CLEAR_LEVEL
    effective_items = jnp.where(
        jnp.logical_and(
            item_view == ItemType.LADDER_DOWN.value,
            ladder_blocked,
        ),
        ItemType.LADDER_DOWN_BLOCKED.value,
        item_view,
    )

    has_item = effective_items != ItemType.NONE.value
    char_grid = jnp.where(has_item, ITEM_CHAR_IDS[effective_items], char_grid)
    color_grid = jnp.where(has_item, ITEM_COLOR_IDS[effective_items], color_grid)

    # --- Mob overlay ---
    view_size = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    melee = jax.tree_util.tree_map(lambda x: x[level], state.melee_mobs)
    passive = jax.tree_util.tree_map(lambda x: x[level], state.passive_mobs)
    ranged = jax.tree_util.tree_map(lambda x: x[level], state.ranged_mobs)
    mob_proj = jax.tree_util.tree_map(lambda x: x[level], state.mob_projectiles)
    player_proj = jax.tree_util.tree_map(lambda x: x[level], state.player_projectiles)

    char_grid, color_grid = _scan_mob_class(
        char_grid, color_grid, melee, 0, view_size, state.player_position
    )
    char_grid, color_grid = _scan_mob_class(
        char_grid, color_grid, passive, 1, view_size, state.player_position
    )
    char_grid, color_grid = _scan_mob_class(
        char_grid, color_grid, ranged, 2, view_size, state.player_position
    )
    char_grid, color_grid = _scan_mob_class(
        char_grid, color_grid, mob_proj, 3, view_size, state.player_position
    )
    char_grid, color_grid = _scan_mob_class(
        char_grid, color_grid, player_proj, 4, view_size, state.player_position
    )

    # --- Light mask ---
    padded_light = jnp.pad(
        state.light_map[level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=0.0,
    )
    light_view = jax.lax.dynamic_slice(padded_light, tl_corner, OBS_DIM)
    light_mask = light_view > 0.05

    # Preserve OOB tiles (they should always be visible as map edge)
    is_oob = map_view == BlockType.OUT_OF_BOUNDS.value

    char_grid = jnp.where(light_mask, char_grid, C_SPACE)
    color_grid = jnp.where(light_mask, color_grid, CLR_NONE)

    # Restore OOB tiles after light mask
    char_grid = jnp.where(is_oob, BLOCK_CHAR_IDS[BlockType.OUT_OF_BOUNDS.value], char_grid)
    color_grid = jnp.where(is_oob, BLOCK_COLOR_IDS[BlockType.OUT_OF_BOUNDS.value], color_grid)

    # --- Player at center (directional: ← → ↑ ↓) ---
    center_r = OBS_DIM[0] // 2
    center_c = OBS_DIM[1] // 2
    player_char = PLAYER_DIR_CHARS[jnp.clip(state.player_direction, 0, 4)]
    player_color = jnp.where(state.is_sleeping, PLAYER_COLOR_SLEEPING, PLAYER_COLOR_NORMAL)
    char_grid = char_grid.at[center_r, center_c].set(player_char)
    color_grid = color_grid.at[center_r, center_c].set(player_color)

    # --- Status data ---
    status_data = _build_status_data(state)

    return char_grid, color_grid, status_data


@jax.jit
def render_craftax_ascii_grid_full(state, discovered_map):
    """JIT-compiled: render the full 48x48 map with fog of war.

    Args:
        state: EnvState
        discovered_map: jnp.bool_ array of shape (9, 48, 48) tracking visited tiles

    Returns:
        char_grid: jnp.int32 array of shape (48, 48)
        color_grid: jnp.int32 array of shape (48, 48)
        discovered_map: updated discovered_map
        status_data: jnp.float32 array of shape (STATUS_DATA_LEN,)
    """
    level = state.player_level
    map_data = state.map[level]

    # --- Handle boss vulnerability ---
    vulnerable = is_boss_vulnerable(state)
    map_view = jnp.where(
        jnp.logical_and(map_data == BlockType.NECROMANCER.value, vulnerable),
        BlockType.NECROMANCER_VULNERABLE.value,
        map_data,
    )

    # --- Block -> char/color ---
    char_grid = BLOCK_CHAR_IDS[map_view]
    color_grid = BLOCK_COLOR_IDS[map_view]

    # --- Item overlay ---
    item_data = state.item_map[level]
    ladder_blocked = state.monsters_killed[level] < MONSTERS_KILLED_TO_CLEAR_LEVEL
    effective_items = jnp.where(
        jnp.logical_and(
            item_data == ItemType.LADDER_DOWN.value,
            ladder_blocked,
        ),
        ItemType.LADDER_DOWN_BLOCKED.value,
        item_data,
    )
    has_item = effective_items != ItemType.NONE.value
    char_grid = jnp.where(has_item, ITEM_CHAR_IDS[effective_items], char_grid)
    color_grid = jnp.where(has_item, ITEM_COLOR_IDS[effective_items], color_grid)

    # --- Mob overlay ---
    map_size = jnp.array([48, 48], dtype=jnp.int32)
    # _add_mob_to_grid computes: local = mob_pos - origin + view_size // 2
    # For full mode we want local = mob_pos (absolute coords), so
    # origin must equal view_size // 2 to cancel the offset.
    origin = map_size // 2

    melee = jax.tree_util.tree_map(lambda x: x[level], state.melee_mobs)
    passive = jax.tree_util.tree_map(lambda x: x[level], state.passive_mobs)
    ranged = jax.tree_util.tree_map(lambda x: x[level], state.ranged_mobs)
    mob_proj = jax.tree_util.tree_map(lambda x: x[level], state.mob_projectiles)
    player_proj = jax.tree_util.tree_map(lambda x: x[level], state.player_projectiles)

    char_grid, color_grid = _scan_mob_class(
        char_grid, color_grid, melee, 0, map_size, origin
    )
    char_grid, color_grid = _scan_mob_class(
        char_grid, color_grid, passive, 1, map_size, origin
    )
    char_grid, color_grid = _scan_mob_class(
        char_grid, color_grid, ranged, 2, map_size, origin
    )
    char_grid, color_grid = _scan_mob_class(
        char_grid, color_grid, mob_proj, 3, map_size, origin
    )
    char_grid, color_grid = _scan_mob_class(
        char_grid, color_grid, player_proj, 4, map_size, origin
    )

    # --- Light mask ---
    light_data = state.light_map[level]
    light_mask = light_data > 0.05

    # --- Update discovered map ---
    r = state.player_position[0]
    c = state.player_position[1]
    rows = jnp.arange(48)
    cols = jnp.arange(48)
    in_view_r = jnp.logical_and(rows >= r - OBS_DIM[0] // 2, rows <= r + OBS_DIM[0] // 2)
    in_view_c = jnp.logical_and(cols >= c - OBS_DIM[1] // 2, cols <= c + OBS_DIM[1] // 2)
    view_mask = in_view_r[:, None] & in_view_c[None, :]  # (48, 48)
    newly_visible = view_mask & light_mask
    updated_discovered = discovered_map.at[level].set(
        jnp.logical_or(discovered_map[level], newly_visible)
    )

    # --- Apply visibility ---
    # Currently visible: full rendering
    # Discovered but not currently visible: show terrain only (dim) - no mobs/items
    # Undiscovered: blank
    is_discovered = updated_discovered[level]

    # For discovered-but-dark tiles, show the base block only (no mobs, no items)
    base_char = BLOCK_CHAR_IDS[map_view]
    base_color = BLOCK_COLOR_IDS[map_view]

    # Merge: start with discovered terrain (dim), then overlay currently visible
    final_char = jnp.where(is_discovered, base_char, C_SPACE)
    final_color = jnp.where(is_discovered, CLR_DIM, CLR_NONE)

    # Currently visible tiles get full rendering (with mobs, items)
    final_char = jnp.where(light_mask, char_grid, final_char)
    final_color = jnp.where(light_mask, color_grid, final_color)

    # --- Player (directional: ← → ↑ ↓) ---
    player_r = state.player_position[0]
    player_c = state.player_position[1]
    player_char = PLAYER_DIR_CHARS[jnp.clip(state.player_direction, 0, 4)]
    player_color = jnp.where(state.is_sleeping, PLAYER_COLOR_SLEEPING, PLAYER_COLOR_NORMAL)
    final_char = final_char.at[player_r, player_c].set(player_char)
    final_color = final_color.at[player_r, player_c].set(player_color)

    # --- Status data ---
    status_data = _build_status_data(state)

    return final_char, final_color, updated_discovered, status_data


# ===================================================================
# Fog of war helpers
# ===================================================================

def create_discovered_map():
    """Create a fresh discovered_map JAX array of shape (9, 48, 48)."""
    return jnp.zeros((9, 48, 48), dtype=jnp.bool_)


# ===================================================================
# Python string formatter
# ===================================================================

def _vital_color(current, maximum):
    """Return ANSI color code for a vital stat bar."""
    if maximum <= 0:
        return ""
    ratio = current / maximum
    if ratio > 0.5:
        return "\033[1;32m"  # bright green
    elif ratio > 0.25:
        return "\033[1;33m"  # bright yellow
    else:
        return "\033[1;31m"  # bright red


def _material_name(level):
    """Convert pickaxe/sword level to material name."""
    names = {0: "None", 1: "Wood", 2: "Stone", 3: "Iron", 4: "Diamond"}
    return names.get(int(level), "None")


def _enchantment_name(level):
    """Convert enchantment level to name."""
    names = {0: "None", 1: "Fire", 2: "Ice"}
    return names.get(int(level), "None")


def _direction_name(d):
    """Convert direction int to name."""
    names = {0: "none", 1: "left", 2: "right", 3: "up", 4: "down"}
    return names.get(int(d), "none")


def _armor_name(level):
    """Convert armor level to material name."""
    names = {0: "None", 1: "Iron", 2: "Diamond"}
    return names.get(int(level), "None")


def ascii_grid_to_string(char_grid, color_grid, status_data,
                         include_legend=True, use_color=True):
    """Convert JAX char/color ID grids + status data to an ANSI string.

    Args:
        char_grid: jnp.int32 array of shape (H, W)
        color_grid: jnp.int32 array of shape (H, W)
        status_data: jnp.float32 array of shape (STATUS_DATA_LEN,)
        include_legend: whether to append the character legend
        use_color: whether to include ANSI color codes

    Returns:
        str: the complete ASCII rendering
    """
    char_np = np.array(char_grid)
    color_np = np.array(color_grid)
    sd = np.array(status_data)
    H, W = char_np.shape

    lines = []

    # --- Map grid ---
    for row in range(H):
        row_chars = []
        for col in range(W):
            ch = CHAR_TABLE[int(char_np[row, col])]
            if use_color:
                clr = ANSI_TABLE[int(color_np[row, col])]
                if clr:
                    row_chars.append(f"{clr}{ch}{ANSI_RESET}")
                else:
                    row_chars.append(ch)
            else:
                row_chars.append(ch)
        lines.append("".join(row_chars))

    lines.append("")  # blank separator

    # --- Status HUD ---
    hp, max_hp = sd[S_HEALTH], sd[S_MAX_HEALTH]
    food, max_food = sd[S_FOOD], sd[S_MAX_FOOD]
    drink, max_drink = sd[S_DRINK], sd[S_MAX_DRINK]
    energy, max_energy = sd[S_ENERGY], sd[S_MAX_ENERGY]
    mana, max_mana = sd[S_MANA], sd[S_MAX_MANA]

    def _fmt_vital(name, cur, mx):
        cur_i, mx_i = int(cur), int(mx)
        if use_color:
            clr = _vital_color(cur, mx)
            return f"{clr}{name}:{cur_i}/{mx_i}{ANSI_RESET}"
        return f"{name}:{cur_i}/{mx_i}"

    pos_r, pos_c = int(sd[S_POS_ROW]), int(sd[S_POS_COL])
    facing = _direction_name(sd[S_DIRECTION])
    vitals = "  ".join([
        _fmt_vital("HP", hp, max_hp),
        _fmt_vital("Food", food, max_food),
        _fmt_vital("Drink", drink, max_drink),
        _fmt_vital("Energy", energy, max_energy),
        _fmt_vital("Mana", mana, max_mana),
        f"XP:{int(sd[S_XP])}",
        f"Pos:({pos_r},{pos_c})",
        f"Facing:{facing}",
    ])
    lines.append(vitals)

    floor_idx = int(sd[S_FLOOR])
    floor_name = FLOOR_NAMES[floor_idx] if floor_idx < len(FLOOR_NAMES) else f"Floor{floor_idx}"
    attrs = (
        f"Str:{int(sd[S_STR])}  Dex:{int(sd[S_DEX])}  Int:{int(sd[S_INT])}  "
        f"Floor:{floor_idx}({floor_name})  "
        f"T:{int(sd[S_TIMESTEP])}"
    )
    lines.append(attrs)

    # --- Inventory ---
    inv_parts = []
    for label, idx in [("Wood", S_INV_WOOD), ("Stone", S_INV_STONE),
                        ("Coal", S_INV_COAL), ("Iron", S_INV_IRON),
                        ("Diamond", S_INV_DIAMOND), ("Sapphire", S_INV_SAPPHIRE),
                        ("Ruby", S_INV_RUBY), ("Sapling", S_INV_SAPLING)]:
        inv_parts.append(f"{label}:{int(sd[idx])}")
    lines.append("[Inv] " + " ".join(inv_parts))

    # --- Equipment ---
    equip_parts = []
    pick_lvl = int(sd[S_INV_PICKAXE])
    equip_parts.append(f"{_material_name(pick_lvl)} Pickaxe")

    sword_lvl = int(sd[S_INV_SWORD])
    sword_ench = int(sd[S_SWORD_ENCH])
    if sword_lvl > 0 and sword_ench > 0:
        equip_parts.append(f"{_material_name(sword_lvl)} Sword({_enchantment_name(sword_ench)})")
    elif sword_lvl > 0:
        equip_parts.append(f"{_material_name(sword_lvl)} Sword")
    else:
        equip_parts.append("No Sword")

    bow = int(sd[S_INV_BOW])
    bow_ench = int(sd[S_BOW_ENCH])
    if bow > 0 and bow_ench > 0:
        equip_parts.append(f"Bow({_enchantment_name(bow_ench)})")
    elif bow > 0:
        equip_parts.append("Bow")
    else:
        equip_parts.append("No Bow")

    equip_parts.append(f"Arrows:{int(sd[S_INV_ARROWS])}")
    equip_parts.append(f"Torches:{int(sd[S_INV_TORCHES])}")
    equip_parts.append(f"Books:{int(sd[S_INV_BOOKS])}")
    lines.append("[Equip] " + " | ".join(equip_parts))

    # --- Armor ---
    armor_names = ["Helm", "Chest", "Legs", "Boots"]
    armor_parts = []
    has_armor = False
    for j, name in enumerate(armor_names):
        a_lvl = int(sd[S_ARMOR_START + j])
        a_ench = int(sd[S_ARMOR_ENCH_START + j])
        if a_lvl > 0:
            has_armor = True
            part = f"{_armor_name(a_lvl)} {name}"
            if a_ench > 0:
                part += f"({_enchantment_name(a_ench)})"
            armor_parts.append(part)
    if has_armor:
        lines.append("[Armor] " + " | ".join(armor_parts))
    else:
        lines.append("[Armor] None")

    # --- Potions ---
    potion_names = ["Red", "Green", "Blue", "Pink", "Cyan", "Yellow"]
    pot_parts = []
    for j, name in enumerate(potion_names):
        pot_parts.append(f"{name}:{int(sd[S_POTION_START + j])}")
    lines.append("[Potions] " + " ".join(pot_parts))

    # --- Status flags ---
    flags = []
    sleeping = "yes" if sd[S_IS_SLEEPING] > 0.5 else "no"
    resting = "yes" if sd[S_IS_RESTING] > 0.5 else "no"
    flags.append(f"Sleeping:{sleeping}")
    flags.append(f"Resting:{resting}")

    fire_learned = "learned" if sd[S_LEARNED_FIRE] > 0.5 else "no"
    ice_learned = "learned" if sd[S_LEARNED_ICE] > 0.5 else "no"
    flags.append(f"Fireball:{fire_learned}")
    flags.append(f"Iceball:{ice_learned}")

    kills = int(sd[S_MONSTERS_KILLED])
    required = int(sd[S_MONSTERS_REQUIRED])
    if kills >= required:
        flags.append("Ladder:OPEN")
    else:
        flags.append(f"Ladder:CLOSED({kills}/{required})")

    boss_status = "VULNERABLE" if sd[S_BOSS_VULNERABLE] > 0.5 else "no"
    flags.append(f"Boss:{boss_status}")

    light = sd[S_LIGHT_LEVEL]
    if light < 0.3:
        flags.append("Night")
    elif light < 0.6:
        flags.append("Dusk")
    else:
        flags.append("Day")

    lines.append("[Status] " + " | ".join(flags))

    # --- Legend ---
    if include_legend:
        lines.append("")
        lines.append("--- Legend ---")
        lines.append("Terrain: .=grass/path ~=water &=lava ,=sand ;=fire/ice_grass :=gravel ==wood")
        lines.append("Blocks:  T=tree %=stone #=wall ^=stalagmite t=ice_shrub {=fountain \u2588=map_edge(impassable)")
        lines.append("Ores:    c=coal i=iron d=diamond s=sapphire r=ruby")
        lines.append("Build:   C=craft_table F=furnace E=enchant_table $=chest +=grave N=necromancer")
        lines.append("Flora:   p=plant P=ripe_plant")
        lines.append("Items:   *=torch >=ladder_down(red=blocked) <=ladder_up")
        lines.append("Player:  \u2190=you(left) \u2192=you(right) \u2191=you(up) \u2193=you(down) @=you(no dir)")
        lines.append(
            "Melee:   Z=zombie G=gnome_war O=orc L=lizard K=knight R=troll H=pigman Y=frost_troll"
        )
        lines.append("Passive: u=cow B=bat n=snail")
        lines.append(
            "Ranged:  S=skeleton g=gnome_arch W=orc_mage k=kobold A=archer "
            "D=deep_thing X=fire_elem J=ice_elem"
        )
        lines.append("Projectiles: -=arrow /=dagger *=fireball/iceball/slimeball")

    return "\n".join(lines)


# ===================================================================
# Convenience wrappers
# ===================================================================

def render_craftax_ascii(
    state,
    discovered_map=None,
    view_mode="local",
    include_legend=True,
    use_color=True,
):
    """Render the game state as an ANSI-colored ASCII string.

    Args:
        state: EnvState
        discovered_map: optional jnp.bool_ array (9, 48, 48) for fog of war.
            Only used when view_mode="full".
        view_mode: "local" for 9x11 window, "full" for 48x48 with fog of war
        include_legend: whether to append the character legend
        use_color: whether to include ANSI color codes

    Returns:
        (ascii_string, discovered_map) tuple.  ``discovered_map`` is the
        updated fog-of-war array in full mode, or ``None`` in local mode.
    """
    if view_mode == "full":
        if discovered_map is None:
            discovered_map = create_discovered_map()
        char_grid, color_grid, updated_discovered, status_data = (
            render_craftax_ascii_grid_full(state, discovered_map)
        )
        result_str = ascii_grid_to_string(
            char_grid,
            color_grid,
            status_data,
            include_legend=include_legend,
            use_color=use_color,
        )
        return result_str, updated_discovered
    else:
        char_grid, color_grid, status_data = render_craftax_ascii_grid_local(
            state
        )
        result_str = ascii_grid_to_string(
            char_grid,
            color_grid,
            status_data,
            include_legend=include_legend,
            use_color=use_color,
        )
        return result_str, None
