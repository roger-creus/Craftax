"""Persistent spatial memory and scratchpad for Craftax LLM agents.

Provides:
  - SpatialMemory: structured key-point store with floor-aware get/set/remove
  - StructuredScratchpad: section-based scratchpad with merge-on-update
  - Parsing utilities to extract memory commands from LLM output
"""

import json
import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# Spatial Memory  (persistent JSON-like key-point store)
# ---------------------------------------------------------------------------

class SpatialMemory:
    """Structured spatial memory organized by floor.

    Stores key locations as {floor -> [{type, row, col, note}]}.
    The agent interacts via commands embedded in its output:
        MAP_ADD: floor=0, type=water, row=20, col=30, note=large lake
        MAP_GET: floor=0
        MAP_GET: floor=0, type=water
        MAP_REMOVE: floor=0, type=cow, row=28, col=15
    """

    def __init__(self):
        self.data = defaultdict(list)  # floor -> list of entries
        self._pending_queries = []     # results to show next turn

    MAX_ENTRIES_PER_FLOOR = 20

    def add(self, floor, type_name, row, col, note=""):
        """Add a key-point. Deduplicates and enforces per-floor cap."""
        floor = int(floor)
        row, col = int(row), int(col)
        type_name = type_name.lower().strip()
        # Check for duplicate at same position (any type)
        for entry in self.data[floor]:
            if entry["row"] == row and entry["col"] == col:
                # Update existing entry at this position
                entry["type"] = type_name
                entry["note"] = note
                return
        # Enforce per-floor cap
        if len(self.data[floor]) >= self.MAX_ENTRIES_PER_FLOOR:
            return  # silently drop — memory is full
        self.data[floor].append({
            "type": type_name,
            "row": row,
            "col": col,
            "note": note,
        })

    def remove(self, floor, type_name=None, row=None, col=None):
        """Remove entries matching the criteria."""
        floor = int(floor)
        if floor not in self.data:
            return 0
        before = len(self.data[floor])
        self.data[floor] = [
            e for e in self.data[floor]
            if not (
                (type_name is None or e["type"] == type_name)
                and (row is None or e["row"] == int(row))
                and (col is None or e["col"] == int(col))
            )
        ]
        return before - len(self.data[floor])

    def get(self, floor=None, type_name=None):
        """Query entries. Returns list of entries matching criteria."""
        results = []
        floors = [int(floor)] if floor is not None else sorted(self.data.keys())
        for f in floors:
            for entry in self.data.get(f, []):
                if type_name is None or entry["type"] == type_name:
                    results.append({"floor": f, **entry})
        return results

    def format_entries(self, entries):
        """Format entries as readable text for the agent."""
        if not entries:
            return "(no entries found)"
        lines = []
        for e in entries:
            line = f"  Floor {e['floor']}: {e['type']} at ({e['row']},{e['col']})"
            if e.get("note"):
                line += f" - {e['note']}"
            lines.append(line)
        return "\n".join(lines)

    def format_floor_summary(self, floor):
        """One-line summary of how many entries per type on a floor."""
        entries = self.data.get(int(floor), [])
        if not entries:
            return f"Floor {floor}: (no saved locations)"
        type_counts = defaultdict(int)
        for e in entries:
            type_counts[e["type"]] += 1
        parts = [f"{t}:{c}" for t, c in sorted(type_counts.items())]
        return f"Floor {floor}: {', '.join(parts)} ({len(entries)}/{self.MAX_ENTRIES_PER_FLOOR})"

    def to_json(self):
        return dict(self.data)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def load(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.data = defaultdict(list)
        for floor_str, entries in raw.items():
            self.data[int(floor_str)] = entries


# ---------------------------------------------------------------------------
# Structured Scratchpad  (section-based, merge-on-update)
# ---------------------------------------------------------------------------

# Section definitions: (name, max_chars, default_text)
_SCRATCHPAD_SECTIONS = [
    ("STRATEGY", 100, "Reach floor 8 and defeat the Necromancer."),
    ("PLAN", 300, "1. [ ] Get wood, place table, craft wood pickaxe\n2. [ ] Mine stone/coal/iron, craft iron tools + armour\n3. [ ] Find ladder and DESCEND"),
    ("LESSONS", 200, "(Record game mechanics discoveries here)"),
    ("TACTICAL", 200, "(Describe your immediate situation and next move)"),
    ("FLOOR_NOTES", 400, "(Record what you find on this floor: dead ends, cleared rooms, unexplored directions, dangerous areas, resource locations)"),
    ("FLOOR_CHECKLIST", 150, "(List what you need before descending)"),
]

_SECTION_NAMES = [s[0] for s in _SCRATCHPAD_SECTIONS]
_SECTION_PATTERN = re.compile(
    r"^(" + "|".join(_SECTION_NAMES) + r")\s*:\s*(.*?)(?=\n(?:" + "|".join(_SECTION_NAMES) + r")\s*:|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)


class StructuredScratchpad:
    """Section-based scratchpad with merge-on-update semantics.

    Sections: STRATEGY, PLAN, LESSONS, TACTICAL, FLOOR_CHECKLIST.
    When the LLM outputs only some sections, the rest are preserved unchanged.
    """

    def __init__(self):
        self.sections = {}
        self._max_chars = {}
        for name, max_chars, default in _SCRATCHPAD_SECTIONS:
            self.sections[name] = default
            self._max_chars[name] = max_chars

    def update_from_text(self, raw_text):
        """Parse section headers from LLM output and merge into current state.

        Only sections explicitly mentioned are updated. Returns True if any
        section was updated.
        """
        if not raw_text or not raw_text.strip():
            return False

        updated = False
        for match in _SECTION_PATTERN.finditer(raw_text):
            name = match.group(1).upper()
            content = match.group(2).strip()
            if name in self.sections and content:
                max_c = self._max_chars[name]
                if len(content) > max_c:
                    content = content[:max_c] + "..."
                self.sections[name] = content
                updated = True

        # Fallback: if no sections were found but text is non-trivial,
        # treat the whole thing as a TACTICAL update (backward compat)
        if not updated and len(raw_text.strip()) > 10:
            content = raw_text.strip()
            max_c = self._max_chars["TACTICAL"]
            if len(content) > max_c:
                content = content[:max_c] + "..."
            self.sections["TACTICAL"] = content
            updated = True

        return updated

    def format_for_prompt(self):
        """Render all sections for injection into the LLM prompt."""
        lines = []
        for name, _, _ in _SCRATCHPAD_SECTIONS:
            lines.append(f"{name}: {self.sections[name]}")
        return "\n".join(lines)

    def __str__(self):
        """Compact display for terminal rendering."""
        parts = []
        for name, _, _ in _SCRATCHPAD_SECTIONS:
            val = self.sections[name]
            if len(val) > 80:
                val = val[:80] + "..."
            parts.append(f"[{name}] {val}")
        return "\n".join(parts)

    def to_json(self):
        return dict(self.sections)


# ---------------------------------------------------------------------------
# Parse memory commands from LLM output
# ---------------------------------------------------------------------------

def _extract_kv(text):
    """Extract key=value pairs from a command string, flexibly.

    Handles: floor=0, type=water, row=20, col=30, note=big lake
    Also:    floor 0 type water row 20 col 30
    Also:    floor=0 type=water x=20 y=30
    """
    result = {}
    # Try key=value pairs first (with optional commas/spaces)
    for m in re.finditer(r"(\w+)\s*=\s*([^,\n]+?)(?:\s*,|\s*$|\s+(?=\w+\s*=))", text):
        key = m.group(1).lower().strip()
        val = m.group(2).strip().rstrip(",")
        result[key] = val
    # Normalize aliases
    if "x" in result and "row" not in result:
        result["row"] = result.pop("x")
    if "y" in result and "col" not in result:
        result["col"] = result.pop("y")
    if "pos" in result and "row" not in result:
        # Handle pos=(20,30) or pos=20,30
        pos = result.pop("pos").strip("() ")
        parts = re.split(r"[,\s]+", pos)
        if len(parts) >= 2:
            result["row"] = parts[0]
            result["col"] = parts[1]
    if "location" in result and "type" not in result:
        result["type"] = result.pop("location")
    if "name" in result and "type" not in result:
        result["type"] = result.pop("name")
    return result


def parse_memory_commands(text, memory):
    """Parse MAP_ADD, MAP_GET, MAP_REMOVE commands from LLM output.

    Only parses commands from the final structured block (after '---' delimiter).
    This prevents the parser from picking up commands mentioned in reasoning.

    Returns (cleaned_text, query_results_str).
    cleaned_text has the command block stripped out.
    query_results_str is text to inject into the next prompt (MAP_GET results).
    """
    query_results = []
    lines_to_remove = []

    # Find the last '---' delimiter — only parse after it
    delimiter_pos = text.rfind("\n---")
    if delimiter_pos == -1:
        delimiter_pos = text.rfind("---")
    if delimiter_pos != -1:
        parse_region = text[delimiter_pos:]
    else:
        # No delimiter — fall back to parsing the whole text
        # (for backward compat / models that don't use the delimiter)
        parse_region = text

    for line_match in re.finditer(r"^.*$", parse_region, re.MULTILINE):
        line = line_match.group().strip()
        line_lower = line.lower()

        if line_lower.startswith("map_add"):
            rest = re.sub(r"^map_add\s*:?\s*", "", line, flags=re.IGNORECASE)
            kv = _extract_kv(rest)
            if "floor" in kv and "type" in kv and "row" in kv and "col" in kv:
                try:
                    memory.add(
                        int(kv["floor"]),
                        kv["type"].lower().strip(),
                        int(kv["row"]),
                        int(kv["col"]),
                        kv.get("note", ""),
                    )
                except (ValueError, KeyError):
                    pass
            lines_to_remove.append(line_match.span())

        elif line_lower.startswith("map_remove"):
            rest = re.sub(r"^map_remove\s*:?\s*", "", line, flags=re.IGNORECASE)
            kv = _extract_kv(rest)
            if "floor" in kv:
                try:
                    removed = memory.remove(
                        int(kv["floor"]),
                        kv.get("type", "").lower().strip() or None,
                        int(kv["row"]) if "row" in kv else None,
                        int(kv["col"]) if "col" in kv else None,
                    )
                    query_results.append(f"(Removed {removed} entries)")
                except (ValueError, KeyError):
                    pass
            lines_to_remove.append(line_match.span())

        elif line_lower.startswith("map_get_all"):
            entries = memory.get()
            query_results.append(
                f"MAP_GET_ALL results:\n{memory.format_entries(entries)}"
            )
            lines_to_remove.append(line_match.span())

        elif line_lower.startswith("map_get"):
            rest = re.sub(r"^map_get\s*:?\s*", "", line, flags=re.IGNORECASE)
            kv = _extract_kv(rest)
            if "floor" in kv:
                try:
                    floor = int(kv["floor"])
                    type_name = kv.get("type", "").lower().strip() or None
                    entries = memory.get(floor, type_name)
                    label = f"floor={floor}"
                    if type_name:
                        label += f", type={type_name}"
                    query_results.append(
                        f"MAP_GET results ({label}):\n{memory.format_entries(entries)}"
                    )
                except (ValueError, KeyError):
                    pass
            lines_to_remove.append(line_match.span())

    # Remove command lines from text (in reverse to preserve offsets)
    cleaned = text
    for start, end in reversed(lines_to_remove):
        cleaned = cleaned[:start] + cleaned[end:]
    # Clean up blank lines left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    results_str = "\n".join(query_results) if query_results else ""
    return cleaned, results_str


def parse_agent_response(text, memory, action_fuzzy):
    """Parse a complete agent response: action, MAP commands, scratchpad.

    Only parses from the final '---' delimiter block. Everything before it
    is treated as reasoning (ignored by parser).

    Args:
        text: full LLM output (may include <think> tags)
        memory: SpatialMemory instance to update
        action_fuzzy: dict mapping lowercase action names to action IDs

    Returns:
        (action_id, action_name, new_scratchpad, query_results_str, reasoning_text,
         no_reasoning: bool — True if reasoning was too short)
    """
    # Strip <think>...</think> blocks for parsing
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Find the --- delimiter
    delimiter_pos = clean.rfind("\n---")
    if delimiter_pos == -1:
        delimiter_pos = clean.rfind("---")

    if delimiter_pos != -1:
        reasoning = clean[:delimiter_pos].strip()
        command_block = clean[delimiter_pos:]
    else:
        # No delimiter — treat entire text as both reasoning and command block
        reasoning = clean
        command_block = clean

    # Parse MAP commands from command block only
    _, query_results = parse_memory_commands(command_block, memory)

    # Parse structured scratchpad sections from command block
    # Find the earliest section header to separate action text from scratchpad
    section_positions = []
    for sname in _SECTION_NAMES:
        m = re.search(
            r"^" + sname + r"\s*:", command_block, re.MULTILINE | re.IGNORECASE
        )
        if m:
            section_positions.append(m.start())

    if section_positions:
        split_pos = min(section_positions)
        new_scratchpad = command_block[split_pos:].strip()
        action_text = command_block[:split_pos]
    else:
        # Backward compat: try old Scratchpad: format
        scratchpad_match = re.search(
            r"Scratchpad:\s*\n(.*)", command_block, re.DOTALL | re.IGNORECASE
        )
        if scratchpad_match:
            new_scratchpad = scratchpad_match.group(1).strip()
            action_text = command_block[: scratchpad_match.start()]
        else:
            new_scratchpad = None
            action_text = command_block

    # Parse action using multi-strategy fallback chain
    action_id, action_name = _parse_action(action_text, action_fuzzy)

    # If command block failed, try the full text as fallback
    if action_name == "NOOP (parse failed)" and command_block != clean:
        action_id, action_name = _parse_action(clean, action_fuzzy)

    # Check reasoning quality
    no_reasoning = bool(text) and len(reasoning) < 30

    return action_id, action_name, new_scratchpad, query_results, reasoning, no_reasoning


def _parse_action(text, action_fuzzy):
    """Multi-strategy action parser. Returns (action_id, action_name).

    Strategies tried in order:
    1. 'Action: <name>' pattern (last match)
    2. Known action name on its own line
    3. Entire text is a single known action name
    4. JSON-like '"action": "<name>"' pattern
    5. Longest known action name found anywhere in text
    """
    text_lower = text.lower()

    # Strategy 1: Action: <name> pattern (last match wins)
    action_matches = list(re.finditer(
        r"(?:action|act|choose|select)\s*[:=]\s*([a-z_\s]+)",
        text_lower,
    ))
    if action_matches:
        candidate = action_matches[-1].group(1).strip().rstrip(".")
        if candidate in action_fuzzy:
            return action_fuzzy[candidate], candidate

    # Strategy 2: Known action name as a standalone line
    for line in text_lower.splitlines():
        candidate = line.strip().rstrip(".")
        if candidate in action_fuzzy:
            return action_fuzzy[candidate], candidate

    # Strategy 3: Entire text (stripped) is a single known action name
    whole = text_lower.strip().rstrip(".")
    if whole in action_fuzzy:
        return action_fuzzy[whole], whole

    # Strategy 4: JSON-like "action": "<name>"
    json_match = re.search(r'"action"\s*:\s*"([^"]+)"', text_lower)
    if json_match:
        candidate = json_match.group(1).strip()
        if candidate in action_fuzzy:
            return action_fuzzy[candidate], candidate

    # Strategy 5: Longest known action name found anywhere
    for name in sorted(action_fuzzy.keys(), key=len, reverse=True):
        if name in text_lower:
            return action_fuzzy[name], name

    return 0, "NOOP (parse failed)"


# ---------------------------------------------------------------------------
# Prompt sections for the agent
# ---------------------------------------------------------------------------

EMPTY_SCRATCHPAD = """\
STRATEGY: (No strategy set yet)
PLAN: 1. [ ] (Add your first step)
LESSONS: (Record game mechanics discoveries here)
TACTICAL: (Describe your immediate situation and next move)
FLOOR_CHECKLIST: (List what you need before descending)"""


MEMORY_INSTRUCTIONS = """\

## Spatial Memory (persistent key-point database)
You have a persistent spatial database to record and recall important locations.
It stores structured (floor, type, row, col) data across turns.

### Available memory commands:
  MAP_ADD: floor=0, type=water, row=20, col=30, note=large lake
  MAP_ADD: floor=1, type=crafting_table, row=12, col=8, note=my table
  MAP_GET: floor=0                    (get all saved locations on floor 0)
  MAP_GET: floor=0, type=water        (get only water locations on floor 0)
  MAP_GET_ALL                         (get everything across all floors)
  MAP_REMOVE: floor=0, type=cow, row=28, col=15   (remove a specific entry)

### What to record (only high-value persistent landmarks):
  YES: water sources, crafting tables YOU placed, furnaces YOU placed, ladders,
       enchantment tables, chests, ore deposits (iron/diamond/sapphire/ruby)
  NO:  trees (too common), cows/mobs (they move), grass, stone (everywhere),
       coal (too common), individual tiles you've walked on
  Memory is limited to 20 entries per floor. Be selective!
Use MAP_REMOVE when a resource is consumed (ore mined, chest opened).
MAP_GET results appear at the top of your next observation.

## Structured Scratchpad (persistent across turns)
Your scratchpad has 6 independent sections. Each section PERSISTS across turns.
**Only include a section header to UPDATE that section.** Omitted sections keep their previous value.

Sections:
- STRATEGY: Your high-level goal. Update RARELY (only when strategy changes).
- PLAN: Numbered checklist of steps. Update when completing/adding steps.
- LESSONS: Game mechanics you've discovered. Update when you learn something new.
- TACTICAL: Your immediate situation and next move. Update EVERY turn.
- FLOOR_NOTES: Your exploration log for the CURRENT floor. Record what you find as you
  explore: dead ends and their coordinates, cleared vs unexplored rooms/corridors,
  dangerous areas with many enemies, resource-rich areas. This prevents backtracking
  into dead ends or re-exploring cleared areas. Update whenever you discover something.
  Example: "West corridor dead end at (15,3). East room has 3 orcs, unexplored. South fountain at (22,8)."
- FLOOR_CHECKLIST: What you need before descending to the next floor.

## Grounding Rules
- The ASCII map is ALWAYS the ground truth. Read it FIRST before consulting memory.
- Your spatial memory or scratchpad may be outdated — verify against the map.
- Your recent action history helps detect loops. If repeating actions, change strategy.

## IMPORTANT: Output Format
Your output is parsed by a system. It reads ONLY the final structured block after
the `---` delimiter. You can freely discuss actions in your reasoning above `---`
without them being executed.

Your response must follow this format:

## Assessment
<What do you see? Threats? Opportunities? 1-3 sentences REQUIRED>

## Decision
<Why this action? How does it advance your plan? 1-2 sentences REQUIRED>

---
Action: <ACTION_NAME>
MAP_ADD: floor=0, type=water, row=20, col=30, note=lake
MAP_REMOVE: floor=0, type=tree, row=22, col=25
TACTICAL: <your updated tactical note for this turn>
PLAN:
1. [x] Done step
2. [ ] Next step

Only `Action:` is required. MAP_* commands and scratchpad sections are optional.
Place MAP_* commands and section updates after Action."""
