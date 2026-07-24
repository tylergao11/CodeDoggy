"""Doggy neon couple splash art — single source of truth.

Copied verbatim from the legacy TUI brand path. Both ``codedoggy.tui`` and
``codedoggy.tui_v2`` must import from here so the two-dog portrait never drifts.
"""

from __future__ import annotations

import shutil
import time
from itertools import groupby

from prompt_toolkit.application.current import get_app
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth


def _terminal_height() -> int:
    try:
        return get_app().output.get_size().rows
    except Exception:  # noqa: BLE001
        return shutil.get_terminal_size(fallback=(100, 30)).lines


def _truncate_display(text: str, width: int) -> str:
    if get_cwidth(text) <= width:
        return text
    if width <= 1:
        return "…"
    out: list[str] = []
    used = 0
    for char in text:
        char_width = get_cwidth(char)
        if used + char_width > width - 1:
            break
        out.append(char)
        used += char_width
    return "".join(out).rstrip() + "…"


def _render_doggy_idle_panel(width: int) -> StyleAndTextTuples:
    """Post-splash empty task area with a calm text hierarchy."""
    w = max(12, width)
    lines: list[tuple[str, str]] = [
        ("class:brand", "  DOGGY"),
        ("class:task.title", "  散步完了，等你下一句。"),
        ("class:meta", "  在下方输入框交代任务…"),
    ]
    out: StyleAndTextTuples = []
    for style, raw in lines:
        text = _truncate_display(raw, w)
        pad = max(0, w - get_cwidth(text))
        out.append((style, text + " " * pad + "\n"))
    return out


# Startup brand: neon street couple (concept image) — black void, not car city.
# Keys: F fur · H golden fur · D dark cloth · B shades · M pink · C cyan ·
# Y gold · W white · P hot pink · N nose · L cream shoe · S soft · . void
_DOGGY_COUPLE_ART = (
    "....................................................",
    "........HHH.........HHHH............................",
    "........HHSF.......HHHHF............................",
    "........H..HH......FHHSH............................",
    "........HSH.HH.HHHHHSFHF............................",
    "........HFF.HFHHHHFFHHHF............................",
    "........HFF.HFFFFFFFFFH.............................",
    "........HHHHFFFFHHFFFHHH............................",
    "........FHHFFHHHHSHS....FH..........................",
    "........SHHHS.....H.....SS........HHHH.MMM..........",
    "........HHHHHH....FFFFFSHHHS....HHHHHHHMMM.M........",
    "........HHFFFH...HFLLFF...F....HHHFFFFHMMMMMM.......",
    ".......HFFFFFLFFFLLLLLFF..F...HHHFFFFFHHHHMM........",
    ".......HHFFFFLLLFFLLLLLLFFF...HHHFFFFFFFFHSSH.......",
    "........HFFFLLLLFHFFFFFFHFF...FHHHFFFHFFFFHHH.......",
    "..........HFFLLLLLFFFFFFFH...HHHFFFFHHHFFHFFFH......",
    ".........MSHFFLLLLLLLFFFFSM..HHFF.HFFFFFFHHFFH......",
    ".......MMMS..FFFFFFFF...MMMMSFHFFSFFFFLFHHFFFFH.....",
    ".......MMMS.SFFFFFFFF..MMMMCCSHFFHFFHFLFFHFFFFHH....",
    ".......MMHMMHFFFFFFFH.MMS.....HHFFFFFLLFHHFFFFFH....",
    ".......CC.SM.HFFFFFH.MM..........HFFFLFH..HFFFFH....",
    ".......C...MSHHFFF.HSS............HFFFS....HSHH.....",
    "......C.....SSH...HHS........CC..MSHSMMM.HFFFS......",
    ".....CC.....S.HH.HF.S..MM...CCHSSHHHMSHSHSHHHS......",
    ".....CC.....S..HHH.SS..MMM..CSH.HFHHHFH.FHHFFFH.....",
    ".....C..C...S..SSS.CC.MMMMS.CHH.HFFFFFHHFHFHFFF.....",
    "....C...CC..M..HSH.CC..MMM..CHS.HHHHHH.FH.HHFHF.....",
    "...CC...CC..M..HHH.CC..SSS..CCS..SSM....H..HHHH.....",
    "..CC....CC..M......CC.......CC...MMMM...HC.HH.C.....",
    "..C.....C..S.......CC.......CCC..MMM.....CC...CC....",
    "..C..CCCC..S.......CC.......C.C...M.....CCC....C....",
    "..C....C...M.......CC.......C.C.........C.C....CC...",
    "..C.......SM.......CC.......C.CSHHHHHHHHC.CC....C...",
    "..CC..MM..S.FFFF...CC.......C.CHFFFFFFFFSC.C....CC..",
    "..CC.SMSHSS.FFFFHFHCC......SS.CCSSSSHHHSCC.C.....C..",
    "...C.MHFH...........M...MSSS..CCCCCCCCCCCC.CC....C..",
    "....CSHFS....C.C....SS..MSSS.C...CCC.....C..C...CC..",
    ".....CSH.....CC.......SMS....C.C...C..C..CC.CCCCC...",
    ".......C......C......CC.C...CC.C...C..S...C..CSSH...",
    ".......C.....CC........CC..CC.S...C....C..CC..HHFH..",
    ".......CC....C.........CC..S..S...S....S...SS.FHFH..",
    "........C....C........CC..SSS.S...S....M..SSM.HSFH..",
    "........CC..C......S..C...SMMMMMMSMMMMSMMMMMM...HH..",
    ".........C..C......C..C......MMHMSSHHHHSMHM..H......",
    ".........CCCC.....CC.C.........H..FFFFFF....HH......",
    "..........CC.........CC........HHSFFFFFH.HHHFH......",
    "..........SC......C.C.CC..HH...HHHFFFFFS.HFFFS......",
    "..........CC.....C.CC..CFFFFH...HHFFFFH.C.HFH.......",
    "..........C......C.C...CCHFH.....FFFFFH.SS..........",
    ".........CC.....C..C....CHH......FFFFF...CS.........",
    "........CCCCC...C.CC...CSH......SFFFFH..CCSH........",
    "........C....CC...C...CC..H....CSSHFFH.CC..H........",
    ".......FFFFH..SC.CC.CCC..FF.....FCCSH.SC...F........",
    "......FFHHFFH..CSC..SH...F...HFFF..CS..H..HH........",
    "......F.....FF..CC.HH...FF...F..FF..S.HS..H.....M...",
    "......HS....FF..C.HS....F....F...HF.SS...HH.....M...",
    "CCCCC..HH....HLHS..FFFFFH.SS.HH...FHSSFHFH.CCSSCCCCC",
    "........SH....HF...SSSSS......SH...FF.SSS...........",
    "MM.C.C...HFFFFFF..CCCCCCCCC....SFFFF...S...MM.SSS.MM",
    "....................................................",
)

# ``.`` is intentionally absent: empty pixels inherit the active Window/theme
# background instead of painting a second canvas color behind the portrait.
_DOGGY_ART_PALETTE = {
    "C": "#00bac5",
    "M": "#ee4b8d",
    "c": "#0b6670",
    "m": "#8f1b58",
    "G": "#ff7a32",
    "Y": "#d9ad32",
    "T": "#f2ca55",
    "P": "#ff68ad",
    "R": "#0a0a0a",
    "F": "#e1d2ae",
    "H": "#c9a978",
    "D": "#1a1a1a",
    "S": "#75644a",
    "W": "#f5f5f7",
    "B": "#121212",
    "N": "#3a2a22",
    "L": "#f0e6cc",
    "K": "#050507",
    "E": "#3b2a20",
}

_DOGGY_COUPLE_FRAMES = 12

# High-priority facial pixels survive terminal resizing instead of dissolving
# into the surrounding tan fur.
_DOGGY_FEMALE_EYE_DETAILS = (
    (32, 13, "K"), (33, 13, "K"), (34, 13, "K"),
    (32, 14, "K"), (33, 14, "W"), (34, 14, "K"),
    (38, 13, "K"), (39, 13, "K"), (40, 13, "K"),
    (38, 14, "K"), (39, 14, "W"), (40, 14, "K"),
)

_DOGGY_FEMALE_MASK_SPANS = (
    (16, 33, 40),
    (17, 31, 42),
    (18, 31, 42),
    (19, 31, 42),
    (20, 33, 40),
)

_DOGGY_FEMALE_MASK_HIGHLIGHTS = (
    (29, 17, "m"), (30, 17, "m"),
    (43, 17, "m"), (44, 17, "m"),
    (34, 18, "P"), (35, 18, "P"), (36, 18, "P"),
    (37, 18, "P"), (38, 18, "P"), (39, 18, "P"),
)

_DOGGY_FEMALE_CROWN_SPANS = (
    (7, 36, 42, "H"),
    (8, 34, 44, "H"),
    (9, 32, 45, "H"),
)

_DOGGY_FEMALE_BOW_DETAILS = (
    (42, 7, "M"), (43, 7, "M"), (46, 7, "M"), (47, 7, "M"),
    (42, 8, "M"), (43, 8, "M"), (44, 8, "M"),
    (45, 8, "P"),
    (46, 8, "M"), (47, 8, "M"), (48, 8, "M"),
    (43, 9, "M"), (44, 9, "M"), (45, 9, "P"),
    (46, 9, "M"), (47, 9, "M"),
)

_DOGGY_CHAIN_DETAILS = (
    (17, 20), (18, 21), (19, 22),
    (23, 20), (22, 21), (21, 22),
    (20, 23), (36, 23),
)


def _animate_doggy_couple(rows: tuple[str, ...], frame: int) -> tuple[str, ...]:
    """Keep the portrait still while tiny jewellery and bow highlights breathe."""

    canvas = [list(row) for row in rows]
    height = len(canvas)
    width = len(canvas[0]) if canvas else 0
    phase = frame % _DOGGY_COUPLE_FRAMES

    for y, start, end, value in _DOGGY_FEMALE_CROWN_SPANS:
        if 0 <= y < height:
            for x in range(start, min(end + 1, width)):
                canvas[y][x] = value

    for x, y, value in _DOGGY_FEMALE_BOW_DETAILS:
        if 0 <= y < height and 0 <= x < width:
            canvas[y][x] = value

    for x, y, value in _DOGGY_FEMALE_EYE_DETAILS:
        if 0 <= y < height and 0 <= x < width:
            canvas[y][x] = value

    for y, start, end in _DOGGY_FEMALE_MASK_SPANS:
        if 0 <= y < height:
            for x in range(start, min(end + 1, width)):
                canvas[y][x] = "m" if x in {start, end} else "M"

    for x, y, value in _DOGGY_FEMALE_MASK_HIGHLIGHTS:
        if 0 <= y < height and 0 <= x < width:
            canvas[y][x] = value

    for index, (x, y) in enumerate(_DOGGY_CHAIN_DETAILS):
        if 0 <= y < height and 0 <= x < width:
            canvas[y][x] = "T" if index == phase % len(_DOGGY_CHAIN_DETAILS) else "Y"

    bow_pixels = [
        (x, y)
        for y in range(min(14, height))
        for x in range(38, width)
        if canvas[y][x] == "M"
    ]
    if bow_pixels:
        x, y = bow_pixels[(phase // 2) % len(bow_pixels)]
        canvas[y][x] = "P"

    return tuple("".join(row) for row in canvas)


def _compose_doggy_night(
    art_rows: tuple[str, ...],
    width: int,
    scene_time: float,
) -> tuple[str, ...]:
    """Place the locked portrait in the reference image's sparse neon night."""

    height = max(len(art_rows), 2)
    if height % 2:
        height += 1
    scene_width = max(1, width)
    tick = int(scene_time * 5)
    canvas = [["."] * scene_width for _ in range(height)]

    def put(x: int, y: int, value: str, *, soft: bool = False) -> None:
        if 0 <= x < scene_width and 0 <= y < height:
            if soft and canvas[y][x] != ".":
                return
            canvas[y][x] = value

    # Pink crescent from the reference, left of the taller dog.
    moon_x = max(2, round(scene_width * 0.25))
    moon_y = max(1, round(height * 0.09))
    for dy, span in ((0, (1, 2)), (1, (0, 3)), (2, (0, 3)), (3, (1, 2))):
        for dx in range(span[0], span[1] + 1):
            put(moon_x + dx, moon_y + dy, "M" if (tick // 3) % 2 == 0 else "m")
    put(moon_x + 2, moon_y + 1, ".")
    put(moon_x + 2, moon_y + 2, ".")

    # Sparse pink/cyan stars; only their intensity changes, never their position.
    spark_seed = (
        (0.20, 0.25, "C", True),
        (0.29, 0.17, "M", False),
        (0.75, 0.13, "M", False),
        (0.72, 0.29, "C", True),
        (0.14, 0.38, "C", False),
        (0.76, 0.48, "M", False),
        (0.18, 0.62, "M", False),
        (0.70, 0.70, "M", False),
        (0.24, 0.78, "C", False),
        (0.83, 0.58, "C", False),
    )
    for i, (fx, fy, color, cross) in enumerate(spark_seed):
        if (tick + i) % 5 == 0:
            continue
        x = int(fx * (scene_width - 1))
        y = int(fy * (height - 1))
        dim = "c" if color == "C" else "m"
        sparkle = color if (tick + i) % 2 == 0 else dim
        put(x, y, sparkle, soft=True)
        if cross and (tick + i) % 3:
            put(x - 1, y, sparkle, soft=True)
            put(x + 1, y, sparkle, soft=True)
            put(x, y - 1, sparkle, soft=True)
            put(x, y + 1, sparkle, soft=True)

    # The couple never bobs: its approved 52x60 height and pose stay locked.
    art_width = len(art_rows[0]) if art_rows else 0
    art_height = len(art_rows)
    art_left = max(0, (scene_width - art_width) // 2)
    art_top = max(0, (height - art_height) // 2)
    for y, row in enumerate(art_rows):
        ty = art_top + y
        if ty >= height:
            break
        for x, value in enumerate(row):
            if value != ".":
                put(art_left + x, ty, value)

    return tuple("".join(row) for row in canvas)


_DOGGY_DESIGN_WIDTH = 120
_DOGGY_DESIGN_TOP_MARGIN = 1
_DOGGY_DESIGN_BOTTOM_MARGIN = 1


def _render_doggy_empty(
    width: int,
    *,
    now: float | None = None,
) -> StyleAndTextTuples:
    """Render the locked 52x60 neon couple portrait and sparse night field."""
    try:
        clock = time.monotonic() if now is None else now
        art_tick = int(clock * 5)
        frame = art_tick % _DOGGY_COUPLE_FRAMES
        rows = _animate_doggy_couple(_DOGGY_COUPLE_ART, frame)
        terminal_height = _terminal_height()

        task_height = max(1, terminal_height - 8)
        # Keep the portrait on its native pixel grid. Fractional nearest-neighbour
        # scaling changes which eye/mask rows survive as the terminal is resized.
        stage_width = max(
            1,
            min(width, _DOGGY_DESIGN_WIDTH),
        )

        target_width = max(1, min(len(rows[0]), stage_width - 4))
        if target_width < len(rows[0]):
            crop_left = max(0, (len(rows[0]) - target_width) // 2)
            rows = tuple(row[crop_left : crop_left + target_width] for row in rows)

        rows = _compose_doggy_night(rows, stage_width, clock)
        if len(rows) % 2:
            rows = rows + (rows[-1] if rows else "." * max(1, stage_width),)
        art_width = len(rows[0])
        outer = max(0, (width - art_width) // 2)
        palette = dict(_DOGGY_ART_PALETTE)

        art_height = len(rows) // 2
        top_margin = _DOGGY_DESIGN_TOP_MARGIN
        bottom_margin = _DOGGY_DESIGN_BOTTOM_MARGIN
        scaled_frame_height = top_margin + art_height + bottom_margin
        vertical_slack = max(0, task_height - scaled_frame_height)
        top_padding = top_margin + vertical_slack // 2

        fragments: StyleAndTextTuples = [("", "\n" * top_padding)]
        for top, bottom in zip(rows[::2], rows[1::2]):
            fragments.append(("", " " * outer))
            pairs = zip(top, bottom)
            for pair, cells in groupby(pairs):
                count = sum(1 for _ in cells)
                style, glyph = _half_block(pair[0], pair[1], palette)
                fragments.append((style, glyph * count))
            fragments.append(("", "\n"))
        return fragments if fragments else [("", "\n")]
    except Exception:  # noqa: BLE001
        # Splash must never take down the whole TUI paint path.
        return _render_doggy_idle_panel(max(1, width))


def _half_block(
    top: str,
    bottom: str,
    palette: dict[str, str],
) -> tuple[str, str]:
    if top == bottom == ".":
        return "", " "
    top_color = palette.get(top, "#000000")
    bottom_color = palette.get(bottom, "#000000")
    if top == bottom:
        return f"fg:{top_color}", "█"
    if top == ".":
        return f"fg:{bottom_color}", "▄"
    if bottom == ".":
        return f"fg:{top_color}", "▀"
    return f"fg:{top_color} bg:{bottom_color}", "▀"


