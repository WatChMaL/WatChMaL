#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
banner.py — animated startup banner for WatChMaL / Hyper-Kamiokande.

A neutrino (ν) comes in from the left along the beam line, enters the
cylindrical tank, interacts, and the Cherenkov ring lights up on the PMT wall.
The animation plays "in place" (cursor moved back up, like tqdm) then freezes
on a static banner that the user keeps in view.

Minimal usage
-------------
    from watchmal.utils.banner import HyperKBanner

    HyperKBanner(
        name="WatChMaL",
        subtitle="Hyper-Kamiokande · multi-ring segmentation",
        info={"engine": "multiring/segmentation",
              "device": "cuda:0",
              "params": "2 285 344",
              "epochs": 5},
    ).play()

Automatic degradations (important on a cluster)
-----------------------------------------------
  * non-TTY stdout (Slurm log, file redirection) -> static banner, no ANSI
  * NO_COLOR / TERM=dumb                          -> no color
  * HK_BANNER=0                                    -> nothing at all
  * terminal too narrow                           -> geometry reduced automatically
"""

from __future__ import annotations

import atexit
import logging
import math
import os
import shutil
import sys
import threading
import time

# --------------------------------------------------------------------------- #
#  ANSI sequences
# --------------------------------------------------------------------------- #
CSI = "\x1b["
RESET = CSI + "0m"
HIDE_CURSOR = CSI + "?25l"
SHOW_CURSOR = CSI + "?25h"
CLEAR_LINE = CSI + "2K"
ERASE_DOWN = CSI + "0J"      # erase from the cursor to the end of the screen
SAVE_CURSOR = "\x1b7"        # DECSC / DECRC: more portable under tmux than CSI s/u
RESTORE_CURSOR = "\x1b8"


def _scroll_region(top: int, bottom: int) -> str:
    """DECSTBM: confine scrolling to rows [top, bottom]. Also homes the cursor."""
    return f"{CSI}{top};{bottom}r"


def _reset_scroll_region() -> str:
    return f"{CSI}r"


def _goto(row: int, col: int = 1) -> str:
    return f"{CSI}{row};{col}H"


class _SerialisedStream:
    """Stream proxy that takes a lock around each write.

    In split-screen mode two writers share the terminal: the logging handlers (from
    the main thread) and the banner thread. Without a shared lock, a log line can land
    between the banner's "position cursor / draw / restore cursor" sequence and end up
    painted inside the animation area. Wrapping the handlers' stream is enough to
    serialise them, and it is undone on stop().
    """

    def __init__(self, stream, lock: threading.RLock):
        self._stream = stream
        self._lock = lock

    def write(self, data):
        with self._lock:
            return self._stream.write(data)

    def flush(self):
        with self._lock:
            return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)

# --------------------------------------------------------------------------- #
#  "Water / Cherenkov" palette
# --------------------------------------------------------------------------- #
C_FRAME = (48, 86, 132)      # tank structure
C_TEXT = (150, 190, 220)     # secondary text
C_TITLE = (140, 240, 255)    # title
C_NU = (245, 255, 255)       # the neutrino itself

# PMT intensity ramp: (threshold, glyph, color)
PMT_LEVELS = (
    (0.12, "·", (30, 52, 86)),      # off
    (0.30, "∙", (44, 96, 150)),
    (0.52, "○", (60, 150, 205)),
    (0.74, "●", (95, 215, 235)),
    (0.90, "◉", (180, 250, 255)),
    (9.99, "◉", (255, 255, 255)),   # saturated core
)


def _fg(rgb) -> str:
    return f"{CSI}38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _ease_in(t: float) -> float:
    """The neutrino accelerates as it approaches the tank."""
    return t * t * (3.0 - 2.0 * t) * 0.35 + t * 0.65


# Layout floors: below these the animation is not worth attempting.
_MIN_BEAM = 8        # columns of beam line left of the tank
_MIN_ROWS = 7        # PMT rows

# Banners currently holding a scroll region. A process that dies without calling stop()
# would otherwise leave the terminal permanently confined to the top rows, which looks
# like a broken shell and needs `reset` to fix.
_ACTIVE_SPLITS: set = set()


@atexit.register
def _restore_split_terminals() -> None:
    for banner in list(_ACTIVE_SPLITS):
        try:
            banner._exit_split()
        except Exception:
            pass


class _HeldLogs(logging.Filter):
    """Buffer log records while the animation owns the terminal, replay them after.

    The async banner repaints in place (cursor up N lines). Anything else writing to
    the same terminal in the meantime desynchronises that cursor arithmetic and the
    frames smear into each other - and loading is precisely when the dataset modules
    are chatty. So records are held for the duration and emitted afterwards, in order,
    with their original timestamps intact.

    Attached to the root *handlers*, not the root logger: a record propagating up from
    a module logger bypasses ancestor loggers' filters but still goes through their
    handlers. Same mechanism as utils.distributed_utils.restrict_logging_to_rank0.
    """

    def __init__(self, min_level: int = logging.ERROR):
        super().__init__()
        self.min_level = min_level
        self.records: list[logging.LogRecord] = []
        self._lock = threading.Lock()
        self._handlers: list[logging.Handler] = []

    def filter(self, record: logging.LogRecord) -> bool:
        # Errors still get through: never hide a failure behind an animation.
        if record.levelno >= self.min_level:
            return True
        with self._lock:
            self.records.append(record)
        return False

    def install(self) -> "_HeldLogs":
        self._handlers = list(logging.getLogger().handlers)
        for handler in self._handlers:
            handler.addFilter(self)
        return self

    def release(self) -> None:
        for handler in self._handlers:
            handler.removeFilter(self)
        with self._lock:
            held, self.records = self.records, []
        for record in held:
            for handler in self._handlers:
                if record.levelno >= handler.level:
                    handler.handle(record)
        self._handlers = []


# --------------------------------------------------------------------------- #
#  Banner
# --------------------------------------------------------------------------- #
class HyperKBanner:
    def __init__(
        self,
        name: str = "WatChMaL",
        subtitle: str = "Hyper-Kamiokande · deep learning framework",
        info: dict | None = None,
        cols: int = 9,           # PMT columns (tank narrower than tall)
        rows: int = 22,          # PMT rows (tank taller than wide)
        gap: int = 26,           # minimum beam line length, on the left
        fps: int = 45,
        stream=None,
        force_animation: bool = False,
        force_color: bool | None = None,
        term_size: tuple[int, int] | None = None,
        enabled: bool = True,
        min_log_lines: int = 10,
    ):
        self.name = name
        self.subtitle = subtitle
        self.info = info or {}
        self.fps = fps
        self.stream = stream or sys.stdout
        self.min_log_lines = min_log_lines

        # Requested geometry. The effective one is derived from the terminal in _fit();
        # these stay as the "ideal" values so a resize can grow back toward them.
        self._req_cols = cols
        self._req_rows = rows
        self._min_gap = gap
        self._forced_term_size = term_size

        env_enabled = enabled and os.environ.get("HK_BANNER", "1") != "0"
        tty = force_animation or (hasattr(self.stream, "isatty") and self.stream.isatty())
        self.animate = env_enabled and tty
        self.enabled = env_enabled

        if force_color is None:
            self.color = tty and not os.environ.get("NO_COLOR") \
                and os.environ.get("TERM") != "dumb"
        else:
            self.color = force_color

        # async state (see start()/stop())
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._log_hold: _HeldLogs | None = None
        self._nlines = 0

        # split-screen state (logs on top, animation pinned to the bottom)
        self._split = False
        self._want_split = False
        self._io_lock = threading.RLock()
        self._wrapped_handlers: list[tuple[logging.Handler, object]] = []

        # Status shown inside the frame. Terminal-only by construction: it is drawn,
        # never logged, so it reaches neither main.log nor wandb.
        self._status = ""

        self._term_size = (0, 0)
        self._fit()

    # ------------------------------------------------------------------ #
    #  Status ("what are we waiting for")
    # ------------------------------------------------------------------ #
    def set_status(self, text: str) -> None:
        """Set the message shown with the animation. Safe to call from any thread.

        This is drawn into the frame and nowhere else - it is deliberately NOT a log
        record, so it never reaches main.log, a file handler, or wandb. It exists only
        for as long as the animation is on screen.
        """
        self._status = text or ""

    # ------------------------------------------------------------------ #
    #  Fitting the terminal
    # ------------------------------------------------------------------ #
    def _measure(self) -> tuple[int, int]:
        if self._forced_term_size is not None:
            return self._forced_term_size
        size = shutil.get_terminal_size((100, 30))
        return size.columns, size.lines

    def _fit(self) -> bool:
        """Lay the banner out for the current terminal size.

        Width: the tank is pinned as far RIGHT as it fits and the neutrino always
        starts at column 0, so the beam line is whatever space is left in between -
        it grows on a wide terminal instead of staying at a fixed 26 columns. The tank
        itself only shrinks when the terminal is too narrow to hold it plus a usable
        beam.

        Height: the frame is clamped to the terminal so the in-place redraw (cursor up
        N lines) can never be defeated by scrolling, which is what turns the animation
        into a stack of half-drawn frames.

        Returns True if the geometry changed, so a running animation can redraw.
        """
        term_w, term_h = self._measure()
        if (term_w, term_h) == self._term_size:
            return False
        self._term_size = (term_w, term_h)

        # -- width: shrink the tank only if it cannot coexist with a usable beam --
        usable_w = max(20, term_w - 1)          # -1: never write the last column
        self.cols = self._req_cols
        while self.cols > 3 and self.box_width + _MIN_BEAM > usable_w:
            self.cols -= 1
        self.gap = max(_MIN_BEAM, usable_w - self.box_width)

        # -- height: title + blank + tank(+2 borders) + blank + status + blank + info --
        self._title = self._build_title()
        self._info_lines = self._build_info()
        overhead = 1 + len(self._title) + 1 + 2 + 1 + 1 + 1 + len(self._info_lines)
        # In split-screen mode the frame shares the terminal with the scrolling log
        # region, so it may not take the whole height: the tank gives up rows first.
        budget = term_h - 1 - (self.min_log_lines if self._want_split else 0)
        self.rows = max(_MIN_ROWS, min(self._req_rows, budget - overhead))

        # -- interaction geometry: vertex at the tank center --
        self.vx = (self.cols - 1) / 2.0
        self.vy = self.rows // 2
        self._vrow = self.rows // 2          # beam row (PMT index)
        half_h = (self.rows - 1) / 2.0
        half_w = (self.cols - 1) / 2.0
        self.r_max = min(half_w, half_h) * 1.02
        self.r_final = self.r_max * 0.85

        # Title/info are centred on the full width, so rebuild once the width is known.
        self._title = self._build_title()
        self._info_lines = self._build_info()

        # Truly too small for an in-place animation: fall back to the static frame.
        needed_h = overhead + _MIN_ROWS + 1 + (self.min_log_lines if self._want_split else 0)
        if self.box_width + _MIN_BEAM > usable_w or term_h < needed_h:
            self.animate = False
        return True

    @property
    def frame_height(self) -> int:
        """Number of terminal lines one frame occupies."""
        return 1 + len(self._title) + 1 + (self.rows + 2) + 1 + 1 + 1 + len(self._info_lines)

    # ------------------------------------------------------------------ #
    #  Geometry
    # ------------------------------------------------------------------ #
    @property
    def inner_width(self) -> int:
        """Inner width of the tank, in characters."""
        return self.cols * 2 - 1

    @property
    def box_width(self) -> int:
        return self.inner_width + 4          # "│ " + content + " │"

    @property
    def total_width(self) -> int:
        return self.gap + self.box_width

    # ------------------------------------------------------------------ #
    #  Rendering helpers
    # ------------------------------------------------------------------ #
    def _c(self, text: str, rgb) -> str:
        return f"{_fg(rgb)}{text}{RESET}" if self.color else text

    def _center(self, text: str, rgb=C_TEXT) -> str:
        # Truncate rather than overflow: on a narrow terminal an over-long subtitle
        # would wrap, and a wrapped line breaks the cursor-up repaint arithmetic.
        if len(text) > self.total_width:
            text = text[: max(1, self.total_width - 1)] + "…"
        pad = max(0, (self.total_width - len(text)) // 2)
        return " " * pad + self._c(text, rgb)

    def _build_title(self) -> list[str]:
        """Framed title. Uses pyfiglet if installed and it fits."""
        art = None
        try:
            import pyfiglet  # optional
            candidate = pyfiglet.figlet_format(self.name, font="slant").rstrip("\n").split("\n")
            candidate = [ln for ln in candidate if ln.strip()]
            if candidate and max(len(ln) for ln in candidate) <= self.total_width:
                art = candidate
        except Exception:
            art = None

        if art is not None:
            lines = [self._center(ln, C_TITLE) for ln in art]
            lines.append(self._center(self.subtitle, C_TEXT))
            return lines

        # fallback: spaced-out title in a double frame
        label = " ".join(self.name.upper())
        w = self.total_width - 2
        top = "╔" + "═" * w + "╗"
        mid = "║" + label.center(w) + "║"
        bot = "╚" + "═" * w + "╝"
        return [
            self._c(top, C_FRAME),
            self._c("║", C_FRAME) + self._c(label.center(w), C_TITLE) + self._c("║", C_FRAME),
            self._c(bot, C_FRAME),
            self._center(self.subtitle, C_TEXT),
        ]

    def _build_info(self) -> list[str]:
        if not self.info:
            return [""]
        parts = [f"{k} : {v}" for k, v in self.info.items()]
        lines, cur = [], ""
        for p in parts:
            cand = p if not cur else cur + "   ·   " + p
            if len(cand) > self.total_width - 4:
                lines.append(cur)
                cur = p
            else:
                cur = cand
        if cur:
            lines.append(cur)
        return [self._center(ln, C_TEXT) for ln in lines]

    # ------------------------------------------------------------------ #
    #  The detector
    # ------------------------------------------------------------------ #
    def _pmt_intensity(self, c: int, r: int, radius: float, sigma: float,
                       amp: float, flash: float) -> float:
        base = 0.05
        if amp <= 0.0:
            return base + flash
        d = math.hypot(c - self.vx, r - self.vy)
        # bright wavefront propagating outward…
        ring = amp * math.exp(-((d - radius) ** 2) / (2.0 * sigma * sigma))
        # …with a glow filling the interior from the center
        if d < radius:
            glow = amp * 0.32 * (1.0 - 0.5 * d / max(radius, 1e-6))
            ring = max(ring, glow)
        return base + ring + flash

    def _detector_lines(self, radius: float, sigma: float, amp: float,
                        flash: float, nu_col: float | None = None) -> list[str]:
        top = "╭" + "─" * (self.inner_width + 2) + "╮"
        bot = "╰" + "─" * (self.inner_width + 2) + "╯"
        out = [self._c(top, C_FRAME)]
        bar = self._c("│", C_FRAME)
        for r in range(self.rows):
            cells = []
            for c in range(self.cols):
                i = self._pmt_intensity(c, r, radius, sigma, amp, flash)
                for thr, glyph, rgb in PMT_LEVELS:
                    if i < thr:
                        cells.append(self._c(glyph, rgb))
                        break
            # the neutrino crosses the tank to the center
            if nu_col is not None and r == self._vrow:
                ci = int(round(nu_col))
                for k in range(1, 4):                 # short trail
                    p = ci - k
                    if 0 <= p < self.cols:
                        f = 1.0 - k / 4.0
                        rgb = (int(40 + 180 * f), int(80 + 170 * f), int(110 + 145 * f))
                        cells[p] = self._c("∙", rgb)
                if 0 <= ci < self.cols:
                    cells[ci] = self._c("ν", C_NU)
            out.append(bar + " " + " ".join(cells) + " " + bar)
        out.append(self._c(bot, C_FRAME))
        return out

    # ------------------------------------------------------------------ #
    #  The beam line
    # ------------------------------------------------------------------ #
    def _beam_line(self, nu_x: float | None) -> str:
        cells = [" "] * self.gap        # no more dotted line: empty background
        if nu_x is not None:
            xi = int(round(nu_x))
            # trail: fainter and fainter toward the back
            for k in range(1, 9):
                p = xi - k
                if 0 <= p < self.gap:
                    f = 1.0 - k / 9.0
                    rgb = (int(30 + 200 * f), int(60 + 190 * f), int(90 + 165 * f))
                    cells[p] = self._c("∙" if k < 4 else "·", rgb)
            if 0 <= xi < self.gap:
                cells[xi] = self._c("ν", C_NU)
        return "".join(cells)

    def _blank_field(self) -> str:
        return " " * self.gap

    # ------------------------------------------------------------------ #
    #  Composing a full frame
    # ------------------------------------------------------------------ #
    def _frame(self, nu_x, radius, sigma, amp, flash, status: str,
               nu_col=None) -> list[str]:
        det = self._detector_lines(radius, sigma, amp, flash, nu_col)
        beam_row = 1 + self._vrow              # +1 for the top border

        body = []
        for i, dline in enumerate(det):
            left = self._beam_line(nu_x) if i == beam_row else self._blank_field()
            body.append(left + dline)

        lines = [""]
        lines += self._title
        lines.append("")
        lines += body
        lines.append("")
        lines.append(self._center(status, C_TEXT))
        lines.append("")
        lines += self._info_lines
        return lines

    def static_frame(self) -> list[str]:
        """The final frozen state: the ring stays lit."""
        return self._frame(
            nu_x=None,
            radius=self.r_final,
            sigma=0.95,
            amp=0.72,
            flash=0.0,
            status="",
        )

    # ------------------------------------------------------------------ #
    #  Output
    # ------------------------------------------------------------------ #
    def _draw(self, lines: list[str], first: bool) -> None:
        """Repaint in place: back up over the previous frame, wipe it, redraw.

        Erasing to the end of the screen (rather than clearing line by line) is what
        lets the frame change height - on a terminal resize the new frame simply
        replaces the old one instead of leaving a tail of orphaned rows behind.
        """
        if self._split:
            return self._draw_pinned(lines)
        buf = [] if first else [f"{CSI}{self._nlines}A", ERASE_DOWN]
        buf += [ln + "\n" for ln in lines]
        self._nlines = len(lines)
        with self._io_lock:
            self.stream.write("".join(buf))
            self.stream.flush()

    def _draw_pinned(self, lines: list[str]) -> None:
        """Paint the frame into the bottom region, leaving the log cursor untouched.

        Absolute positioning, one line at a time, and no trailing newline: writing a
        newline on the last row would scroll the screen. The cursor is saved and
        restored around the whole frame so the logging handlers keep appending exactly
        where they left off, inside the scroll region above.
        """
        term_h = self._term_size[1]
        top = max(1, term_h - len(lines) + 1)
        buf = [SAVE_CURSOR]
        for i, line in enumerate(lines):
            buf.append(_goto(top + i) + CLEAR_LINE + line)
        buf.append(RESTORE_CURSOR)
        with self._io_lock:
            self.stream.write("".join(buf))
            self.stream.flush()

    # ------------------------------------------------------------------ #
    #  Split screen: logs scroll on top, animation pinned to the bottom
    # ------------------------------------------------------------------ #
    def _enter_split(self) -> None:
        """Reserve the bottom rows for the animation and confine scrolling above.

        DECSTBM (`CSI top;bottom r`) makes the terminal scroll only within the log
        region, so every line the run logs from here on stays in the top part and the
        animation below is never pushed around. Blank lines are printed first so the
        reserved rows are genuinely free - existing output scrolls up instead of being
        overwritten.
        """
        term_h = self._term_size[1]
        n = self.frame_height
        with self._io_lock:
            self.stream.write("\n" * n)
            self.stream.write(_scroll_region(1, term_h - n))
            self.stream.write(_goto(term_h - n))     # cursor at the end of the log area
            self.stream.write(SAVE_CURSOR)
            self.stream.flush()
        self._split = True
        _ACTIVE_SPLITS.add(self)

    def _resync_split(self) -> None:
        """Re-establish the scroll region after a resize (the frame height changed)."""
        if not self._split:
            return
        term_h = self._term_size[1]
        with self._io_lock:
            self.stream.write(_scroll_region(1, max(1, term_h - self.frame_height)))
            self.stream.write(RESTORE_CURSOR)
            self.stream.flush()

    def _exit_split(self) -> None:
        """Give the terminal back: drop the region and wipe the animation area."""
        if not self._split:
            return
        self._split = False
        _ACTIVE_SPLITS.discard(self)
        with self._io_lock:
            # Reset first (DECSTBM homes the cursor), then go back to the log cursor
            # and erase everything below it - which is exactly the animation.
            self.stream.write(_reset_scroll_region())
            self.stream.write(RESTORE_CURSOR)
            self.stream.write("\n" + ERASE_DOWN)
            self.stream.flush()

    def _wrap_log_handlers(self) -> None:
        lock = self._io_lock
        for handler in logging.getLogger().handlers:
            stream = getattr(handler, "stream", None)
            if stream is not None and not isinstance(stream, _SerialisedStream):
                self._wrapped_handlers.append((handler, stream))
                handler.stream = _SerialisedStream(stream, lock)

    def _unwrap_log_handlers(self) -> None:
        for handler, stream in self._wrapped_handlers:
            handler.stream = stream
        self._wrapped_handlers = []

    # ------------------------------------------------------------------ #
    #  Animation phases (shared by the blocking and the async player)
    # ------------------------------------------------------------------ #
    T_APPROACH, T_BURST = 0.52, 0.78          # phase bounds, as a fraction of a cycle

    def _phase(self, u: float):
        """Frame parameters at normalised cycle time u in [0, 1]."""
        if u < self.T_APPROACH:                          # --- approach ---
            p = u / self.T_APPROACH
            # the neutrino travels along the beam then continues, invisible, behind
            # the wall to the central vertex: the interaction only starts when it
            # reaches the center
            total = self.gap + self.vx
            pos = _ease_in(p) * total
            nu_x = pos if pos < self.gap else None
            return nu_x, 0.0, 1.0, 0.0, 0.0, None

        if u < self.T_BURST:                             # --- interaction ---
            p = (u - self.T_APPROACH) / (self.T_BURST - self.T_APPROACH)
            radius = self.r_max * (p ** 0.55)
            sigma = 0.75 + 0.5 * p
            # no global flash: the wave starts from the center
            return None, radius, sigma, 1.15, 0.0, None

        p = (u - self.T_BURST) / (1.0 - self.T_BURST)    # --- stabilization ---
        radius = self.r_max + (self.r_final - self.r_max) * p
        sigma = 1.25 - 0.3 * p
        amp = 1.15 + (0.72 - 1.15) * p
        return None, radius, sigma, amp, 0.0, None

    def _frame_at(self, u: float) -> list[str]:
        nu_x, radius, sigma, amp, flash, nu_col = self._phase(u)
        # The status sits just under the tank: closest to the animation it describes,
        # and next to the log region in split mode.
        return self._frame(nu_x, radius, sigma, amp, flash, self._status, nu_col)

    def _static_fallback(self) -> None:
        self.stream.write("\n".join(self.static_frame()) + "\n")
        self.stream.flush()

    def play(self, duration: float = 3.2) -> None:
        """Play one cycle, then freeze the banner. Blocking. No effect if non-TTY."""
        if not self.enabled:
            return
        if not self.animate:
            self._static_fallback()
            return

        n = max(24, int(duration * self.fps))
        dt = 1.0 / self.fps
        self._nlines = 0
        first = True

        if self.color:
            self.stream.write(HIDE_CURSOR)
        try:
            for k in range(n):
                if self._fit():          # terminal resized mid-animation
                    first = True
                self._draw(self._frame_at(k / (n - 1)), first)
                first = False
                time.sleep(dt)
            self._draw(self.static_frame(), False)       # clean frozen state
        except KeyboardInterrupt:
            pass
        finally:
            if self.color:
                self.stream.write(SHOW_CURSOR)
            self.stream.write("\n")
            self.stream.flush()

    # ------------------------------------------------------------------ #
    #  Async: run the animation while something else does the real work
    # ------------------------------------------------------------------ #
    def start(self, cycle: float = 3.2, hold: float = 0.7, split: bool = True):
        """Start animating in a background thread and return immediately.

        The animation loops (a new neutrino arrives after each ring settles) for as
        long as the caller takes, which is the point: the wait is of unknown length.
        Call stop() - or use the object as a context manager - when the work is done.

        split: share the terminal instead of owning it. The top rows keep scrolling
        the run's normal log output, so the user can watch what is happening, and the
        animation is pinned to the bottom rows. Falls back to buffering the log records
        and replaying them on stop() when the terminal is too short to split.

        A no-op that still prints the static banner when stdout is not a TTY, so the
        cluster path is unchanged.
        """
        if not self.enabled or self._thread is not None:
            return self

        # Re-fit: in split mode the frame has to leave room for the log region.
        self._want_split = split
        self._term_size = (0, 0)
        self._fit()

        if not self.animate:
            self._static_fallback()
            return self

        if split:
            self._wrap_log_handlers()
            self._enter_split()
        else:
            self._log_hold = _HeldLogs().install()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(cycle, hold), name="hk-banner", daemon=True
        )
        self._thread.start()
        return self

    def _run(self, cycle: float, hold: float) -> None:
        n = max(24, int(cycle * self.fps))
        dt = 1.0 / self.fps
        self._nlines = 0
        first = True
        if self.color:
            self.stream.write(HIDE_CURSOR)
        try:
            while not self._stop_event.is_set():
                for k in range(n):
                    if self._stop_event.is_set():
                        return
                    if self._fit():      # terminal resized mid-animation
                        first = True
                        self._resync_split()
                    self._draw(self._frame_at(k / (n - 1)), first)
                    first = False
                    self._stop_event.wait(dt)
                self._stop_event.wait(hold)   # let the ring glow before the next one
        except Exception:
            # A dead banner thread must never take the run down with it.
            pass

    def stop(self, freeze: bool = True) -> None:
        """Stop the animation, hand the terminal back, release any held logs."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None
            self._stop_event = None

            self._status = ""
            try:
                if self._split:
                    # Drop the region and wipe the pinned area, then let the frozen
                    # banner print as ordinary output so it stays in the scrollback,
                    # above whatever the run logs next.
                    self._exit_split()
                    self._unwrap_log_handlers()
                    if freeze:
                        with self._io_lock:
                            self.stream.write("\n".join(self.static_frame()) + "\n")
                elif freeze and self.animate:
                    self._draw(self.static_frame(), first=(self._nlines == 0))
            finally:
                with self._io_lock:
                    if self.color:
                        self.stream.write(SHOW_CURSOR)
                    self.stream.write("\n")
                    self.stream.flush()
        if self._log_hold is not None:
            self._log_hold.release()
            self._log_hold = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        # On an exception, drop the frozen frame: the traceback is what matters.
        self.stop(freeze=exc_type is None)
        return False


# --------------------------------------------------------------------------- #
#  Convenience: play the start-of-training banner
# --------------------------------------------------------------------------- #
def show_training_banner(
    engine: str | None = None,
    device=None,
    params: int | None = None,
    epochs=None,
    subtitle: str = "Hyper-Kamiokande · deep learning framework",
    stream=None,
    force_color: bool | None = None,
) -> None:
    """Play the start-of-training banner (drop-in replacement for a plain
    "starting training" log line).

    Honours the same degradations as ``HyperKBanner`` (HK_BANNER=0 disables it,
    non-TTY stdout falls back to a static frame, NO_COLOR/TERM=dumb drop color),
    so it is safe to call unconditionally on a cluster. Call it on rank 0 only.

    Args:
        engine: engine name shown in the info line (e.g. "multiring/segmentation").
        device: device shown in the info line (any object; str() is applied).
        params: number of trainable parameters (formatted with thin spaces).
        epochs: number of epochs shown in the info line.
        subtitle: subtitle under the title.
        stream: output stream (defaults to sys.stdout).
        force_color: force ANSI color on/off (default: auto-detect).
    """
    info: dict = {}
    if engine is not None:
        info["engine"] = engine
    if device is not None:
        info["device"] = str(device)
    if params is not None:
        info["params"] = f"{params:,}".replace(",", " ")
    if epochs is not None:
        info["epochs"] = epochs

    _banner(info, subtitle, stream, force_color).play()


def _banner(info, subtitle, stream, force_color, enabled: bool = True) -> HyperKBanner:
    return HyperKBanner(
        name="WatChMaL",
        subtitle=subtitle,
        info=info,
        stream=stream,
        force_color=force_color,
        enabled=enabled,
    )


def loading_banner(
    engine: str | None = None,
    device=None,
    params: int | None = None,
    epochs=None,
    subtitle: str = "Hyper-Kamiokande · deep learning framework",
    stream=None,
    force_color: bool | None = None,
    enabled: bool = True,
) -> HyperKBanner:
    """Banner that animates *while* something slow happens, as a context manager.

        with loading_banner(engine="graph/reconstruction", device=device):
            dataset = build_the_expensive_thing()   # animation runs meanwhile

    That is the whole wiring: one `with` around the slow section. The animation loops
    until the block exits, so it covers a wait of unknown length instead of adding a
    fixed 3 s of its own - which is what calling .play() at the top of train() does,
    where everything heavy has already been loaded.

    While it runs, the terminal is split: the run's normal log output keeps scrolling
    in the top rows and the animation is pinned below, with `set_status(...)` saying
    what is currently being waited on. The status is drawn only - it is never logged,
    so it does not reach main.log or wandb.

    Same degradations as HyperKBanner: HK_BANNER=0 disables it, a non-TTY stdout prints
    the static frame once and starts no thread. `enabled=False` makes every method a
    no-op, which is how non-zero DDP ranks share the same call site.
    """
    info: dict = {}
    if engine is not None:
        info["engine"] = engine
    if device is not None:
        info["device"] = str(device)
    if params is not None:
        info["params"] = f"{params:,}".replace(",", " ")
    if epochs is not None:
        info["epochs"] = epochs
    return _banner(info, subtitle, stream, force_color, enabled=enabled)


# --------------------------------------------------------------------------- #
#  Bonus: matching tqdm bar (same glyph ramp as the PMTs)
# --------------------------------------------------------------------------- #
def tqdm_kwargs(desc: str = "training") -> dict:
    """
    Pass as-is to tqdm:
        for batch in tqdm(loader, **tqdm_kwargs("epoch 1/5")):
    """
    return dict(
        desc=desc,
        ascii=" ·∙○●◉",
        bar_format="{desc} ⟨{bar}⟩ {percentage:3.0f}%  {n_fmt}/{total_fmt}  "
                   "· {rate_fmt} · ETA {remaining}",
        ncols=88,
        colour="cyan",
    )


if __name__ == "__main__":
    # Demos:
    #   python -m watchmal.utils.banner            one cycle, blocking (as before)
    #   python -m watchmal.utils.banner --async    loops while fake work + logs run
    #   python -m watchmal.utils.banner --static   static frame only
    demo = HyperKBanner(
        name="WatChMaL",
        subtitle="Hyper-Kamiokande · multi-ring segmentation",
        info={
            "engine": "multiring/segmentation",
            "device": "cuda:0",
            "params": "2 285 344",
            "epochs": 5,
        },
        force_animation="--static" not in sys.argv,
        force_color=True,
    )

    if "--async" in sys.argv:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        demo_log = logging.getLogger("demo")
        with demo:                       # animation runs in the background
            for step in range(8):        # pretend to load something heavy
                demo_log.info("loading shard %d/8 ...", step + 1)
                time.sleep(1.0)
        demo_log.info("held log lines are replayed above, in order")
    else:
        demo.play()
