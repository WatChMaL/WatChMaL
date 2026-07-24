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

import math
import os
import shutil
import sys
import time

# --------------------------------------------------------------------------- #
#  ANSI sequences
# --------------------------------------------------------------------------- #
CSI = "\x1b["
RESET = CSI + "0m"
HIDE_CURSOR = CSI + "?25l"
SHOW_CURSOR = CSI + "?25h"
CLEAR_LINE = CSI + "2K"

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
        gap: int = 26,           # beam line length, on the left
        fps: int = 45,
        stream=None,
        force_animation: bool = False,
        force_color: bool | None = None,
    ):
        self.name = name
        self.subtitle = subtitle
        self.info = info or {}
        self.rows = rows
        self.fps = fps
        self.stream = stream or sys.stdout

        enabled = os.environ.get("HK_BANNER", "1") != "0"
        tty = force_animation or (hasattr(self.stream, "isatty") and self.stream.isatty())
        self.animate = enabled and tty
        self.enabled = enabled

        if force_color is None:
            self.color = tty and not os.environ.get("NO_COLOR") \
                and os.environ.get("TERM") != "dumb"
        else:
            self.color = force_color

        # --- adjust to the real terminal width ----------------------------- #
        term_w = shutil.get_terminal_size((100, 30)).columns
        self.cols = cols
        self.gap = gap
        while self.total_width > term_w - 2 and self.cols > 10:
            self.cols -= 1
            self.gap = max(10, self.gap - 1)
        while self.total_width > term_w - 2 and self.gap > 8:
            self.gap -= 1
        if self.total_width > term_w:            # really too narrow
            self.animate = False

        # interaction geometry: vertex at the tank center
        self.vx = (self.cols - 1) / 2.0
        self.vy = self.rows // 2
        self._vrow = self.rows // 2          # beam row (PMT index)
        half_h = (self.rows - 1) / 2.0
        half_w = (self.cols - 1) / 2.0
        self.r_max = min(half_w, half_h) * 1.02
        self.r_final = self.r_max * 0.85

        self._title = self._build_title()
        self._info_lines = self._build_info()

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
        buf = [] if first else [f"{CSI}{self._nlines}A"]
        for ln in lines:
            buf.append(CLEAR_LINE + ln + "\n")
        self._nlines = len(lines)
        self.stream.write("".join(buf))
        self.stream.flush()

    def play(self, duration: float = 3.2) -> None:
        """Play the animation then freeze the banner. No effect if non-TTY."""
        if not self.enabled:
            return
        if not self.animate:
            self.stream.write("\n".join(self.static_frame()) + "\n")
            self.stream.flush()
            return

        n = max(24, int(duration * self.fps))
        t_approach, t_burst = 0.52, 0.78      # bounds as a fraction of n
        dt = 1.0 / self.fps
        self._nlines = 0
        first = True

        if self.color:
            self.stream.write(HIDE_CURSOR)
        try:
            for k in range(n):
                u = k / (n - 1)

                if u < t_approach:                       # --- approach ---
                    p = u / t_approach
                    # the neutrino travels along the beam then continues,
                    # invisible, behind the wall to the central vertex: the
                    # interaction only starts when it reaches the center
                    total = self.gap + self.vx
                    pos = _ease_in(p) * total
                    nu_x = pos if pos < self.gap else None
                    nu_col = None
                    radius, sigma, amp, flash = 0.0, 1.0, 0.0, 0.0
                    status = ""

                elif u < t_burst:                        # --- interaction ---
                    p = (u - t_approach) / (t_burst - t_approach)
                    nu_x, nu_col = None, None
                    radius = self.r_max * (p ** 0.55)
                    sigma = 0.75 + 0.5 * p
                    amp = 1.15
                    flash = 0.0            # no global flash: the wave starts from the center
                    status = ""

                else:                                    # --- stabilization ---
                    p = (u - t_burst) / (1.0 - t_burst)
                    nu_x, nu_col = None, None
                    radius = self.r_max + (self.r_final - self.r_max) * p
                    sigma = 1.25 - 0.3 * p
                    amp = 1.15 + (0.72 - 1.15) * p
                    flash = 0.0
                    status = ""

                self._draw(self._frame(nu_x, radius, sigma, amp, flash, status, nu_col), first)
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

    HyperKBanner(
        name="WatChMaL",
        subtitle=subtitle,
        info=info,
        stream=stream,
        force_color=force_color,
    ).play()


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
    HyperKBanner(
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
    ).play()
