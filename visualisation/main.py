"""MLP *training* visualizer, pygame edition (rendering only).

All the neural-network / autograd / training logic lives in `viz_core.py`; this file
is purely the visualisation + interaction layer. `viz_core.Trainer` runs the real
`fit` loop (Adam, BCE, shuffled mini-batches) one batch at a time and hands us, per
batch, the computation graph as `ViewGraph`s (nodes, edges, layer groups, per-op
forward/backward steps + equations). We:
  * lay the graph out in layers and draw every node as its full matrix of numbers,
  * animate the forward->backward wave op-by-op, driven by real batches,
  * show the 2D dataset in a side panel with a live decision boundary,
  * run a 60 FPS pygame loop with a zoom/pan camera and a mathtext equation card.

Run:  python viz_pygame.py [--dataset moons|circles]
      python viz_pygame.py --selftest        (headless checks, no window)

Controls:
  SPACE pause/resume        N / →  one op forward     ← one op back
  ↑ / ↓ playback speed      A auto-play (op-by-op)     F2 fast-forward (turbo)
  V values/grads/auto       C collapse Linear ops      F fit view
  R restart training        ESC / Q quit
  mouse wheel: zoom the graph (toward cursor), or resize the dataset square when
  the cursor is over it        left-drag pan
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from collections import defaultdict

import numpy as np

import visualisation.core as core

# ======================================================================
#  CONFIG  (what to visualise — the model / training choices live here)
# ======================================================================
ARCH = (2, [8, 6], 1)
BATCH = 8
SEED = 42
DROPOUT = 0.0  # 0.0 -> clean matmul/add/relu graph (bump to add Dropout nodes)
LR = 0.02  # Adam learning rate (train.fit uses Adam; a touch higher so it visibly learns)
N_SAMPLES = 500  # dataset size (like toy_classification); train set trimmed to a whole # of batches
DATASET = "moons"  # default; overridden by --dataset

FPS = 60
WIN_W, WIN_H = 1760, 1040  # desired window size (capped to the display at launch)
TOP_H = 96  # fixed caption/legend bar at the top (screen space, not zoomed)

# base box geometry, in *world* units at zoom = 1 (a cell holds one number)
CELL_W, CELL_H = 52, 32
TITLE_H = 26
COL_GAP = 100
ROW_GAP = 36

MIN_SCALE, MAX_SCALE = 0.12, 6.0

# playback: auto-play advances ops at SPEED ops/second; turbo skips animation
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 1.0, 240.0, 2.0
TURBO_BATCHES_PER_FRAME = 6  # whole batches trained per frame in fast-forward
BOUNDARY_RES = 72  # decision-boundary grid resolution

# dataset panel: its base size, and how far the wheel can grow/shrink it
PANEL_BASE_FRAC = 0.22  # base side as a fraction of the window width
PANEL_SCALE_MIN, PANEL_SCALE_MAX = 0.5, 3.5

# VJP FOCUS panel size knobs
VJP_PANEL_MAX_FRAC = 0.60  # how far down the window the panel may extend (bigger -> bigger)
VJP_CELL_MAX = 64  # max px per matrix cell in the panel (bigger -> bigger numbers)

# ---- palette (RGB) ----
BG = (238, 241, 244)
HUD_BG = (250, 251, 252)
CARD_BG = (255, 255, 255)
WHITE = (255, 255, 255)
HIDDEN_FILL = (232, 236, 239)
BORDER_HIDDEN = (183, 189, 195)
BORDER_DONE = (90, 134, 173)
BORDER_ACTIVE = (243, 156, 18)
BORDER_GRAD = (39, 174, 96)
EDGE = (150, 158, 164)
INK = (26, 32, 40)
MUTED = (120, 130, 140)
LAYER_TINTS = [(41, 128, 185), (39, 174, 96), (155, 89, 182), (211, 84, 0)]

# diverging colour ramps: (negative end, white middle, positive end)
DATA_RAMP = ((59, 111, 181), (247, 247, 247), (192, 57, 43))  # blue - white - red
GRAD_RAMP = ((125, 60, 152), (247, 247, 247), (230, 126, 34))  # purple - white - orange

MODES = ("auto", "values", "grads")  # matrix content switch (key V)


# ======================================================================
#  SMALL VISUAL HELPERS
# ======================================================================
def as2d(a) -> np.ndarray:
    """How a tensor is *shown*: scalar -> 1x1, vector -> a single row."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a


def fmt(v: float) -> str:
    return "0" if v == 0 else f"{v:.2f}"


def pfmt(v: float) -> str:
    return "0" if v == 0 else f"{v:.2f}"


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(round(lerp(c1[i], c2[i], t)) for i in range(3))


def diverging(value: float, vmax: float, ramp):
    if vmax <= 0:
        return ramp[1]
    t = max(-1.0, min(1.0, value / vmax))
    return mix(ramp[1], ramp[0], -t) if t < 0 else mix(ramp[1], ramp[2], t)


def luminance(c) -> float:
    return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0


def box_size(mat2d: np.ndarray):
    """World (width, height) of a node box holding this 2D matrix."""
    r, c = mat2d.shape
    return c * CELL_W, r * CELL_H + TITLE_H


# ======================================================================
#  MATH RENDERING  — matplotlib mathtext -> pygame Surface (cached)
# ======================================================================
import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "cm"  # Computer Modern
import pygame  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

_eq_cache: dict = {}
_eq_scaled: dict = {}


def render_equation(latex: str, fontsize: int = 30, color: str = "#12222f") -> pygame.Surface:
    """Rasterise a LaTeX string to a transparent pygame Surface (cached per string)."""
    key = (latex, fontsize, color)
    if key in _eq_cache:
        return _eq_cache[key]
    fig = Figure()
    fig.patch.set_alpha(0.0)
    FigureCanvasAgg(fig)
    fig.text(0.01, 0.5, latex, fontsize=fontsize, color=color, ha="left", va="center")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.06)
    buf.seek(0)
    surf = pygame.image.load(buf, "equation.png").convert_alpha()
    _eq_cache[key] = surf
    return surf


def fit_equation(latex: str, max_w: int, max_h: int) -> pygame.Surface:
    """Equation surface scaled down once (cached) to fit a box."""
    base = render_equation(latex)
    key = (latex, max_w, max_h)
    if key in _eq_scaled:
        return _eq_scaled[key]
    w, h = base.get_size()
    s = min(1.0, max_w / w, max_h / h)
    surf = base if s >= 0.999 else pygame.transform.smoothscale(base, (max(1, int(w * s)), max(1, int(h * s))))
    _eq_scaled[key] = surf
    return surf


# ======================================================================
#  CAMERA  — single (scale, offset) transform for all world coordinates
# ======================================================================
class Camera:
    def __init__(self):
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0

    def w2s(self, x, y):
        return (x * self.scale + self.ox, y * self.scale + self.oy)

    def s2w(self, sx, sy):
        return ((sx - self.ox) / self.scale, (sy - self.oy) / self.scale)

    def zoom_at(self, factor, px, py):
        """Zoom by `factor`, keeping the world point under (px, py) fixed on screen."""
        ns = max(MIN_SCALE, min(MAX_SCALE, self.scale * factor))
        self.ox = px - (px - self.ox) * (ns / self.scale)
        self.oy = py - (py - self.oy) * (ns / self.scale)
        self.scale = ns

    def pan(self, dx, dy):
        self.ox += dx
        self.oy += dy

    def fit(self, bounds, view):
        x0, y0, x1, y1 = bounds
        bw, bh = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
        vx, vy, vw, vh = view
        self.scale = max(MIN_SCALE, min(MAX_SCALE, min(vw / bw, vh / bh) * 0.9))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.ox = vx + vw / 2 - cx * self.scale
        self.oy = vy + vh / 2 - cy * self.scale


# ======================================================================
#  SCENE  — a ViewGraph plus the pixel layout the renderer needs
# ======================================================================
class Scene:
    """Wraps a core.ViewGraph with world-space positions/sizes for its nodes."""

    def __init__(self, view: core.ViewGraph):
        self.view = view
        self.sizes = {nid: box_size(as2d(view.nodes[nid].mat)) for nid in view.order}
        self.pos = self._layout()
        self.boxes = self._layer_boxes()
        self.bounds = self._bounds()

    # convenience passthroughs
    @property
    def steps(self):
        return self.view.steps

    @property
    def nodes(self):
        return self.view.nodes

    @property
    def edges(self):
        return self.view.edges

    @property
    def order(self):
        return self.view.order

    def _layout(self):
        """Layered layout in world pixels.

        x-column: an op keeps its longest-path depth; a weight/bias leaf sits in the
        SAME column as the op it feeds (so a Linear's ops + params can be boxed); the
        input x sits just before its consumer. Column x-positions use the widest box
        in each column; within a column the chain sits on y=0 and params fan off-axis.
        """
        view = self.view
        idset = set(view.order)
        parents = defaultdict(list)
        for c, p in view.edges:
            parents[c].append(p)

        xcol = {}
        for nid in view.order:
            n = view.nodes[nid]
            if n.is_op:
                xcol[nid] = n.depth
            else:
                ps = [view.nodes[p].depth for p in parents.get(nid, []) if p in idset]
                if not ps:
                    xcol[nid] = n.depth
                elif n.is_param:  # weight/bias -> its consumer's column
                    xcol[nid] = min(ps)
                else:  # input x -> just before its consumer
                    xcol[nid] = min(ps) - 1

        cols = defaultdict(list)
        for nid in view.order:
            cols[xcol[nid]].append(nid)
        colw = {d: max(self.sizes[n][0] for n in ns) for d, ns in cols.items()}
        xcenter, cursor = {}, 0.0
        for d in sorted(cols):
            xcenter[d] = cursor + colw[d] / 2
            cursor += colw[d] + COL_GAP

        params_above = view.title == "detailed"
        pos = {}
        for d, ns in cols.items():
            cx = xcenter[d]
            chain = sorted((n for n in ns if not view.nodes[n].is_param), key=lambda n: view.nodes[n].order)
            params = sorted((n for n in ns if view.nodes[n].is_param), key=lambda n: view.nodes[n].order)
            yc = -sum(self.sizes[n][1] for n in chain) / 2
            for n in chain:
                hh = self.sizes[n][1]
                pos[n] = (cx, yc + hh / 2)
                yc += hh + ROW_GAP
            top = min((pos[n][1] - self.sizes[n][1] / 2 for n in chain), default=0.0)
            bot = max((pos[n][1] + self.sizes[n][1] / 2 for n in chain), default=0.0)
            up, down = top, bot
            for i, n in enumerate(params):
                hh = self.sizes[n][1]
                if params_above or i % 2 == 1:  # above the chain
                    up -= ROW_GAP + hh / 2
                    pos[n] = (cx, up)
                    up -= hh / 2
                else:  # below the chain
                    down += ROW_GAP + hh / 2
                    pos[n] = (cx, down)
                    down += hh / 2
        return pos

    def _layer_boxes(self):
        boxes = []
        for g in self.view.layer_groups:
            mem = [m for m in g["members"] if m in self.pos]
            if not mem:
                continue
            x0 = min(self.pos[m][0] - self.sizes[m][0] / 2 for m in mem) - 16
            x1 = max(self.pos[m][0] + self.sizes[m][0] / 2 for m in mem) + 16
            y0 = min(self.pos[m][1] - self.sizes[m][1] / 2 for m in mem) - 36
            y1 = max(self.pos[m][1] + self.sizes[m][1] / 2 for m in mem) + 16
            boxes.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "label": g["label"],
                    "color": LAYER_TINTS[g["index"] % len(LAYER_TINTS)],
                    "members": set(g["members"]),
                }
            )
        return boxes

    def _bounds(self):
        xs0, ys0, xs1, ys1 = [], [], [], []
        for nid, (cx, cy) in self.pos.items():
            w, h = self.sizes[nid]
            xs0.append(cx - w / 2)
            ys0.append(cy - h / 2)
            xs1.append(cx + w / 2)
            ys1.append(cy + h / 2)
        for b in self.boxes:
            xs0.append(b["x0"])
            ys0.append(b["y0"])
            xs1.append(b["x1"])
            ys1.append(b["y1"])
        return (min(xs0), min(ys0), max(xs1), max(ys1))


# ======================================================================
#  APP
# ======================================================================
class App:
    def __init__(self, headless=False, dataset=DATASET):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.display.set_caption(f"micrograd — training ({dataset})")
        # open large: up to WIN_W x WIN_H, but capped to ~92% of the actual display
        win = (WIN_W, WIN_H)
        if not headless:
            try:
                dw, dh = pygame.display.get_desktop_sizes()[0]
                win = (min(WIN_W, int(dw * 0.92)), min(WIN_H, int(dh * 0.92)))
            except (pygame.error, IndexError, ValueError):
                pass
        self.screen = pygame.display.set_mode(win, pygame.RESIZABLE)
        self.w, self.h = self.screen.get_size()
        self.clock = pygame.time.Clock()

        # fonts — fixed screen-space sizes so numbers stay crisp under zoom
        self.f_num = pygame.font.SysFont("couriernew,menlo,consolas,monospace", 14)
        self.f_title = pygame.font.SysFont("arial,dejavusans,sans", 15, bold=True)
        self.f_cap = pygame.font.SysFont("arial,dejavusans,sans", 26, bold=True)
        self.f_info = pygame.font.SysFont("arial,dejavusans,sans", 18)
        self.f_hint = pygame.font.SysFont("arial,dejavusans,sans", 14)
        self.f_card = pygame.font.SysFont("arial,dejavusans,sans", 19, bold=True)
        self.f_plabel = pygame.font.SysFont("arial,dejavusans,sans", 15, bold=True)  # VJP term labels

        # the real training run lives in viz_core; the app only visualises it
        self.dataset = dataset
        self.trainer = core.Trainer(dataset, ARCH, BATCH, SEED, DROPOUT, LR, N_SAMPLES)
        self._rebuild_scenes()
        self.collapse = False

        self.cam = Camera()
        self.step_i = 0  # index of the current op within the current batch's steps
        self.mode = "auto"  # matrix content: auto | values | grads
        self.time = 0.0
        self.dragging = False
        self.grad_alpha: dict = {}
        self.spot = None
        self._num_cache: dict = {}
        self._pnum_cache: dict = {}  # VJP panel glyphs, keyed (text, colour, font size)
        self._font_by_size: dict = {}  # monospace fonts by pixel size (panel numbers)

        # playback state
        self.playing = False  # auto-advance ops
        self.turbo = False  # fast-forward whole batches
        self.speed = SPEED_DEFAULT  # ops / second when playing
        self.play_acc = 0.0

        self.boundary_surf = None  # small decision-boundary image (res x res)
        self.panel_scale = 1.0  # dataset panel zoom (wheel over the panel)
        self.recompute_boundary()
        self.fit_view()

    @property
    def scene(self) -> Scene:
        return self.scenes[self.collapse]

    def panel_geom(self):
        """Screen rect (px, top, size) of the dataset panel, incl. its own zoom."""
        base = min(PANEL_BASE_FRAC * self.w, self.h - TOP_H - 70)
        size = int(max(120, min(base * self.panel_scale, self.w - 40, self.h - TOP_H - 70)))
        return self.w - size - 20, TOP_H + 42, size

    @property
    def step(self) -> core.Step:
        return self.scene.steps[self.step_i]

    def world_rect(self):
        return pygame.Rect(0, TOP_H, self.w, self.h - TOP_H)

    def fit_view(self):
        """Default 'reading' view: scale the graph to (mostly) fill the width so the
        matrices are legible, and sit it low in the window — leaving the top clear
        for the equation card and the dataset panel."""
        wr = self.world_rect()
        x0, y0, x1, y1 = self.scene.bounds
        bw, bh = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
        s = wr.w * 0.97 / bw  # fill the width (small side margins)
        if bh * s > wr.h * 0.95:  # ...unless that would overflow the height
            s = wr.h * 0.95 / bh
        s = max(MIN_SCALE, min(MAX_SCALE, s))
        self.cam.scale = s
        free = max(0.0, wr.h - bh * s)  # vertical slack: put ~72% of it above the graph
        self.cam.ox = wr.x + wr.w / 2 - (x0 + x1) / 2 * s  # centred horizontally
        self.cam.oy = wr.y + free * 0.72 - y0 * s

    def num_surf(self, text, color):
        s = self._num_cache.get((text, color))
        if s is None:
            s = self.f_num.render(text, True, color)
            self._num_cache[(text, color)] = s
        return s

    def _panel_font(self, size):  # monospace font of a given pixel size, cached
        f = self._font_by_size.get(size)
        if f is None:
            f = pygame.font.SysFont("couriernew,menlo,consolas,monospace", size)
            self._font_by_size[size] = f
        return f

    def pnum_surf(self, text, color, size):  # cached VJP-panel glyphs (size-aware)
        s = self._pnum_cache.get((text, color, size))
        if s is None:
            s = self._panel_font(size).render(text, True, color)
            self._pnum_cache[(text, color, size)] = s
        return s

    # ---------- training <-> graph plumbing ----------
    def _rebuild_scenes(self):
        """Lay out the current batch's ViewGraphs (structure identical each batch,
        only the numbers change, so positions stay put)."""
        self.scenes = {k: Scene(v) for k, v in self.trainer.views.items()}

    def _on_batch_change(self):
        self._rebuild_scenes()
        self.step_i = 0
        self.grad_alpha.clear()
        self.spot = None
        self.recompute_boundary()  # boundary refreshes after every batch's update

    def advance_op(self):
        """One op forward. Past the end of a batch -> apply the update, redraw the
        decision boundary, and move on to the next batch's forward."""
        self.step_i += 1
        if self.step_i >= len(self.scene.steps):
            self.trainer.advance_batch()  # applies optimizer.step, preps next batch
            self._on_batch_change()

    def op_back(self):
        self.step_i = max(0, self.step_i - 1)  # within the current batch only

    def recompute_boundary(self):
        """Re-evaluate the model on a grid (eval mode) into a small RdBu image."""
        probs = self.trainer.decision_grid(BOUNDARY_RES)  # [iy, ix] in (0, 1)
        t = (probs * 2.0 - 1.0)[..., None]  # -> [-1, 1]
        neg, mid, pos = (np.array(c, dtype=float) for c in DATA_RAMP)
        col = mid + (neg - mid) * (-np.clip(t, -1, 0)) + (pos - mid) * np.clip(t, 0, 1)
        # rows go top(+y) -> bottom, and surfarray wants [x, y]
        img = np.transpose(col[::-1].astype(np.uint8), (1, 0, 2))
        self.boundary_surf = pygame.surfarray.make_surface(img)

    def set_collapse(self, value):
        if value == self.collapse:
            return
        self.collapse = value
        self.step_i = min(self.step_i, len(self.scene.steps) - 1)
        self.grad_alpha.clear()
        self.spot = None
        self.fit_view()

    def restart(self):
        self.trainer.restart()
        self._rebuild_scenes()
        self.step_i = 0
        self.playing = self.turbo = False
        self.grad_alpha.clear()
        self.spot = None
        self.recompute_boundary()
        self.fit_view()

    # ---------- events ----------
    def handle_event(self, e):
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.VIDEORESIZE:
            self.w, self.h = e.w, e.h
            self.screen = pygame.display.set_mode((self.w, self.h), pygame.RESIZABLE)
            self.fit_view()  # re-fit the reading view when the window (e.g.) maximises
        elif e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if e.key == pygame.K_SPACE:
                self.playing = not self.playing
            elif e.key in (pygame.K_RIGHT, pygame.K_n):
                self.playing = False
                self.advance_op()
            elif e.key == pygame.K_LEFT:
                self.playing = False
                self.op_back()
            elif e.key == pygame.K_UP:
                self.speed = min(SPEED_MAX, self.speed * 1.3)
            elif e.key == pygame.K_DOWN:
                self.speed = max(SPEED_MIN, self.speed / 1.3)
            elif e.key == pygame.K_a:
                self.playing = not self.playing
            elif e.key == pygame.K_F2:  # fast-forward whole batches (turbo)
                self.turbo = not self.turbo
                self.playing = False
                if not self.turbo:  # leaving turbo: resync the shown graph
                    self.trainer.prepare_current_batch()
                    self._on_batch_change()
            elif e.key == pygame.K_v:
                self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
            elif e.key == pygame.K_r:
                self.restart()
            elif e.key == pygame.K_f:
                self.fit_view()
            elif e.key == pygame.K_c:
                self.set_collapse(not self.collapse)
        elif e.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            px, top, size = self.panel_geom()
            if pygame.Rect(px, top, size, size).collidepoint(mx, my):
                # wheel over the dataset panel RESIZES the whole square (same view,
                # just bigger/smaller) — it does not zoom into the points
                self.panel_scale = max(PANEL_SCALE_MIN, min(PANEL_SCALE_MAX, self.panel_scale * 1.12**e.y))
            else:
                self.cam.zoom_at(1.12**e.y, mx, my)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            self.dragging = True
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.dragging = False
        elif e.type == pygame.MOUSEMOTION and self.dragging:
            self.cam.pan(*e.rel)
        return True

    # ---------- update ----------
    def update(self, dt):
        self.time += dt
        if self.turbo:
            # skip the op animation; train whole batches fast, then redraw the
            # boundary once for the frame (it still visibly evolves per batch)
            for _ in range(TURBO_BATCHES_PER_FRAME):
                self.trainer.fast_train_batch()
            self.recompute_boundary()
        elif self.playing:
            self.play_acc += dt * self.speed
            while self.play_acc >= 1.0:
                self.play_acc -= 1.0
                self.advance_op()

        st = self.step
        rate = 1.0 - pow(0.001, dt)  # frame-rate independent easing
        for nid in self.scene.nodes:  # green gradient-known outline fades in per node
            cur = self.grad_alpha.get(nid, 0.0)
            self.grad_alpha[nid] = cur + ((1.0 if nid in st.known else 0.0) - cur) * rate

        if st.active:  # spotlight eases toward the active node(s) — starts on x
            ax = sum(self.scene.pos[i][0] for i in st.active) / len(st.active)
            ay = sum(self.scene.pos[i][1] for i in st.active) / len(st.active)
            if self.spot is None:
                self.spot = [ax, ay]
            else:
                self.spot[0] += (ax - self.spot[0]) * rate
                self.spot[1] += (ay - self.spot[1]) * rate

    # ---------- matrix chosen by the values/grads switch ----------
    def matrix_for(self, nid, st):
        revealed, known = nid in st.revealed, nid in st.known
        if self.mode == "values":
            return (as2d(self.scene.nodes[nid].mat), DATA_RAMP) if revealed else None
        if self.mode == "grads":
            return (as2d(st.grads[nid]), GRAD_RAMP) if known else None
        # auto: data during forward, gradient during backward
        if st.phase == "backward" and known:
            return as2d(st.grads[nid]), GRAD_RAMP
        if revealed:
            return as2d(self.scene.nodes[nid].mat), DATA_RAMP
        return None

    # ---------- drawing ----------
    def draw(self):
        self.screen.fill(BG)
        wr = self.world_rect()
        self.screen.set_clip(wr)
        self._draw_boxes()
        self._draw_edges()
        self._draw_spotlight()
        self._draw_nodes()
        self.screen.set_clip(None)
        self._draw_hud()
        self._draw_eq_card()  # anchored to the graph's upper-left
        self._draw_dataset_panel()  # anchored to the graph's upper-right

    def _draw_boxes(self):
        st = self.step
        for b in self.scene.boxes:
            x0, y0 = self.cam.w2s(b["x0"], b["y0"])
            x1, y1 = self.cam.w2s(b["x1"], b["y1"])
            rect = pygame.Rect(x0, y0, x1 - x0, y1 - y0)
            active = bool(st.active & b["members"])
            pygame.draw.rect(self.screen, mix(b["color"], WHITE, 0.9), rect, border_radius=10)
            pygame.draw.rect(self.screen, b["color"], rect, width=3 if active else 1, border_radius=10)
            if rect.w > 90:
                self.screen.blit(self.f_title.render(b["label"], True, b["color"]), (x0 + 8, y0 + 6))

    def _draw_edges(self):
        st = self.step
        backward = st.phase == "backward"
        for c, p in self.scene.edges:
            nc, npar = self.scene.pos[c], self.scene.pos[p]
            a = self._edge_point(c, npar)  # child side
            b = self._edge_point(p, nc)  # parent side
            sa, sb = self.cam.w2s(*a), self.cam.w2s(*b)
            touch = bool(st.active & {c, p})
            # forward arrow child -> parent (highlighted orange only during forward)
            fc = BORDER_ACTIVE if (touch and not backward) else EDGE
            self._arrow(sa, sb, fc, 3 if (touch and not backward) else 1)
            # backward: a separate green arrow facing back (parent -> child) on top
            if backward and p in st.active:
                self._reverse_arrow(sa, sb)

    def _edge_point(self, nid, toward):
        """Where a line from node `nid`'s centre toward `toward` exits its box."""
        cx, cy = self.scene.pos[nid]
        w, h = self.scene.sizes[nid]
        dx, dy = toward[0] - cx, toward[1] - cy
        if dx == 0 and dy == 0:
            return cx, cy
        t = min((w / 2) / abs(dx) if dx else 1e9, (h / 2) / abs(dy) if dy else 1e9)
        return cx + dx * t, cy + dy * t

    def _arrow(self, s, e, color, width):
        pygame.draw.line(self.screen, color, s, e, width)
        self._arrow_head(s, e, color)

    def _reverse_arrow(self, sa, sb):
        """Green gradient arrow pointing parent->child, offset above the forward one."""
        dx, dy = sb[0] - sa[0], sb[1] - sa[1]
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        # perpendicular offset so it sits on top of, not hidden by, the forward arrow
        ox, oy = -dy / d * 7.0, dx / d * 7.0
        # pulse the width a little so the flowing gradient reads as "live"
        wdt = 3 + int(1.5 + 1.5 * np.sin(self.time * 6.0))
        p_start = (sb[0] + ox, sb[1] + oy)  # from parent
        p_end = (sa[0] + ox, sa[1] + oy)  # to child (arrowhead here)
        self._arrow(p_start, p_end, BORDER_GRAD, wdt)

    def _arrow_head(self, s, e, color):
        dx, dy = e[0] - s[0], e[1] - s[1]
        d = (dx * dx + dy * dy) ** 0.5
        if d < 1e-3:
            return
        ux, uy = dx / d, dy / d
        size = 10
        base = (e[0] - ux * size, e[1] - uy * size)
        perp = (-uy, ux)
        p1 = (base[0] + perp[0] * size * 0.55, base[1] + perp[1] * size * 0.55)
        p2 = (base[0] - perp[0] * size * 0.55, base[1] - perp[1] * size * 0.55)
        pygame.draw.polygon(self.screen, color, [e, p1, p2])

    def _draw_spotlight(self):
        if self.spot is None:
            return
        sx, sy = self.cam.w2s(*self.spot)
        r = int(max(46, 78 * self.cam.scale))
        glow = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        for rr, alpha in ((r, 26), (int(r * 0.7), 30), (int(r * 0.4), 34)):
            pygame.draw.circle(glow, (*BORDER_ACTIVE, alpha), (r, r), rr)
        self.screen.blit(glow, (sx - r, sy - r))

    def _draw_nodes(self):
        st = self.step
        wr = self.world_rect()
        pulse = 0.5 + 0.5 * np.sin(self.time * 6.0)
        # in backward, the active op's _backward writes the grad of its CHILD(REN);
        # highlight those receivers so the "active op -> child gets grad" shift is clear
        recv = set(st.vjp["children"]) if (st.phase == "backward" and st.vjp) else set()
        for nid in self.scene.view.order:
            n = self.scene.nodes[nid]
            cx, cy = self.scene.pos[nid]
            w0, h0 = self.scene.sizes[nid]
            tlx, tly = self.cam.w2s(cx - w0 / 2, cy - h0 / 2)
            bw, bh = w0 * self.cam.scale, h0 * self.cam.scale
            rect = pygame.Rect(tlx, tly, bw, bh)
            if not rect.colliderect(wr):
                continue

            picked = self.matrix_for(nid, st)
            pygame.draw.rect(self.screen, WHITE if picked else HIDDEN_FILL, rect, border_radius=6)
            if picked is not None:
                self._draw_matrix(rect, picked[0], picked[1], wr)

            if bw >= 56:  # title only when the box is wide enough to read it
                is_grad = picked is not None and picked[1] is GRAD_RAMP
                tag = "  grad" if is_grad else ""
                col = INK if picked is not None else MUTED
                self.screen.blit(self.f_title.render(f"{n.name}  {n.shape_str}{tag}", True, col), (tlx + 6, tly + 5))

            base = BORDER_DONE if nid in st.revealed else BORDER_HIDDEN
            border = mix(base, BORDER_GRAD, self.grad_alpha.get(nid, 0.0))
            pygame.draw.rect(self.screen, border, rect, width=2, border_radius=6)
            if nid in st.active:  # source of the VJP (orange)
                pygame.draw.rect(
                    self.screen, BORDER_ACTIVE, rect.inflate(6, 6), width=int(2 + pulse * 3), border_radius=8
                )
            elif nid in recv:  # child that just received the gradient (green)
                pygame.draw.rect(
                    self.screen, BORDER_GRAD, rect.inflate(6, 6), width=int(2 + pulse * 2.5), border_radius=8
                )

    def _draw_matrix(self, rect, mat, ramp, wr):
        r, c = mat.shape
        cw = rect.w / c
        ch = (rect.h - TITLE_H * self.cam.scale) / r
        top = rect.y + TITLE_H * self.cam.scale
        vmax = float(np.abs(mat).max())
        # number font follows the (zoomed) cell size, sized to fit a ".2f" value
        fs = int(max(6, min(44, cw * 0.34, ch * 0.72)))
        show_numbers = cw >= 22 and ch >= 12
        for i in range(r):
            for j in range(c):
                cell = pygame.Rect(rect.x + j * cw, top + i * ch, cw + 1, ch + 1)
                if not cell.colliderect(wr):
                    continue
                col = diverging(float(mat[i, j]), vmax, ramp)
                pygame.draw.rect(self.screen, col, cell.inflate(-1, -1))
                if show_numbers:
                    tc = WHITE if luminance(col) < 0.5 else (30, 30, 30)
                    g = self.pnum_surf(fmt(float(mat[i, j])), tc, fs)
                    self.screen.blit(g, g.get_rect(center=cell.center))

    # ---- fixed UI overlays (never affected by the camera) ----
    def _draw_dataset_panel(self):
        """Square panel (upper-right): the 2D dataset over a live decision-boundary
        heatmap; the current batch's samples are ringed (the same ones flowing as x).
        The wheel resizes the whole square when the cursor is over it (panel_geom)."""
        tr = self.trainer
        px, top, size = self.panel_geom()
        if size < 60:
            return
        rect = pygame.Rect(px, top, size, size)
        if self.boundary_surf is not None:  # decision-boundary background (per epoch)
            self.screen.blit(pygame.transform.smoothscale(self.boundary_surf, (size, size)), (px, top))
        pygame.draw.rect(self.screen, MUTED, rect, width=2, border_radius=6)
        self.screen.blit(
            self.f_card.render(f"dataset: {self.dataset}   ·   live boundary (epoch {tr.epoch})", True, INK),
            (px + 2, TOP_H + 14),
        )

        x_min, x_max, y_min, y_max = tr.extent

        def to_px(pt):
            u = (pt[0] - x_min) / (x_max - x_min)
            v = (pt[1] - y_min) / (y_max - y_min)
            return int(px + u * size), int(top + (1.0 - v) * size)

        dot = max(2, size // 110)  # point / ring radii grow with the panel
        ring = max(5, size // 70)  # small tight ring around the batch samples
        ring_w = max(2, min(ring - 1, size // 150 + 3))  # thicker stroke (never fills)
        c0, c1 = DATA_RAMP[0], DATA_RAMP[2]  # class 0 = blue, class 1 = red (RdBu)
        for pt, lbl in zip(tr.X_train, tr.y_train.ravel(), strict=True):
            sx, sy = to_px(pt)
            pygame.draw.circle(self.screen, c1 if lbl > 0.5 else c0, (sx, sy), dot)
            pygame.draw.circle(self.screen, (40, 40, 40), (sx, sy), dot, 1)
        for i in tr.batch_indices:  # ring the current batch's samples
            sx, sy = to_px(tr.X_train[i])
            pygame.draw.circle(self.screen, BORDER_ACTIVE, (sx, sy), ring, ring_w)

    def _draw_hud(self):
        pygame.draw.rect(self.screen, HUD_BG, pygame.Rect(0, 0, self.w, TOP_H))
        pygame.draw.line(self.screen, (210, 216, 221), (0, TOP_H), (self.w, TOP_H), 1)
        self.screen.blit(self.f_cap.render(self.step.caption, True, INK), (18, 12))
        ts = self.trainer.state()
        mode = "TURBO" if self.turbo else ("PLAY" if self.playing else "paused")
        info = (
            f"epoch {ts.epoch}   ·   batch {ts.batch_in_epoch + 1}/{ts.nb_batches}"
            f"   ·   loss {ts.batch_loss:.4f}   ·   op {self.step_i + 1}/{len(self.scene.steps)}"
            f"   ·   {self.scene.view.title}   ·   matrices: {self.mode}"
            f"   ·   {mode} @ {self.speed:.0f} ops/s"
        )
        self.screen.blit(self.f_info.render(info, True, MUTED), (18, 50))
        hint = (
            "SPACE play  N/→ op  ← back  ↑↓ speed  A auto  F2 turbo  V vals/grads"
            "  C collapse  wheel: zoom graph / resize dataset  F fit  R restart  Q quit"
        )
        surf = self.f_hint.render(hint, True, MUTED)
        self.screen.blit(surf, (self.w - surf.get_width() - 16, 72))

    def _blit_matrix(self, x, y, mat, cell, ramp):
        """Draw a labelled matrix in screen space (VJP panel); the number font scales
        with the cell so it stays readable when there's room, and hides when tiny."""
        r, c = mat.shape
        vmax = float(np.abs(mat).max())
        fs = int(max(9, min(32, cell * 0.35)))  # number font size follows the cell
        show = cell >= 16
        for i in range(r):
            for j in range(c):
                v = float(mat[i, j])
                col = diverging(v, vmax, ramp)
                cr = pygame.Rect(int(x + j * cell), int(y + i * cell), int(cell) + 1, int(cell) + 1)
                pygame.draw.rect(self.screen, col, cr.inflate(-1, -1))
                if show:
                    tc = WHITE if luminance(col) < 0.5 else (30, 30, 30)
                    g = self.pnum_surf(pfmt(v), tc, fs)
                    self.screen.blit(g, g.get_rect(center=cr.center))

    def _draw_vjp_panel(self, st):
        """The priority backward view: the VJP formula on top, and below it EVERY
        matrix the step touches — incoming gradient, forward operand values, and the
        produced child gradients — each labelled, in equation order."""
        vjp = st.vjp
        terms = [(lbl, kind, as2d(m)) for lbl, kind, m in vjp["terms"]]
        x0, y0, pad, gap = 16, TOP_H + 14, 14, 24
        avail_w = max(360, self.panel_geom()[0] - x0 - 24)  # stop before the dataset panel
        # use the empty band under the panel (may overlap the low graph's top — fine,
        # the panel is the focus in backward). VJP_PANEL_MAX_FRAC controls how tall.
        avail_bottom = self.h * VJP_PANEL_MAX_FRAC

        header = self.f_card.render(f"VJP FOCUS · {vjp['target']}", True, BORDER_GRAD)
        eq = fit_equation(st.eq, int(avail_w - 2 * pad), 108) if st.eq else None
        eq_h = eq.get_height() if eq else 0
        label_h = self.f_plabel.get_height() + 4
        matrix_y = y0 + pad + header.get_height() + 8 + eq_h + 12 + label_h

        # one shared cell size that fits the tallest matrix (height) and the row (width)
        max_rows = max(m.shape[0] for _, _, m in terms)
        total_cols = sum(m.shape[1] for _, _, m in terms)
        h_fit = (avail_bottom - matrix_y - pad) / max_rows
        w_fit = (avail_w - (len(terms) + 1) * gap) / max(1, total_cols)
        cell = max(7.0, min(float(VJP_CELL_MAX), h_fit, w_fit))  # big cells when there's room

        slots = [(max(m.shape[1] * cell, self.f_plabel.size(lbl)[0]), m.shape[0] * cell) for lbl, _, m in terms]
        content_w = sum(w for w, _ in slots) + (len(slots) + 1) * gap
        min_w = max(content_w, header.get_width() + 2 * pad, (eq.get_width() + 2 * pad) if eq else 0)
        panel_w = int(min(avail_w, min_w))
        panel_h = int((matrix_y - y0) + max(h for _, h in slots) + pad)

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(bg, (*CARD_BG, 240), bg.get_rect(), border_radius=12)
        pygame.draw.rect(bg, (*BORDER_GRAD, 255), bg.get_rect(), width=2, border_radius=12)
        self.screen.blit(bg, (x0, y0))
        self.screen.blit(header, (x0 + pad, y0 + pad))
        if eq:
            self.screen.blit(eq, (x0 + pad, y0 + pad + header.get_height() + 8))

        kind_col = {"incoming": GRAD_RAMP[0], "forward": BORDER_DONE, "produced": BORDER_GRAD}
        cx = x0 + gap
        for (lbl, kind, m), (sw, _sh) in zip(terms, slots, strict=True):
            self.screen.blit(self.f_plabel.render(lbl, True, kind_col.get(kind, INK)), (cx, matrix_y - label_h))
            ramp = DATA_RAMP if kind == "forward" else GRAD_RAMP
            self._blit_matrix(cx + (sw - m.shape[1] * cell) / 2, matrix_y, m, cell, ramp)
            cx += sw + gap

    def _draw_eq_card(self):
        """The op / gradient formula, in a card anchored to the graph's upper-left.
        During backward this expands into the full VJP FOCUS panel."""
        st = self.step
        if st.phase == "backward" and st.vjp:
            self._draw_vjp_panel(st)
            return
        backward = st.phase == "backward"
        header = "gradient rule (VJP)" if backward else "forward op"
        hcol = BORDER_GRAD if backward else BORDER_DONE
        hsurf = self.f_card.render(header, True, hcol)
        eq_surf = fit_equation(st.eq, 620, 84) if st.eq else self.f_info.render("(inputs - no op)", True, MUTED)

        pad = 14
        cw = max(hsurf.get_width(), eq_surf.get_width()) + 2 * pad
        chh = pad + hsurf.get_height() + 10 + eq_surf.get_height() + pad
        wr = self.world_rect()
        x, y = wr.x + 18, wr.y + 18

        card = pygame.Surface((cw, chh), pygame.SRCALPHA)
        card.fill((0, 0, 0, 0))
        pygame.draw.rect(card, (*CARD_BG, 238), card.get_rect(), border_radius=12)
        pygame.draw.rect(card, (*hcol, 255), card.get_rect(), width=2, border_radius=12)
        card.blit(hsurf, (pad, pad))
        card.blit(eq_surf, ((cw - eq_surf.get_width()) // 2, pad + hsurf.get_height() + 10))
        self.screen.blit(card, (x, y))

    # ---------- main loop ----------
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            for e in pygame.event.get():
                running = self.handle_event(e) and running
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()


# ======================================================================
#  SELF-TEST  (headless: no window opens)
# ======================================================================
def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def selftest() -> int:
    print("[selftest] importing + init ...")
    app = App(headless=True, dataset="moons")
    tr = app.trainer
    print(f"  pygame {pygame.version.ver}, display {app.screen.get_size()}, dataset moons")
    print(f"  arch {ARCH}, batch {BATCH}, train samples {tr.nb_samples}, {tr.nb_batches} batches/epoch")

    for collapse in (False, True):
        sc = app.scenes[collapse]
        _check(sc.steps, f"{sc.view.title}: no steps")
        first = sc.steps[0]  # first focus must be the single input node x
        _check(first.active == {"x"}, f"{sc.view.title}: first focus is {first.active}, expected {{'x'}}")
        for nid, (w, h) in sc.sizes.items():
            _check(w > 0 and h > 0, f"{sc.nodes[nid].name} bad box")
        fwd = [s for s in sc.steps if s.phase == "forward"]
        bwd = [s for s in sc.steps if s.phase == "backward"]
        grows = all(len(fwd[i].revealed) <= len(fwd[i + 1].revealed) for i in range(len(fwd) - 1))
        _check(grows, f"{sc.view.title}: forward reveals not monotone")
        _check(len(bwd[-1].known) == len(sc.nodes), f"{sc.view.title}: grads did not reach all nodes")
        print(f"  scene '{sc.view.title}': {len(sc.nodes)} nodes, {len(sc.steps)} steps, first focus = x, ok")

    # the x node holds THIS batch's real data (same samples ringed in the panel)
    x0 = app.scenes[False].nodes["x"].mat.copy()
    _check(np.allclose(x0, tr.X_train[tr.batch_indices]), "x node != current batch data")
    _check(app.scenes[False].nodes["x"].mat.shape == (BATCH, 2), "x node wrong shape")

    # advance one whole batch op-by-op -> optimizer.step updates the weights, next batch loads
    w_before = tr.model.parameters[0].data.copy()
    for _ in range(len(app.scenes[False].steps)):
        app.advance_op()
    _check(not np.allclose(w_before, tr.model.parameters[0].data), "weights did not update after a batch")
    _check(not np.allclose(x0, app.scenes[False].nodes["x"].mat), "batch did not advance")
    print("  op-by-op advance: batch values propagate into `x`, optimizer.step updates weights")

    # decision-boundary grid eval returns a valid probability field
    probs = tr.decision_grid(16)
    _check(probs.shape == (16, 16), f"grid shape {probs.shape}")
    _check(float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0, "grid values not in [0,1]")
    print("  decision_grid -> valid (16,16) probabilities in [0,1]")

    # VJP focus data: every detailed backward op carries its full term set, and the
    # forward operand values are present even though we are in the backward phase
    det = app.scenes[False]
    bwd_ops = [s for s in det.steps if s.phase == "backward" and s.vjp]
    _check(bwd_ops, "no backward steps carry VJP terms")
    for s in bwd_ops:
        _check(s.vjp["children"], f"{s.vjp['target']}: no grad receivers")
        kinds = {k for _, k, _ in s.vjp["terms"]}
        _check("incoming" in kinds and "produced" in kinds, f"{s.vjp['target']}: missing incoming/produced")
        for _lbl, _k, m in s.vjp["terms"]:
            _check(np.asarray(m).size > 0, f"{s.vjp['target']}: empty VJP matrix")
    mm = next(s for s in bwd_ops if s.vjp["target"].startswith("matmul"))
    fwd_terms = [lbl for lbl, k, _ in mm.vjp["terms"] if k == "forward"]
    _check(len(fwd_terms) == 2, f"matmul VJP should expose X and W forward values, got {fwd_terms}")
    print(f"  VJP focus: {len(bwd_ops)} backward ops carry full labelled term sets (incl. forward X/W)")

    # turbo trains whole batches fast and advances epochs
    e0 = tr.epoch
    app.turbo = True
    for _ in range(30):  # 30 * TURBO_BATCHES_PER_FRAME batches
        app.update(0.016)
    app.turbo = False
    _check(tr.epoch > e0, f"turbo did not advance an epoch ({e0} -> {tr.epoch})")
    _check(app.boundary_surf is not None, "no boundary surface")
    print(f"  turbo fast-forward: epoch {e0} -> {tr.epoch}, boundary recomputed")

    # equations rasterise; camera math holds
    eqs = {s.eq for sc in app.scenes.values() for s in sc.steps if s.eq}
    for eq in eqs:
        surf = render_equation(eq)
        _check(isinstance(surf, pygame.Surface) and surf.get_width() > 0 and surf.get_height() > 0, f"bad eq {eq}")
    cam = Camera()
    cam.scale, cam.ox, cam.oy = 1.3, 40.0, -15.0
    before = cam.s2w(300.0, 220.0)
    cam.zoom_at(1.5, 300.0, 220.0)
    after = cam.s2w(300.0, 220.0)
    _check(abs(before[0] - after[0]) < 1e-6 and abs(before[1] - after[1]) < 1e-6, "zoom-to-cursor drift")
    print(f"  {len(eqs)} equations rasterised (mathtext), camera zoom-to-cursor OK")

    # render frames for both views x 3 matrix modes (incl. a backward step + panel)
    app.trainer.prepare_current_batch()
    app._rebuild_scenes()
    for collapse in (False, True):
        app.set_collapse(collapse)
        for app.mode in MODES:
            for idx in (0, len(app.scene.steps) // 2, len(app.scene.steps) - 1):
                app.step_i = idx
                app.update(0.016)
                app.draw()
    _check(isinstance(app.screen, pygame.Surface), "screen not a Surface")
    print("  rendered graph + dataset panel for both views x 3 modes (no errors)")

    pygame.quit()
    print("[selftest] PASS")
    return 0


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser(description="micrograd training visualiser")
    ap.add_argument("--dataset", choices=["moons", "circles"], default=DATASET)
    args = ap.parse_args()
    App(dataset=args.dataset).run()


if __name__ == "__main__":
    main()
