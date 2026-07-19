"""Training visualizer, pygame edition (rendering only).

All the neural-network / autograd / training logic lives in `visualisation.core`;
this file is purely the visualisation + interaction layer. `core.make_trainer(...)`
runs a real training loop one mini-batch at a time and hands us, per batch, the
computation graph as `ViewGraph`s (nodes, edges, layer groups, per-op forward/backward
steps + equations). We:
  * lay the graph out in layers and draw every node as its full matrix of numbers,
  * animate the forward->backward wave op-by-op, driven by real batches,
  * (mlp) show the 2D dataset in a side panel with a live decision boundary,
  * (attention) flag the softmax attention-weights node as the centerpiece,
  * run a 60 FPS pygame loop with a zoom/pan camera and a mathtext equation card.

Two models share this exact engine — only which graph gets built branches, via the
MODEL_KIND constant (top of file) or --model:
  mlp        -> MLP on moons/circles  (examples/toy_classification + examples/train.fit)
  attention  -> tiny MaxClassificationModel  (examples/maximum), attention scores shown

Run:  python -m visualisation.main [--model mlp|attention] [--dataset moons|circles]
      python -m visualisation.main --selftest        (headless checks, both models)

Controls:
  SPACE pause/resume        N / →  one op forward     ← one op back
  ↑ / ↓ playback speed      A auto-play (op-by-op)     F2 fast-forward (turbo)
  V values/grads/auto       C collapse layers          F fit view
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
MODEL_KIND = "mlp"  # "mlp" | "attention"  -> which model's graph to visualise (or --model)
TASK = "classification"  # attention only: "classification" (max position) | "regression" (max value); or --task

# --- MLP mode (toy_classification: moons / circles) ---
ARCH = (2, [8, 6], 1)
BATCH = 8
DROPOUT = 0.1  # 0.0 -> clean matmul/add/relu graph (bump to add Dropout nodes)
LR = 0.02  # Adam learning rate (train.fit uses Adam; a touch higher so it visibly learns)
N_SAMPLES = 500  # dataset size (like toy_classification); train set trimmed to a whole # of batches
DATASET = "moons"  # default; overridden by --dataset

# --- attention mode (examples/attention: MaxClassificationModel or MaxRegressionModel) ---
# tiny on purpose: d_k=d_v=8, hidden=[8], seq_len 4 (+1 padded key) so every matrix —
# especially the attention weights — stays readable. Classification uses n_in=3 (1 value
# + 2 positional dims); regression uses n_in=1 (raw value only).
ATTN_BATCH = 1  # one sequence per batch, so EVERY node (incl. the 2D logits) shows that
#                 single sample — no mixing "sample 0 only" (3D nodes) with "all rows" (2D)
ATTN_LR = 0.01
ATTN_N_SAMPLES = 64

SEED = 42

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
ATTN_BORDER = (142, 68, 173)  # vivid purple: the attention-weights centerpiece node
MAX_MARK = (214, 40, 120)  # magenta: marks the true-max key position (the thing to watch)

# diverging colour ramps: (negative end, white middle, positive end)
DATA_RAMP = ((59, 111, 181), (247, 247, 247), (192, 57, 43))  # blue - white - red
GRAD_RAMP = ((125, 60, 152), (247, 247, 247), (230, 126, 34))  # purple - white - orange

MODES = ("auto", "values", "grads")  # matrix content switch (key V)


# ======================================================================
#  SMALL VISUAL HELPERS
# ======================================================================
def as2d(a) -> np.ndarray:
    """How a tensor is *shown*: scalar -> 1x1, vector -> a single row, and a batched
    3D+ tensor (batch, seq, dim) -> sample 0, so attention's (seq, seq) weights and
    (seq, dim) activations render as a plain matrix."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    while a.ndim > 2:  # show sample 0 of a batched tensor
        a = a[0]
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


def sequential(value: float, vmax: float):
    """White -> deep blue, intensity proportional to the (>=0) attention weight."""
    t = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax))
    return mix((255, 255, 255), (33, 102, 172), t)


def luminance(c) -> float:
    return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0


def box_size(mat2d: np.ndarray, scale: float = 1.0):
    """World (width, height) of a node box holding this 2D matrix."""
    r, c = mat2d.shape
    return c * CELL_W * scale, r * CELL_H * scale + TITLE_H


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

    ATTN_SCALE = 1.6  # the attention-weights node is drawn bigger than the rest

    def __init__(self, view: core.ViewGraph):
        self.view = view
        self.sizes = {
            nid: box_size(as2d(view.nodes[nid].mat), self.ATTN_SCALE if view.nodes[nid].highlight else 1.0)
            for nid in view.order
        }
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
    def __init__(self, headless=False, dataset=DATASET, model_kind=MODEL_KIND, task=TASK):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        if model_kind == "mlp":
            title = dataset
        else:
            title = "attention (max-position)" if task == "classification" else "attention (max-value)"
        pygame.display.set_caption(f"micrograd — training ({title})")
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

        # the real training run lives in core; the app only visualises it. The ONLY
        # thing that branches on the model is which trainer we build here.
        self.dataset = dataset
        self.model_kind = model_kind
        self.task = task
        self.has_dataset_panel = model_kind == "mlp"  # 2D boundary only makes sense for moons/circles
        self.trainer = core.make_trainer(
            model_kind,
            seed=SEED,
            lr=LR if model_kind == "mlp" else ATTN_LR,
            n_samples=N_SAMPLES if model_kind == "mlp" else ATTN_N_SAMPLES,
            dataset=dataset,
            arch=ARCH,
            batch_size=BATCH if model_kind == "mlp" else ATTN_BATCH,
            dropout=DROPOUT,
            task=task,
        )
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
        """Screen rect (px, top, size) of the dataset panel, incl. its own zoom, or
        None in attention mode (there is no 2D decision boundary for a sequence task)."""
        if not self.has_dataset_panel:
            return None
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
        """Re-evaluate the model on a grid (eval mode) into a small RdBu image.
        No-op in attention mode (no 2D boundary)."""
        if not self.has_dataset_panel:
            self.boundary_surf = None
            return
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
            pg = self.panel_geom()
            if pg and pygame.Rect(pg[0], pg[1], pg[2], pg[2]).collidepoint(mx, my):
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
        if self.model_kind == "attention":
            self._draw_seq_panel()  # the sequence + where its max is
        else:
            self._draw_dataset_panel()  # 2D dataset + live decision boundary

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

    def _mark_max_key(self, rect, shape):
        """Frame the true-max key COLUMN of the attention-weights matrix + caret above,
        so you can see attention concentrate on the max position (attention mode)."""
        info = getattr(self.trainer, "attn_info", None)
        if info is None:
            return
        r, c = shape
        mp = info["max_pos"]
        if not (0 <= mp < c):
            return
        cw = rect.w / c
        ch = (rect.h - TITLE_H * self.cam.scale) / r
        top = rect.y + TITLE_H * self.cam.scale
        col = pygame.Rect(int(rect.x + mp * cw), int(top), int(cw) + 1, int(r * ch) + 1)
        pygame.draw.rect(self.screen, MAX_MARK, col, width=4)
        # caret BELOW the node (the top is taken by the "ATTENTION WEIGHTS" label)
        noun = "position" if info.get("task", "classification") == "classification" else "value"
        caret = self.f_title.render(f"^ true max {noun} (key {mp})", True, MAX_MARK)
        self.screen.blit(caret, (col.centerx - caret.get_width() // 2, rect.bottom + 6))

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
            # the attention-weights node shows a sequential heatmap for its forward values
            hl_data = n.highlight and picked is not None and picked[1] is DATA_RAMP
            if picked is not None:
                self._draw_matrix(rect, picked[0], picked[1], wr, attn=hl_data)

            if bw >= 56:  # title only when the box is wide enough to read it
                is_grad = picked is not None and picked[1] is GRAD_RAMP
                tag = "  grad" if is_grad else ""
                col = INK if picked is not None else MUTED
                self.screen.blit(self.f_title.render(f"{n.name}  {n.shape_str}{tag}", True, col), (tlx + 6, tly + 5))

            base = BORDER_DONE if nid in st.revealed else BORDER_HIDDEN
            border = mix(base, BORDER_GRAD, self.grad_alpha.get(nid, 0.0))
            pygame.draw.rect(self.screen, border, rect, width=2, border_radius=6)
            if n.highlight:  # the centerpiece: thick purple ring + floating label, always
                pygame.draw.rect(self.screen, ATTN_BORDER, rect.inflate(10, 10), width=4, border_radius=10)
                lbl = self.f_card.render("ATTENTION WEIGHTS", True, ATTN_BORDER)
                self.screen.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.top - lbl.get_height() - 8))
                if hl_data:  # mark the true-max key column: attention should pile onto it
                    self._mark_max_key(rect, picked[0].shape)
            if nid in st.active:  # source of the VJP (orange)
                pygame.draw.rect(
                    self.screen, BORDER_ACTIVE, rect.inflate(6, 6), width=int(2 + pulse * 3), border_radius=8
                )
            elif nid in recv:  # child that just received the gradient (green)
                pygame.draw.rect(
                    self.screen, BORDER_GRAD, rect.inflate(6, 6), width=int(2 + pulse * 2.5), border_radius=8
                )

    def _draw_matrix(self, rect, mat, ramp, wr, attn=False):
        """Draw the matrix of numbers. `attn` uses a sequential heatmap (intensity ∝
        weight) and always shows the value, so masked ~0 keys stay visible, not blank."""
        r, c = mat.shape
        cw = rect.w / c
        ch = (rect.h - TITLE_H * self.cam.scale) / r
        top = rect.y + TITLE_H * self.cam.scale
        vmax = float(np.abs(mat).max())
        # number font follows the (zoomed) cell size, sized to fit a ".2f" value
        fs = int(max(6, min(44, cw * 0.34, ch * 0.72)))
        show_numbers = attn or (cw >= 22 and ch >= 12)
        for i in range(r):
            for j in range(c):
                cell = pygame.Rect(rect.x + j * cw, top + i * ch, cw + 1, ch + 1)
                if not cell.colliderect(wr):
                    continue
                v = float(mat[i, j])
                col = sequential(v, vmax) if attn else diverging(v, vmax, ramp)
                pygame.draw.rect(self.screen, col, cell.inflate(-1, -1))
                if show_numbers:
                    tc = WHITE if luminance(col) < 0.5 else (30, 30, 30)
                    g = self.pnum_surf(fmt(v), tc, fs)
                    self.screen.blit(g, g.get_rect(center=cell.center))

    # ---- fixed UI overlays (never affected by the camera) ----
    def _draw_dataset_panel(self):
        """Square panel (upper-right): the 2D dataset over a live decision-boundary
        heatmap; the current batch's samples are ringed (the same ones flowing as x).
        The wheel resizes the whole square when the cursor is over it (panel_geom)."""
        pg = self.panel_geom()
        if pg is None:  # attention mode: no 2D dataset panel
            return
        tr = self.trainer
        px, top, size = pg
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

    def _draw_seq_panel(self):
        """Attention mode (upper-right): the current sequence (sample 0) as a row of
        value cells with the true-max position ringed in magenta and the model's
        predicted position ringed in orange, plus the true vs predicted index — so the
        max marked on the attention-weights matrix has an obvious referent."""
        info = getattr(self.trainer, "attn_info", None)
        if info is None:
            return
        vals, mask, mp = info["seq_vals"], info["mask"], info["max_pos"]
        classification = info.get("task", "classification") == "classification"
        if classification:
            tp, pp = info["true_pos"], info["pred_pos"]
            hit = "✓" if pp == tp else "✗"
            label = f"sequence (sample 0)   ·   true max @ k{tp}   ·   pred @ k{pp}  {hit}"
        else:
            label = f"sequence (sample 0)   ·   true max {info['true_max']:.1f}   ·   pred {info['pred_max']:.2f}"
        s = len(vals)
        cell = min(64, max(34, (self.w - 40) // max(12, s)))
        panel_w = s * cell + 24
        x0, y0 = self.w - panel_w - 20, TOP_H + 40
        self.screen.blit(self.f_card.render(label, True, INK), (x0, y0 - 26))
        for j in range(s):
            cr = pygame.Rect(int(x0 + j * cell), int(y0), int(cell), int(cell))
            padded = mask[j] <= 0.5
            fill = (236, 238, 240) if padded else mix((255, 255, 255), MAX_MARK, 0.10)
            pygame.draw.rect(self.screen, fill, cr)
            pygame.draw.rect(self.screen, MUTED, cr, width=1)
            txt = "pad" if padded else f"{vals[j]:.0f}"
            g = self.f_info.render(txt, True, MUTED if padded else INK)
            self.screen.blit(g, g.get_rect(center=cr.center))
            kg = self.f_hint.render(f"k{j}", True, MUTED)  # key index under each cell
            self.screen.blit(kg, kg.get_rect(center=(cr.centerx, cr.bottom + 9)))
        if classification and 0 <= pp < s:  # ring the predicted position (orange), behind the true-max ring
            prect = pygame.Rect(int(x0 + pp * cell), int(y0), int(cell), int(cell))
            pygame.draw.rect(self.screen, BORDER_ACTIVE, prect.inflate(12, 12), width=3)
        mrect = pygame.Rect(int(x0 + mp * cell), int(y0), int(cell), int(cell))  # ring the true max
        pygame.draw.rect(self.screen, MAX_MARK, mrect.inflate(6, 6), width=3)

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
        pg = self.panel_geom()  # stop before the dataset panel (full width if none)
        avail_w = max(360, (pg[0] if pg else self.w) - x0 - 24)
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


def _check_common(app):
    """Checks shared by both modes: legible graph, forward reveal grows, grads reach
    every node, op-by-op advances training, equations rasterise, both views x 3
    matrix modes render without error."""
    for collapse in (False, True):
        sc = app.scenes[collapse]
        _check(sc.steps, f"{sc.view.title}: no steps")
        _check(sc.steps[0].active == {"x"}, f"{sc.view.title}: first focus {sc.steps[0].active}, expected x")
        for _nid, (w, h) in sc.sizes.items():
            _check(w > 0 and h > 0, "bad box")
        fwd = [s for s in sc.steps if s.phase == "forward"]
        bwd = [s for s in sc.steps if s.phase == "backward"]
        grows = all(len(fwd[i].revealed) <= len(fwd[i + 1].revealed) for i in range(len(fwd) - 1))
        _check(grows, f"{sc.view.title}: forward reveals not monotone")
        _check(len(bwd[-1].known) == len(sc.nodes), f"{sc.view.title}: grads did not reach all nodes")
        print(f"    scene '{sc.view.title}': {len(sc.nodes)} nodes, {len(sc.steps)} steps")

    w_before = app.trainer.model.parameters[0].data.copy()
    for _ in range(len(app.scene.steps)):
        app.advance_op()
    _check(not np.allclose(w_before, app.trainer.model.parameters[0].data), "weights did not update after a batch")
    print("    op-by-op advance: optimizer.step updates weights")

    eqs = {s.eq for sc in app.scenes.values() for s in sc.steps if s.eq}
    for eq in eqs:
        surf = render_equation(eq)
        _check(isinstance(surf, pygame.Surface) and surf.get_width() > 0 and surf.get_height() > 0, f"bad eq {eq}")
    print(f"    {len(eqs)} equations rasterised (mathtext)")

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
    print("    rendered both views x 3 matrix modes (no errors)")


def selftest() -> int:
    print("[selftest] pygame", pygame.version.ver)

    # ---------------- MLP mode (must keep working exactly as before) ----------------
    print("[selftest] MODEL_KIND='mlp' (moons) ...")
    app = App(headless=True, dataset="moons", model_kind="mlp")
    tr = app.trainer
    print(f"  arch {ARCH}, batch {BATCH}, train samples {tr.nb_samples}, {tr.nb_batches} batches/epoch")
    x0 = app.scenes[False].nodes["x"].mat.copy()
    _check(np.allclose(x0, tr.X_train[tr.batch_indices]), "x node != current batch data")
    _check(app.scenes[False].nodes["x"].mat.shape == (BATCH, 2), "x node wrong shape")
    _check_common(app)
    probs = tr.decision_grid(16)
    _check(probs.shape == (16, 16) and 0.0 <= float(probs.min()) and float(probs.max()) <= 1.0, "bad grid")
    mm = next(
        s for s in app.scenes[False].steps if s.phase == "backward" and s.vjp and s.vjp["target"].startswith("matmul")
    )
    _check(sum(1 for _, k, _ in mm.vjp["terms"] if k == "forward") == 2, "matmul VJP must expose two fwd operands")
    e0 = tr.epoch
    app.turbo = True
    for _ in range(20):
        app.update(0.016)
    app.turbo = False
    _check(tr.epoch > e0, "turbo did not advance an epoch")
    _check(app.boundary_surf is not None, "no boundary surface in mlp mode")
    print("  decision_grid ok, matmul VJP ok, turbo advanced an epoch + redrew boundary")

    # ---------------- ATTENTION mode (both tasks: classification + regression) ----------------
    for task in ("classification", "regression"):
        print(f"[selftest] MODEL_KIND='attention' task={task!r} ...")
        app = App(headless=True, model_kind="attention", task=task)
        tr = app.trainer
        print(f"  batch {ATTN_BATCH}, train samples {tr.nb_samples}, {tr.nb_batches} batches/epoch")
        det = app.scenes[False]
        hl = [k for k, ni in det.nodes.items() if ni.highlight]
        _check(len(hl) == 1, f"expected exactly one attention-weights node, got {hl}")
        wnode = det.nodes[hl[0]]
        _check(wnode.role == "softmax", "highlight node is not the softmax output")
        W = as2d(wnode.mat)  # sample-0 (seq_q, seq_k) attention weights
        _check(W.ndim == 2 and W.shape[0] == W.shape[1], f"attn weights not square (seq,seq): {W.shape}")
        _check(np.allclose(W.sum(axis=-1), 1.0, atol=1e-4), "attention rows do not sum to 1")
        _check(float(W[:, -1].max()) < 1e-3, "masked (padded) key column is not ~0")
        # the highlight (attention-weights) node uses bigger CELLS than an ordinary node
        cell_attn = det.sizes[hl[0]][0] / W.shape[1]
        cell_x = det.sizes["x"][0] / as2d(det.nodes["x"].mat).shape[1]
        _check(cell_attn > cell_x + 1, "attention-weights node is not drawn bigger (per cell)")
        _check("max_pos" in tr.attn_info and 0 <= tr.attn_info["max_pos"] < W.shape[1], "attn_info missing max_pos")
        _check(app.has_dataset_panel is False and app.panel_geom() is None, "attention should hide the dataset panel")
        print(f"  attention '{hl[0]}' {W.shape}: rows sum to 1, masked key ~0, cells {cell_attn:.0f}px > {cell_x:.0f}px")
        _check_common(app)

        # THE GOAL: after training, the model learns the task (score improves).
        before = tr.score()
        app.turbo = True
        for _ in range(120):  # ~700 batches of turbo training
            app.update(0.016)
        app.turbo = False
        app.trainer.prepare_current_batch()
        app._rebuild_scenes()
        after = tr.score()
        _check(tr.epoch > 0, "attention turbo did not advance an epoch")
        _check(after >= 0.5, f"{task} score too low after training ({after:.2f})")
        if task == "classification":  # classification genuinely learns from ~chance; regression
            _check(after > before, f"{task} score did not improve ({before:.2f} -> {after:.2f})")  # is near-perfect at init (value-only attention)
        metric = "max-position accuracy" if task == "classification" else "max-value R²"
        print(f"  TRAINED [{task}]: {metric} {before:.2f} -> {after:.2f}")

    pygame.quit()
    print("[selftest] PASS (mlp + attention[classification, regression])")
    return 0


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser(description="micrograd training visualiser")
    ap.add_argument("--model", choices=["mlp", "attention"], default=MODEL_KIND)
    ap.add_argument("--dataset", choices=["moons", "circles"], default=DATASET)
    ap.add_argument(
        "--task",
        choices=["classification", "regression"],
        default=TASK,
        help="attention only: classification (max position) or regression (max value)",
    )
    args = ap.parse_args()
    App(dataset=args.dataset, model_kind=args.model, task=args.task).run()


if __name__ == "__main__":
    main()
