"""the *logic* behind the MLP training visualiser (no rendering here).

This module owns everything about the neural network, its autograd, and the real
training run — independent of how it is drawn. It mirrors the repo exactly:

  * dataset / split like `examples/toy_classification.py`
    (make_moons / make_circles + train_test_split),
  * the training loop of `examples/train.py` (`fit`): Adam, BCE-with-logits,
    per-epoch shuffled mini-batches, `loss.backward()`, `optimizer.step()`.

For the *visualisation* it exposes the computation graph of one mini-batch at a
time, op by op: `BatchGraph` runs the real forward (capturing every layer output)
and then a stepped backward that calls each `Tensor._backward` closure one node at a
time — which is exactly `loss.backward()`, so the parameter grads it leaves behind
are the real ones the optimizer then consumes.

`Trainer` ties it together: it steps through batches/epochs of the real `fit`,
hands the current batch's `ViewGraph`s (detailed + compact) to the renderer, tracks
which dataset samples are in the batch, and evaluates the decision boundary on a grid
(in eval mode) after every epoch.

Nodes are identified by a **stable string key** (``"x"``, ``"W1"``, ``"matmul2"``,
``"Linear1"``, ``"BCE"``, …) rather than ``id()``, so the layout stays put while the
numbers change from batch to batch.

Only dependencies: numpy, scikit-learn, and the local micrograd package.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split

# micrograd lives next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from micrograd import Tensor
from micrograd.loss import binary_cross_entropy_with_logits
from micrograd.nn import MLP, Dropout, Linear
from micrograd.optim import Adam

__all__ = ["NodeInfo", "Step", "Trainer", "ViewGraph", "make_dataset"]

FRIENDLY = {"@": "matmul", "+": "add", "ReLU": "relu"}

# ---- equations (LaTeX), keyed by role -------------------------------------
FWD_EQ = {
    "matmul": r"$Z = X\,W$",
    "add": r"$Y = Z + b$",
    "relu": r"$y = \mathrm{relu}(x)$",
    "dropout": r"$y = \mathrm{mask}\odot x/(1-p)$",
    "loss": r"$L = \mathrm{BCE}(\mathrm{logits},\, y)$",
    "linear": r"$Y = \mathrm{relu}(X W + b)$",
    "linear_noact": r"$Y = X W + b$",
}
BWD_EQ = {
    "matmul": r"$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z}\,W^\top"
    r"\qquad \frac{\partial L}{\partial W} = X^\top\,\frac{\partial L}{\partial Z}$",
    "add": r"$\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial Y}"
    r"\qquad \frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial y}$",
    "relu": r"$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \mathbf{1}_{x>0}$",
    "dropout": r"$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\odot \mathrm{mask}/(1-p)$",
    "loss": r"$\frac{\partial L}{\partial \mathrm{logits}} = \frac{\sigma(\mathrm{logits}) - y}{N}$",
    "linear": r"$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z}\,W^\top"
    r"\qquad \frac{\partial L}{\partial W} = X^\top\,\frac{\partial L}{\partial Z}$",
}
SEED_EQ = r"$\frac{\partial L}{\partial L} = 1$"

FWD_CAP = {
    "matmul": "matrix product  X . W",
    "add": "add the bias  Z + b (broadcast over batch)",
    "relu": "ReLU activation  max(0, .)",
    "dropout": "Bernoulli mask + inverted scaling",
    "loss": "binary cross-entropy loss  ->  scalar",
    "linear": "affine map X.W + b, then ReLU",
    "linear_noact": "affine map X.W + b (no activation)",
}
BWD_CAP = {
    "matmul": "matmul VJP  ->  gradients flow to X and W",
    "add": "add VJP  ->  grad passes through; bias sums over the batch axis",
    "relu": "ReLU VJP  ->  gradient masked by x > 0",
    "dropout": "dropout VJP  ->  scaled by mask / (1 - p)",
    "loss": "BCE gradient w.r.t. the logits",
    "linear": "chained relu -> add -> matmul VJP  ->  grads to W, b, input",
}


# ======================================================================
#  Public data description (string-keyed; no coordinates)
# ======================================================================
@dataclass
class NodeInfo:
    id: str  # stable key: "x", "W1", "matmul2", "Linear1", "BCE", ...
    name: str  # same as id (shown as the node title)
    shape_str: str
    mat: np.ndarray  # this batch's data matrix (raw shape)
    role: str  # input|param|matmul|add|relu|dropout|loss|linear
    is_param: bool
    is_op: bool
    depth: int
    order: int


@dataclass
class Step:
    phase: str  # "forward" | "backward"
    active: set  # node keys highlighted this step
    revealed: set
    known: set
    grads: dict  # {key: gradient matrix}
    caption: str
    eq: str | None
    # backward only (detailed view): every matrix that the active op's VJP touches.
    #   {"target": active key, "children": [child keys that receive the grad],
    #    "terms": [(label, kind, matrix)]}  with kind in {incoming, forward, produced}
    vjp: dict | None = None


@dataclass
class ViewGraph:
    title: str  # "detailed" | "compact"
    nodes: dict  # {key: NodeInfo}
    order: list  # keys in draw order
    edges: list  # (child_key, parent_key)
    layer_groups: list = field(default_factory=list)
    steps: list = field(default_factory=list)


# ======================================================================
#  1. One mini-batch's real graph + stepped-backward snapshots
# ======================================================================
class BatchGraph:
    """Runs the real forward + a stepped backward for a single mini-batch on an
    existing model. The stepped backward *is* `loss.backward()` (it calls each
    node's `_backward` in reverse-topo order), so afterwards the parameter grads
    are exactly what the optimizer needs. We also snapshot the full gradient
    matrix at every op so the animation can replay them one node at a time."""

    def __init__(self, model: MLP, x_batch: np.ndarray, y_batch: np.ndarray):
        self.model = model
        self.y_batch = np.asarray(y_batch, dtype=float)  # labels, needed by the BCE VJP
        # real forward, capturing each layer's output tensor
        self.X = Tensor(x_batch)
        h = self.X
        layer_outputs = []
        for lay in model.layers:
            h = lay(h)
            layer_outputs.append(h)
        self.loss = binary_cross_entropy_with_logits(h, y_batch)

        # names + display order for the leaves (input + parameters)
        self.name = {id(self.X): "x"}
        self.order = {id(self.X): 0}
        self.param_ids: set[int] = set()
        k = 0
        for lay in model.layers:
            if isinstance(lay, Linear):
                k += 1
                self.name[id(lay.w)] = f"W{k}"
                self.name[id(lay.b)] = f"b{k}"
                self.order[id(lay.w)] = 2 * k - 1
                self.order[id(lay.b)] = 2 * k
                self.param_ids.update({id(lay.w), id(lay.b)})
        self.name[id(self.loss)] = "BCE"

        # topo order (children before parents)
        self.topo = self._build_topo(self.loss)
        self.by_id = {id(t): t for t in self.topo}
        self.ids = [id(t) for t in self.topo]

        # attach each internal op to its Linear layer (via captured outputs)
        self.node_layer: dict[int, int] = {}
        self.collapse_rep: dict[int, int] = {}
        self.col_name: dict[int, str] = {}
        self.col_dims: dict[int, tuple] = {}
        self.linear_layers = []
        lin_k = drop_i = 0
        for lay, out in zip(model.layers, layer_outputs, strict=True):
            if isinstance(lay, Linear):
                lin_k += 1
                g: set[int] = set()
                t = out
                if t.op == "ReLU":  # the activation belongs to the layer
                    self.node_layer[id(t)] = lin_k
                    g.add(id(t))
                    (t,) = tuple(t.children)  # relu -> add
                if t.op == "+":
                    self.node_layer[id(t)] = lin_k
                    g.add(id(t))
                    mm = next(c for c in t.children if c.op == "@")
                    self.node_layer[id(mm)] = lin_k
                    g.add(id(mm))
                for oid in g:
                    self.collapse_rep[oid] = id(out)
                self.col_name[id(out)] = f"Linear{lin_k}"
                self.col_dims[id(out)] = (int(lay.w.shape[0]), int(lay.w.shape[1]))
                self.linear_layers.append(
                    {"k": lin_k, "members": g | {id(lay.w), id(lay.b)}, "dims": self.col_dims[id(out)], "out": id(out)}
                )
            elif isinstance(lay, Dropout):
                drop_i += 1
                self.name[id(out)] = f"dropout{drop_i}"
                self.col_name[id(out)] = f"dropout{drop_i}"

        # depths: longest path from the leaves
        self.depth: dict[int, int] = {}
        for t in self.topo:
            self.depth[id(t)] = 0 if not t.children else 1 + max(self.depth[id(c)] for c in t.children)

        # constant raw data matrices for this batch
        self.data = {i: np.asarray(self.by_id[i].data, dtype=float) for i in self.ids}

        # REAL backward, one closure at a time, snapshotting full grads
        bw_nodes = sorted((t for t in self.topo if t.children), key=lambda t: -self.depth[id(t)])
        self.loss.grad = np.array(1.0)  # seed dL/dL = 1, exactly like Tensor.backward
        known = {id(self.loss)}
        seed_grad = {id(self.loss): np.array(self.loss.grad)}
        self.snaps = [{"active": id(self.loss), "known": set(known), "grads": seed_grad}]
        for t in bw_nodes:
            t._backward()
            for c in t.children:
                known.add(id(c))
            self.snaps.append(
                {"active": id(t), "known": set(known), "grads": {i: self.by_id[i].grad.copy() for i in known}}
            )

    @staticmethod
    def _build_topo(root):
        topo, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for c in t.children:
                visit(c)
            topo.append(t)

        visit(root)
        return topo

    def role(self, nid: int) -> str:
        t = self.by_id[nid]
        if t is self.loss:
            return "loss"
        if t.op in FRIENDLY:
            return FRIENDLY[t.op]
        if nid in self.param_ids:
            return "param"
        if self.name.get(nid, "").startswith("dropout"):
            return "dropout"
        return "input"

    def friendly(self, nid: int) -> str:
        t = self.by_id[nid]
        if nid in self.name:
            return self.name[nid]
        if t.op in FRIENDLY:
            k = self.node_layer.get(nid)
            return f"{FRIENDLY[t.op]}{k}" if k else FRIENDLY[t.op]
        return t.op or "?"

    @staticmethod
    def shape_str(shape) -> str:
        return "scalar" if shape == () else str(tuple(shape))


# ======================================================================
#  2. Turn a BatchGraph into (detailed, compact) string-keyed ViewGraphs
# ======================================================================
def _make_views(bg: BatchGraph) -> dict:
    return {False: _view(bg, collapse=False), True: _view(bg, collapse=True)}


def _view(bg: BatchGraph, collapse: bool) -> ViewGraph:
    if not collapse:
        node_ids = list(bg.ids)
        edges = [(id(c), id(t)) for t in bg.topo for c in t.children]
        depth_of = bg.depth
        title = "detailed"
    else:

        def rep(i):
            return bg.collapse_rep.get(i, i)

        ce = set()
        for t in bg.topo:
            for c in t.children:
                a, b = rep(id(c)), rep(id(t))
                if a != b:
                    ce.add((a, b))
        node_ids = sorted({n for e in ce for n in e})
        edges = list(ce)
        children = defaultdict(list)
        for a, b in ce:
            children[b].append(a)
        cdepth: dict[int, int] = {}

        def get_cdepth(n):
            if n not in cdepth:
                cdepth[n] = 1 + max((get_cdepth(c) for c in children[n]), default=-1)
            return cdepth[n]

        for n in node_ids:
            get_cdepth(n)
        depth_of = cdepth
        title = "compact"

    # stable string key for each node in this view
    def key(nid):
        if collapse and nid in bg.col_name:
            return bg.col_name[nid]
        return bg.friendly(nid)

    def role_of(nid):
        if collapse and nid in bg.col_name:
            return "linear" if bg.col_name[nid].startswith("Linear") else "dropout"
        return bg.role(nid)

    ops = {p for _, p in edges}
    nodes = {}
    for nid in node_ids:
        kk = key(nid)
        nodes[kk] = NodeInfo(
            id=kk,
            name=kk,
            shape_str=bg.shape_str(bg.by_id[nid].shape),
            mat=bg.data[nid],
            role=role_of(nid),
            is_param=nid in bg.param_ids,
            is_op=nid in ops,
            depth=depth_of[nid],
            order=bg.order.get(nid, 10_000),
        )

    edges_k = [(key(c), key(p)) for c, p in edges]

    layer_groups = []
    if not collapse:
        for i, info in enumerate(bg.linear_layers):
            din, dout = info["dims"]
            layer_groups.append(
                {
                    "members": {key(m) for m in info["members"] if m in node_ids},
                    "label": f"Linear {info['k']}  ({din}->{dout})",
                    "index": i,
                }
            )

    steps = _build_steps(bg, node_ids, edges, depth_of, collapse, key, role_of)
    return ViewGraph(
        title=title,
        nodes=nodes,
        order=[key(n) for n in node_ids],
        edges=edges_k,
        layer_groups=layer_groups,
        steps=steps,
    )


def _vjp_terms(bg, aid, snap, key):
    """Every matrix the active op's VJP touches, in equation order: the incoming
    gradient, the forward operand values it needs (kept available even in backward),
    and the produced gradients written into its children. `None` for a leaf.

    kind is one of {"incoming", "forward", "produced"} for colour/labelling.
    """
    t = bg.by_id[aid]
    children = list(t.children)
    if not children:
        return None
    g = snap["grads"]  # raw grad arrays by id, valid at this step
    incoming = g[aid]  # dL/d(output of this op) — computed by the parent earlier
    op = t.op

    if op == "@":  # Z = X @ W  ->  dL/dX = dL/dZ Wᵀ ;  dL/dW = Xᵀ dL/dZ
        wid = next(id(c) for c in children if id(c) in bg.param_ids)
        xid = next(id(c) for c in children if id(c) not in bg.param_ids)
        terms = [
            ("∂L/∂Z  (incoming)", "incoming", incoming),
            (f"X = {key(xid)}  (fwd)", "forward", bg.data[xid]),
            (f"W = {key(wid)}  (fwd)", "forward", bg.data[wid]),
            (f"∂L/∂X → {key(xid)}", "produced", g[xid]),
            (f"∂L/∂W → {key(wid)}", "produced", g[wid]),
        ]
    elif op == "+":  # Y = Z + b  ->  dL/dZ = dL/dY ;  dL/db = sum_batch dL/dY
        bid = next(id(c) for c in children if id(c) in bg.param_ids)
        zid = next(id(c) for c in children if id(c) not in bg.param_ids)
        terms = [
            ("∂L/∂Y  (incoming)", "incoming", incoming),
            (f"∂L/∂Z → {key(zid)}", "produced", g[zid]),
            (f"∂L/∂b → {key(bid)}  (Σ batch)", "produced", g[bid]),
        ]
    elif op == "ReLU":  # y = relu(x)  ->  dL/dx = dL/dy ⊙ 1[x>0]
        xid = id(children[0])
        terms = [
            ("∂L/∂y  (incoming)", "incoming", incoming),
            (f"x = {key(xid)}  (fwd, for 1[x>0])", "forward", bg.data[xid]),
            (f"∂L/∂x → {key(xid)}", "produced", g[xid]),
        ]
    elif t is bg.loss:  # BCE: dL/dlogits = (sigmoid(logits) - y) / N
        lid = id(children[0])
        terms = [
            ("∂L/∂L = 1  (incoming)", "incoming", incoming),
            (f"logits = {key(lid)}  (fwd)", "forward", bg.data[lid]),
            ("y  (labels)", "forward", bg.y_batch),
            (f"∂L/∂logits → {key(lid)}", "produced", g[lid]),
        ]
    else:  # generic fallback (e.g. dropout): incoming + produced
        terms = [("∂L/∂y  (incoming)", "incoming", incoming)]
        terms += [(f"∂L/∂x → {key(id(c))}", "produced", g[id(c)]) for c in children]

    return {"target": key(aid), "children": [key(id(c)) for c in children], "terms": terms}


def _build_steps(bg, node_ids, edges, depth_of, collapse, key, role_of):
    """Forward (reveal by depth, focus starts on x) then backward (from snapshots)."""
    steps: list[Step] = []
    revealed: set = set()
    dmax = max(depth_of[n] for n in node_ids)
    input_id = id(bg.X)
    for d in range(dmax + 1):
        ids_d = [n for n in node_ids if depth_of[n] == d]
        revealed = revealed | {key(n) for n in ids_d}
        if d == 0:
            cap = "FORWARD · start from the input x (W, b are ready too)"
            steps.append(Step("forward", {key(input_id)}, set(revealed), set(), {}, cap, None))
            continue
        n0 = ids_d[0]
        role = role_of(n0)
        if role == "linear" and bg.by_id[n0].op == "+":  # last Linear: no ReLU
            cap, eq = f"FORWARD · {key(n0)}: {FWD_CAP['linear_noact']}", FWD_EQ["linear_noact"]
        else:
            cap, eq = f"FORWARD · {key(n0)}: {FWD_CAP.get(role, 'compute')}", FWD_EQ.get(role)
        steps.append(Step("forward", {key(n) for n in ids_d}, set(revealed), set(), {}, cap, eq))

    allids = set(node_ids)
    allkeys = {key(n) for n in node_ids}
    if not collapse:
        for i, s in enumerate(bg.snaps):
            aid = s["active"]
            kn = {key(x) for x in s["known"]}
            grads = {key(x): m for x, m in s["grads"].items()}
            if i == 0:
                cap, eq, vjp = "BACKWARD · seed the gradient at the root", SEED_EQ, None
            else:
                role = role_of(aid)
                cap, eq = f"BACKWARD · {key(aid)}: {BWD_CAP.get(role, '')}", BWD_EQ.get(role)
                vjp = _vjp_terms(bg, aid, s, key)  # every matrix this VJP touches
            steps.append(Step("backward", {key(aid)}, set(allkeys), kn, grads, cap, eq, vjp=vjp))
    else:
        contained = defaultdict(set)
        for oid, r in bg.collapse_rep.items():
            contained[r].add(oid)
        s0 = bg.snaps[0]
        kn0 = {key(x) for x in (s0["known"] & allids)}
        g0 = {key(x): s0["grads"][x] for x in (s0["known"] & allids)}
        steps.append(
            Step(
                "backward",
                {key(id(bg.loss))},
                set(allkeys),
                kn0,
                g0,
                "BACKWARD · seed the gradient at the root",
                SEED_EQ,
            )
        )
        col_children = defaultdict(list)
        for a, b in edges:
            col_children[b].append(a)
        col_bw = sorted((n for n in node_ids if col_children[n]), key=lambda n: -depth_of[n])
        for n in col_bw:
            cont = contained.get(n) or {n}
            b = max(i for i, s in enumerate(bg.snaps) if s["active"] in cont)
            s = bg.snaps[b]
            inter = s["known"] & allids
            kn = {key(x) for x in inter}
            grads = {key(x): s["grads"][x] for x in inter}
            role = role_of(n)
            if n == id(bg.loss):
                cap, eq = f"BACKWARD · {key(n)}: {BWD_CAP['loss']}", BWD_EQ["loss"]
            else:
                cap, eq = f"BACKWARD · {key(n)}: {BWD_CAP.get(role, '')}", BWD_EQ.get(role)
            steps.append(Step("backward", {key(n)}, set(allkeys), kn, grads, cap, eq))
    return steps


# ======================================================================
#  3. Dataset (mirrors examples/toy_classification.make_dataset)
# ======================================================================
def make_dataset(name: str, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    match name:
        case "moons":
            return cast("tuple[np.ndarray, np.ndarray]", make_moons(n_samples=n_samples, noise=0.2))
        case "circles":
            return cast("tuple[np.ndarray, np.ndarray]", make_circles(n_samples=n_samples, noise=0.1, factor=0.5))
        case _:
            raise ValueError(f"Unknown dataset: {name}")


# ======================================================================
#  4. Trainer — runs the real fit loop, exposing it batch by batch
# ======================================================================
@dataclass
class TrainState:
    epoch: int
    batch_in_epoch: int
    nb_batches: int
    batch_loss: float


class Trainer:
    """Drives the real `fit` loop from examples/train.py, one mini-batch at a
    time, so the visualiser can animate each batch op-by-op and redraw the
    decision boundary once an epoch completes."""

    def __init__(self, dataset, arch, batch_size, seed, dropout, lr, n_samples):
        self.dataset = dataset
        self.arch = arch
        self.batch_size = batch_size
        self.seed = seed
        self.dropout = dropout
        self.lr = lr
        self.n_samples = n_samples
        self._setup()

    def _setup(self):
        np.random.seed(self.seed)  # deterministic dataset + init for the demo
        X, y = make_dataset(self.dataset, self.n_samples)
        # exactly toy_classification's split (train 70% / val 15% / test 15%)
        X_train, X_tv, y_train, y_tv = train_test_split(X, y, test_size=0.3)
        X_test, X_val, y_test, y_val = train_test_split(X_tv, y_tv, test_size=0.5)

        def col(a):
            return np.array(a).reshape(len(a), 1)

        # trim the training set to a whole number of batches -> every batch is the
        # same shape, so the graph layout never jumps
        n = (len(X_train) // self.batch_size) * self.batch_size
        self.X_train, self.y_train = np.array(X_train)[:n], col(y_train)[:n]
        self.X_val, self.y_val = np.array(X_val), col(y_val)
        self.X_test, self.y_test = np.array(X_test), col(y_test)

        allX = np.vstack([self.X_train, self.X_val, self.X_test])
        pad = 0.5
        self.extent = (
            allX[:, 0].min() - pad,
            allX[:, 0].max() + pad,
            allX[:, 1].min() - pad,
            allX[:, 1].max() + pad,
        )

        self.model = MLP(*self.arch, dropout=self.dropout)
        self.optimizer = Adam(parameters=self.model.parameters, lr=self.lr)
        self.criterion = binary_cross_entropy_with_logits

        self.nb_samples = len(self.X_train)
        self.nb_batches = self.nb_samples // self.batch_size
        self.epoch = 0
        self._pending = False  # is an optimizer.step owed for the shown batch?
        self._new_epoch()
        self.prepare_current_batch()

    # ---- epoch / batch bookkeeping (mirrors fit's shuffled batches) ----
    def _new_epoch(self):
        self.perm = np.random.permutation(self.nb_samples)
        self.cursor = 0
        self.batch_in_epoch = 0

    def _slice(self):
        idx = self.perm[self.cursor : self.cursor + self.batch_size]
        return idx, self.X_train[idx], self.y_train[idx]

    def _advance_cursor(self) -> bool:
        """Move to the next batch; return True if a new epoch just started."""
        self.cursor += self.batch_size
        self.batch_in_epoch += 1
        if self.cursor >= self.nb_samples:
            self.epoch += 1
            self._new_epoch()
            return True
        return False

    # ---- op-by-op path: build the shown batch's graph (grads ready, no step) ----
    def prepare_current_batch(self):
        idx, xb, yb = self._slice()
        self.batch_indices = idx
        self.optimizer.zero_grad()
        bg = BatchGraph(self.model, xb, yb)  # forward + stepped backward (real grads)
        self.views = _make_views(bg)
        self.batch_loss = float(bg.loss.data)
        self._pending = True

    def advance_batch(self) -> bool:
        """Apply the shown batch's update, then prepare the next one.
        Returns True if this crossed an epoch boundary."""
        if self._pending:
            self.optimizer.step()
            self._pending = False
        new_epoch = self._advance_cursor()
        self.prepare_current_batch()
        return new_epoch

    # ---- fast path (turbo): train whole batches with no snapshots ----
    def fast_train_batch(self) -> bool:
        if self._pending:  # finish the shown batch first
            self.optimizer.step()
            self._pending = False
            return self._advance_cursor()
        _idx, xb, yb = self._slice()
        self.optimizer.zero_grad()
        logits = self.model(Tensor(xb))
        self.criterion(logits, yb).backward()
        self.optimizer.step()
        return self._advance_cursor()

    def state(self) -> TrainState:
        return TrainState(self.epoch, self.batch_in_epoch, self.nb_batches, self.batch_loss)

    # ---- decision boundary on a grid, in eval mode (like toy_classification) ----
    def decision_grid(self, res: int = 64):
        x_min, x_max, y_min, y_max = self.extent
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
        grid = np.c_[xx.ravel(), yy.ravel()]
        with self.model.eval():
            logits = self.model(Tensor(grid))
        probs = 1.0 / (1.0 + np.exp(-logits.data))
        return probs.reshape(res, res)  # [iy, ix], values in (0, 1)

    def restart(self):
        self._setup()
