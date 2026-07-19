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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split

# micrograd lives one level up from this file (repo root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import micrograd.nn as nnmod
import micrograd.operations as opsmod
from micrograd import Tensor
from micrograd.loss import binary_cross_entropy_with_logits, mse
from micrograd.nn import MLP, Dropout, Linear
from micrograd.optim import Adam

__all__ = ["AttentionTrainer", "NodeInfo", "Step", "Trainer", "ViewGraph", "make_trainer"]

FRIENDLY = {"@": "matmul", "+": "add", "ReLU": "relu", "swapaxes": "transpose", "*": "scale", "softmax": "softmax"}

# ======================================================================
#  OP TAGGING  (viz-only; the micrograd Tensor class carries no op label)
# ======================================================================
# The "clean" Tensor class has no `.op` field, so we recover the op that produced
# each tensor by *temporarily* wrapping the Tensor methods (and softmax) during the
# forward pass and recording a label into this id-keyed dict. Nothing in the library
# changes; the tags live only here and are read while building the graph.
_OP_TAG: dict[int, str] = {}
_OP_ARGS: dict[int, tuple] = {}  # id(out) -> ordered ids of its Tensor operands


def op_of(t: Tensor) -> str:
    """The op that produced tensor `t` ("" for a leaf), from the tagging pass."""
    return _OP_TAG.get(id(t), "")


def operands_of(t: Tensor) -> list[int]:
    """Child ids of `t` in operand order (left, right), recovered from the tagging
    pass — `t.children` is an unordered set, but matmul's VJP needs A vs B."""
    child_ids = {id(c) for c in t.children}
    ordered = [i for i in _OP_ARGS.get(id(t), ()) if i in child_ids]
    ordered += [i for i in child_ids if i not in ordered]  # any extras, arbitrary order
    return ordered


@contextmanager
def tag_ops():
    """Run a forward inside this context so every produced Tensor records its op
    (and its operand order). Nothing else is changed; the tags live only here."""
    _OP_TAG.clear()
    _OP_ARGS.clear()
    patches: list = []

    def wrap(owner, attr, tag):
        orig = getattr(owner, attr)

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            _OP_TAG[id(out)] = tag
            _OP_ARGS[id(out)] = tuple(id(a) for a in args if isinstance(a, Tensor))
            return out

        patches.append((owner, attr, orig))
        setattr(owner, attr, wrapped)

    wrap(Tensor, "__matmul__", "@")
    wrap(Tensor, "__add__", "+")
    wrap(Tensor, "relu", "ReLU")
    wrap(Tensor, "swapaxes", "swapaxes")
    wrap(Tensor, "squeeze", "squeeze")
    wrap(Tensor, "__mul__", "*")
    wrap(Tensor, "__truediv__", "/")
    # softmax is a module function; nn.py did `from .operations import softmax`, so the
    # name is bound in both modules — patch both so either call site gets tagged.
    wrap(opsmod, "softmax", "softmax")
    wrap(nnmod, "softmax", "softmax")
    try:
        yield
    finally:
        for owner, attr, orig in patches:
            setattr(owner, attr, orig)


# ---- equations (LaTeX), keyed by role -------------------------------------
FWD_EQ = {
    "matmul": r"$Z = A\,B$",
    "add": r"$Y = Z + b$",
    "relu": r"$y = \mathrm{relu}(x)$",
    "dropout": r"$y = \frac{1}{1-p}\mathrm{mask}\odot x$",
    "loss": r"$L = \mathrm{BCE}(\mathrm{logits},\, y)$",
    "linear": r"$Y = \mathrm{relu}(X W + b)$",
    "linear_noact": r"$Y = X W + b$",
    # ---- attention ----
    "transpose": r"$K^\top$",
    "scale": r"$\frac{Q K^\top}{\sqrt{d_k}}$",
    "logitsmask": r"$\text{logits} + (1-\text{mask})\times(-10^9)$",
    "softmax_attn": r"$A = \mathrm{softmax}_{\mathrm{keys}}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)$",
    "ce_loss": r"$L = \mathrm{CE}(\mathrm{logits},\, y)$",
    "attention": r"$\mathrm{ctx} = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$",
    "pool": r"$\mathrm{pool} = \frac{1}{|V|}\sum_{i \in V} \mathrm{ctx}_i$",
    "mse_loss": r"$L = \frac{1}{N}\sum (\hat{y} - y_{\max})^2$",
}
BWD_EQ = {
    "matmul": r"$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Z}\,B^\top"
    r"\qquad \frac{\partial L}{\partial B} = A^\top\,\frac{\partial L}{\partial Z}$",
    "add": r"$\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial Y}"
    r"\qquad \frac{\partial L}{\partial b} = \sum_{\text{batch}} \frac{\partial L}{\partial y}$",
    "relu": r"$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \mathbf{1}_{x>0}$",
    "dropout": r"$\frac{\partial L}{\partial x} = \frac{1}{1-p}\frac{\partial L}{\partial y}\odot \mathrm{mask}$",
    "loss": r"$\frac{\partial L}{\partial \mathrm{logits}} = \frac{\sigma(\mathrm{logits}) - y}{N}$",
    "linear": r"$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z}\,W^\top"
    r"\qquad \frac{\partial L}{\partial W} = X^\top\,\frac{\partial L}{\partial Z}$",
    # ---- attention ----
    "transpose": r"$\frac{\partial L}{\partial K} = \left(\frac{\partial L}{\partial K^\top}\right)^{\top}$",
    "scale": r"$\frac{\partial L}{\partial (QK^\top)} = \frac{1}{\sqrt{d_k}}\frac{\partial L}{\partial y}$",
    "softmax": r"$\frac{\partial L}{\partial s} = A \odot \left(\frac{\partial L}{\partial A}"
    r" - \sum_k \frac{\partial L}{\partial A}\odot A\right)$",
    "ce_loss": r"$\frac{\partial L}{\partial \mathrm{logits}} ="
    r" \frac{\mathrm{softmax}(\mathrm{logits}) - \mathrm{onehot}(y)}{N}$",
    "logitsmask": r"$\frac{\partial L}{\partial \mathrm{logits}} = \frac{\partial L}{\partial y}$",
    "attention": r"$\frac{\partial L}{\partial x},\ \frac{\partial L}{\partial W_{q,k,v}}"
    r" \leftarrow \text{through } A,\, V$",
    "pool": r"$\frac{\partial L}{\partial \mathrm{ctx}_i} = \frac{1}{|V|}\frac{\partial L}{\partial \mathrm{pool}}$",
    "mse_loss": r"$\frac{\partial L}{\partial \hat{y}} = \frac{2}{N}(\hat{y} - y_{\max})$",
}
SEED_EQ = r"$\frac{\partial L}{\partial L} = 1$"

FWD_CAP = {
    "matmul": "matrix product  A . B",
    "add": "add the bias  Z + b (broadcast over batch)",
    "relu": "ReLU activation  max(0, .)",
    "dropout": "Bernoulli mask + inverted scaling",
    "loss": "binary cross-entropy loss  ->  scalar",
    "linear": "affine map X.W + b, then ReLU",
    "linear_noact": "affine map X.W + b (no activation)",
    # ---- attention ----
    "transpose": "transpose keys  K -> Kᵀ  (swap last two axes)",
    "scale": "raw scores QKᵀ scaled by 1/sqrt(d_k)",
    "logitsmask": "add -1e9 to padded positions so they vanish after softmax",
    "attn_weights": "ATTENTION WEIGHTS: row i = how much query i attends to each key; masked keys ~ 0",
    "ce_loss": "softmax cross-entropy over positions  ->  scalar",
    "softmax": "softmax over keys (last axis)",
    "attention": "self-attention block  (collapsed)",
    "squeeze": "drop the trailing dim  ->  per-position logits",
    "pool": "mean-pool the context over valid positions",
    "mse_loss": "mean squared error vs the true max value  ->  scalar",
}
BWD_CAP = {
    "matmul": "matmul VJP  ->  gradients flow to A and B",
    "add": "add VJP  ->  grad passes through; bias sums over the batch axis",
    "relu": "ReLU VJP  ->  gradient masked by x > 0",
    "dropout": "dropout VJP  ->  scaled by mask / (1 - p)",
    "loss": "BCE gradient w.r.t. the logits",
    "linear": "chained relu -> add -> matmul VJP  ->  grads to W, b, input",
    # ---- attention ----
    "transpose": "transpose VJP  ->  gradient is transposed back",
    "scale": "scale VJP  ->  gradient divided by sqrt(d_k)",
    "softmax": "softmax VJP over keys  ->  gradient to the scaled scores",
    "ce_loss": "cross-entropy gradient w.r.t. the position logits",
    "logitsmask": "mask-add VJP  ->  gradient passes to logits (mask constant is fixed)",
    "attention": "self-attention VJP  ->  grads to x and W_q, W_k, W_v",
    "pool": "mean-pool VJP  ->  gradient split evenly over valid positions",
    "mse_loss": "MSE gradient w.r.t. the predicted max value",
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
    role: str  # input|param|matmul|add|relu|dropout|loss|linear|softmax|...
    is_param: bool
    is_op: bool
    depth: int
    order: int
    highlight: bool = False  # the attention-weights node: rendered as the centerpiece


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
        self.highlight: set[int] = set()  # no special centerpiece node in MLP mode
        # real forward, capturing each layer's output tensor (op-tagged for the viz)
        self.X = Tensor(x_batch)
        with tag_ops():
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
        self.by_id = {id(t): t for t in self._build_topo(self.loss)}

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
                if op_of(t) == "ReLU":  # the activation belongs to the layer
                    self.node_layer[id(t)] = lin_k
                    g.add(id(t))
                    (t,) = tuple(t.children)  # relu -> add
                if op_of(t) == "+":
                    self.node_layer[id(t)] = lin_k
                    g.add(id(t))
                    mm = next(c for c in t.children if op_of(c) == "@")
                    self.node_layer[id(mm)] = lin_k
                    g.add(id(mm))
                for oid in g:
                    self.collapse_rep[oid] = id(out)
                self.col_name[id(out)] = f"Linear{lin_k}"
                self.col_dims[id(out)] = (int(lay.w.shape[0]), int(lay.w.shape[1]))
                din, dout = self.col_dims[id(out)]
                self.linear_layers.append(
                    {
                        "k": lin_k,
                        "members": g | {id(lay.w), id(lay.b)},
                        "label": f"Linear {lin_k}  ({din}->{dout})",
                        "out": id(out),
                    }
                )
            elif isinstance(lay, Dropout):
                drop_i += 1
                self.name[id(out)] = f"dropout{drop_i}"
                self.col_name[id(out)] = f"dropout{drop_i}"

        self._finalize()

    def _finalize(self):
        """Shared machinery: topo order, depths, data matrices, and the stepped
        backward snapshots. Runs once self.loss (+ names/groups) are set. Model-
        agnostic, so both the MLP and the attention builders reuse it."""
        self.topo = self._build_topo(self.loss)
        self.by_id = {id(t): t for t in self.topo}
        self.ids = [id(t) for t in self.topo]

        # depths: longest path from the leaves
        self.depth = {}
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
            return getattr(self, "loss_role", "loss")
        if hasattr(self, "role_by_id") and nid in self.role_by_id:  # explicit (attention)
            return self.role_by_id[nid]
        op = op_of(t)
        if op in FRIENDLY:
            return FRIENDLY[op]
        if nid in self.param_ids:
            return "param"
        if self.name.get(nid, "").startswith("dropout"):
            return "dropout"
        return "input"

    def friendly(self, nid: int) -> str:
        t = self.by_id[nid]
        if nid in self.name:  # leaves, loss, and (attention) explicitly-named ops
            return self.name[nid]
        op = op_of(t)
        if op in FRIENDLY:
            k = self.node_layer.get(nid)
            return f"{FRIENDLY[op]}{k}" if k else FRIENDLY[op]
        return op or "?"

    @staticmethod
    def shape_str(shape) -> str:
        if shape == ():
            return "scalar"
        return f"{tuple(shape)} b0" if len(shape) >= 3 else str(tuple(shape))


# ======================================================================
#  1b. Attention: one mini-batch's real MaxClassificationModel graph
# ======================================================================
class AttentionBatchGraph(BatchGraph):
    """Same public interface as BatchGraph, but built from the repo's real
    `MaxRegressionModel` (attention -> mean-pool -> linears -> the max VALUE). We
    replay the exact forward math of `Attention.__call__` / the model on the model's
    real parameters (so `optimizer.step` trains it), capture every intermediate to
    name them, and flag the softmax attention-weights node as the centerpiece.

    Why regression (predict the max value) and not classification (predict the max
    position): the classification model reaches 100% accuracy with ~uniform attention
    — it never needs to point attention at the max, so there is nothing to see. To
    output the max value the pooled context must equal the max token's value, which
    forces attention to concentrate on the max position — exactly what we want to
    watch appear in the scores. Reuses BatchGraph's `_finalize`, `role`, `friendly`."""

    def __init__(self, model, x_batch, y_target, mask):
        self.model = model
        self.y_batch = np.asarray(y_target, dtype=float)  # the max VALUE target (batch, 1)
        self.mask = np.asarray(mask, dtype=float)
        self.loss_role = "mse_loss"
        self.name, self.order, self.role_by_id = {}, {}, {}
        self.param_ids, self.node_layer = set(), {}
        self.collapse_rep, self.col_name, self.col_dims = {}, {}, {}
        self.linear_layers = []
        self.highlight: set[int] = set()
        self._order_ctr = 0

        attn = model.attention
        b, s = self.mask.shape

        # ---- real forward, mirroring Attention.__call__ + MaxRegressionModel head ----
        # (predict the max VALUE: attention must PUT WEIGHT ON THE MAX POSITION to read
        # it, so the attention scores end up pointing at the max — the whole goal here.)
        with tag_ops():
            x = Tensor(np.array(x_batch, dtype=float))
            self.X = x
            q = x @ attn.w_q
            k = x @ attn.w_k
            v = x @ attn.w_v
            kt = k.swapaxes(-1, -2)
            scores = q @ kt
            mask_k = Tensor((1 - self.mask.reshape(b, 1, s)) * -1e9)  # -inf on padded keys
            masked = scores + mask_k
            # scale by 1/sqrt(d_k) as a FULL-shape constant, not a scalar: micrograd's
            # __mul__ backward mishandles bigtensor * scalar-Tensor (broadcast into a
            # 0-d grad). A full-shape factor is identical maths and sidesteps it.
            inv = Tensor(np.full(masked.data.shape, 1.0 / np.sqrt(float(attn.d_k))))
            scaled = masked * inv
            weights = opsmod.softmax(scaled, axis=-1)  # <-- attention weights (the centerpiece)
            ctx = weights @ v
            pooled = model.mean_pool(ctx, self.mask)  # average context over valid positions
            h = pooled
            lin_outs = []
            for lin in model.linears:
                h = lin(h)
                lin_outs.append(h)
            self.pred = np.asarray(h.data, dtype=float)  # predicted max value (batch, 1)
            self.loss = mse(h, self.y_batch)

        # ---- max-position info (for the renderer's max marker + sequence readout) ----
        raw = np.asarray(x_batch, dtype=float)[:, :, 0]  # raw sequence values (batch, seq)
        self.seq_vals = raw
        self.max_pos = np.argmax(np.where(self.mask > 0, raw, -np.inf), axis=1)  # (batch,)

        # ---- name leaves / ops, mark groups + the highlight node ----
        self._leaf(x, "x")
        self._leaf(attn.w_q, "Wq", param=True)
        self._leaf(attn.w_k, "Wk", param=True)
        self._leaf(attn.w_v, "Wv", param=True)
        self._leaf(mask_k, "mask(K)")
        self._leaf(inv, "1/sqrt(dk)")  # the 1/sqrt(d_k) scale constant

        self._op(q, "Q", "matmul")
        self._op(k, "K", "matmul")
        self._op(v, "V", "matmul")
        self._op(kt, "K^T", "transpose")
        self._op(scores, "QK^T", "matmul")
        self._op(masked, "QK^T+mask", "add")
        self._op(scaled, "scaled", "scale")
        self._op(weights, "attn", "softmax")
        self.highlight.add(id(weights))  # THE centerpiece
        self._op(ctx, "context", "matmul")
        self._op(pooled, "pool", "pool")

        attn_ops = [q, k, v, kt, scores, masked, scaled, weights, ctx]
        for t in attn_ops:
            self.collapse_rep[id(t)] = id(pooled)
        self.col_name[id(pooled)] = "Attention"
        self.linear_layers.append(
            {
                "k": 0,
                "members": {id(t) for t in [*attn_ops, pooled, attn.w_q, attn.w_k, attn.w_v, mask_k]},
                "label": f"Attention + pool  (d_k={attn.d_k}, d_v={attn.d_v})",
                "out": id(pooled),
            }
        )

        for i, (lin, out) in enumerate(zip(model.linears, lin_outs, strict=True), start=1):
            members: set[int] = set()
            t = out
            if op_of(t) == "ReLU":
                self._op(t, f"reluL{i}", "relu")
                members.add(id(t))
                (t,) = tuple(t.children)
            if op_of(t) == "+":
                self._op(t, f"addL{i}", "add")
                members.add(id(t))
                mm = next(c for c in t.children if op_of(c) == "@")
                self._op(mm, f"matmulL{i}", "matmul")
                members.add(id(mm))
            for oid in members:
                self.collapse_rep[oid] = id(out)
            self.col_name[id(out)] = f"Linear{i}"
            self._leaf(lin.w, f"Wl{i}", param=True)
            self._leaf(lin.b, f"bl{i}", param=True)
            members |= {id(lin.w), id(lin.b)}
            din, dout = int(lin.w.shape[0]), int(lin.w.shape[1])
            self.linear_layers.append(
                {"k": i, "members": members, "label": f"Linear {i}  ({din}->{dout})", "out": id(out)}
            )

        self.name[id(self.loss)] = "MSE"
        self._finalize()

    def _leaf(self, t, nm, param=False):
        self.name[id(t)] = nm
        self.order[id(t)] = self._order_ctr
        self._order_ctr += 1
        if param:
            self.param_ids.add(id(t))

    def _op(self, t, nm, role):
        self.name[id(t)] = nm
        self.role_by_id[id(t)] = role


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
            nm = bg.col_name[nid]
            if nm.startswith("Linear"):
                return "linear"
            if nm.startswith("Attention"):
                return "attention"
            return "dropout"
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
            highlight=nid in bg.highlight,  # attention-weights centerpiece
        )

    edges_k = [(key(c), key(p)) for c, p in edges]

    layer_groups = []
    if not collapse:
        for i, info in enumerate(bg.linear_layers):
            layer_groups.append(
                {
                    "members": {key(m) for m in info["members"] if m in node_ids},
                    "label": info["label"],
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
    op = op_of(t)

    if op == "@":  # Z = A @ B  ->  dL/dA = dL/dZ Bᵀ ;  dL/dB = Aᵀ dL/dZ
        oid = operands_of(t)
        a_id, b_id = (oid[0], oid[1]) if len(oid) >= 2 else (id(children[0]), id(children[-1]))
        terms = [
            ("∂L/∂Z  (incoming)", "incoming", incoming),
            (f"A = {key(a_id)}  (fwd)", "forward", bg.data[a_id]),
            (f"B = {key(b_id)}  (fwd)", "forward", bg.data[b_id]),
            (f"∂L/∂A → {key(a_id)}", "produced", g[a_id]),
            (f"∂L/∂B → {key(b_id)}", "produced", g[b_id]),
        ]
    elif op == "+":  # Y = Σ operands  ->  grad passes to each (summed where broadcast)
        terms = [("∂L/∂Y  (incoming)", "incoming", incoming)]
        for cid in (id(c) for c in children):
            broadcast = bg.by_id[cid].data.shape != t.data.shape
            note = "  (Σ broadcast)" if broadcast else ""
            terms.append((f"∂L/∂{key(cid)} → {key(cid)}{note}", "produced", g[cid]))
    elif op == "ReLU":  # y = relu(x)  ->  dL/dx = dL/dy ⊙ 1[x>0]
        xid = id(children[0])
        terms = [
            ("∂L/∂y  (incoming)", "incoming", incoming),
            (f"x = {key(xid)}  (fwd, for 1[x>0])", "forward", bg.data[xid]),
            (f"∂L/∂x → {key(xid)}", "produced", g[xid]),
        ]
    elif op == "softmax":  # A = softmax(s) ->  dL/ds = A * (dL/dA - sum_k(dL/dA * A))
        sid = id(children[0])
        terms = [
            ("∂L/∂A  (incoming)", "incoming", incoming),
            (f"A = {key(aid)}  (fwd weights)", "forward", bg.data[aid]),
            (f"∂L/∂s → {key(sid)}", "produced", g[sid]),
        ]
    elif t is bg.loss:  # cross-entropy: dL/dlogits = (softmax(logits) - onehot(y)) / N
        lid = id(children[0])
        terms = [
            ("∂L/∂L = 1  (incoming)", "incoming", incoming),
            (f"logits = {key(lid)}  (fwd)", "forward", bg.data[lid]),
            ("y  (labels)", "forward", np.asarray(bg.y_batch, dtype=float)),
            (f"∂L/∂logits → {key(lid)}", "produced", g[lid]),
        ]
    else:  # generic (scale, transpose, squeeze, dropout, ...): incoming + produced
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
        if n0 in bg.highlight:  # the attention-weights node: spell out what it means
            cap, eq = f"FORWARD · {key(n0)}: {FWD_CAP['attn_weights']}", FWD_EQ["softmax_attn"]
        elif role == "linear" and op_of(bg.by_id[n0]) == "+":  # last Linear: no ReLU
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


# ======================================================================
#  5. Attention trainer — real MaxClassificationModel, one mini-batch at a time
# ======================================================================
def _attention_pred(model, x_batch: np.ndarray, mask: np.ndarray) -> Tensor:
    """Connected forward for the fast/turbo path (mirrors Attention.__call__ +
    MaxRegressionModel head -> the predicted max value). Full-shape 1/sqrt(d_k)
    factor avoids the __mul__ scalar-broadcast bug."""
    attn = model.attention
    b, s = mask.shape
    x = Tensor(np.array(x_batch, dtype=float))
    scores = (x @ attn.w_q) @ (x @ attn.w_k).swapaxes(-1, -2)
    masked = scores + Tensor((1 - mask.reshape(b, 1, s)) * -1e9)
    scaled = masked * Tensor(np.full(masked.data.shape, 1.0 / np.sqrt(float(attn.d_k))))
    ctx = opsmod.softmax(scaled, axis=-1) @ (x @ attn.w_v)
    h = model.mean_pool(ctx, mask)
    for lin in model.linears:
        h = lin(h)
    return h


class AttentionTrainer:
    """Mirror of `Trainer` for the attention model (real `MaxRegressionModel`,
    predicting the max value). Same public surface the renderer uses (views / state /
    advance_batch / fast_train_batch / prepare_current_batch / restart). No 2D
    decision boundary (sequence task) — instead it exposes `attn_info`: the current
    sample's sequence values + true/pred max, so the renderer can mark the max on the
    attention-weights matrix."""

    HIGH = 10
    MIN_SEQ_LEN = 4
    MAX_SEQ_LEN = 5  # 4 valid + 1 padded key position (so masking is visible)

    def __init__(self, batch_size, seed, lr, n_samples):
        self.batch_size = batch_size
        self.seed = seed
        self.lr = lr
        self.n_samples = n_samples
        self._setup()

    def _setup(self):
        # the example pulls in matplotlib; import lazily to keep core light
        from examples.attention.maximum_regression import (
            MaxRegressionModel,
            make_dataset as make_seq_dataset,
        )

        np.random.seed(self.seed)
        X, _y_pos, mask = make_seq_dataset(
            n_samples=self.n_samples,
            high=self.HIGH,
            min_seq_len=self.MIN_SEQ_LEN,
            max_seq_len=self.MAX_SEQ_LEN,
        )
        # regression target: the max VALUE among valid positions (batch, 1)
        y = (X[:, :, 0] * mask).max(axis=1, keepdims=True)
        n = (len(X) // self.batch_size) * self.batch_size  # whole batches only
        self.X_train, self.y_train, self.mask_train = X[:n], y[:n], mask[:n]

        self.model = MaxRegressionModel(n_in=1, d_k=8, d_v=8, hidden_dims=[8], out_dim=1)
        self.optimizer = Adam(parameters=self.model.parameters, lr=self.lr)

        self.nb_samples = len(self.X_train)
        self.nb_batches = self.nb_samples // self.batch_size
        self.epoch = 0
        self._pending = False
        self._new_epoch()
        self.prepare_current_batch()

    def _new_epoch(self):
        self.perm = np.random.permutation(self.nb_samples)
        self.cursor = 0
        self.batch_in_epoch = 0

    def _slice(self):
        idx = self.perm[self.cursor : self.cursor + self.batch_size]
        return idx, self.X_train[idx], self.y_train[idx], self.mask_train[idx]

    def _advance_cursor(self) -> bool:
        self.cursor += self.batch_size
        self.batch_in_epoch += 1
        if self.cursor >= self.nb_samples:
            self.epoch += 1
            self._new_epoch()
            return True
        return False

    def prepare_current_batch(self):
        idx, xb, yb, mb = self._slice()
        self.batch_indices = idx
        self.optimizer.zero_grad()
        bg = AttentionBatchGraph(self.model, xb, yb, mb)  # real forward + stepped backward
        self.views = _make_views(bg)
        self.batch_loss = float(bg.loss.data)
        # sample-0 context for the "see the max" markers
        self.attn_info = {
            "seq_vals": bg.seq_vals[0],  # (seq,) raw values
            "mask": self.mask_train[idx][0],  # (seq,)
            "max_pos": int(bg.max_pos[0]),  # true argmax key position
            "true_max": float(yb[0, 0]),
            "pred_max": float(bg.pred[0, 0]),
        }
        self._pending = True

    def advance_batch(self) -> bool:
        if self._pending:
            self.optimizer.step()
            self._pending = False
        new_epoch = self._advance_cursor()
        self.prepare_current_batch()
        return new_epoch

    def fast_train_batch(self) -> bool:
        if self._pending:
            self.optimizer.step()
            self._pending = False
            return self._advance_cursor()
        _idx, xb, yb, mb = self._slice()
        self.optimizer.zero_grad()
        mse(_attention_pred(self.model, xb, mb), yb).backward()
        self.optimizer.step()
        return self._advance_cursor()

    def state(self) -> TrainState:
        return TrainState(self.epoch, self.batch_in_epoch, self.nb_batches, self.batch_loss)

    def restart(self):
        self._setup()


def make_trainer(kind, *, seed, lr, n_samples, dataset, arch, batch_size, dropout):
    """Pick which model's graph the visualiser builds. Everything downstream is shared."""
    if kind == "mlp":
        return Trainer(dataset, arch, batch_size, seed, dropout, lr, n_samples)
    if kind == "attention":
        return AttentionTrainer(batch_size=batch_size, seed=seed, lr=lr, n_samples=n_samples)
    raise ValueError(f"Unknown MODEL_KIND: {kind!r} (expected 'mlp' or 'attention')")
