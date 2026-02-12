"""Rex benchmark — MLX symbol search quality and timing.

Extensively queries every public symbol in the mlx package to
measure Rex's coverage. MLX is a namespace package (no __init__.py
at top level) with compiled extensions (.pyi stubs), making it a
challenging and representative test case.

Organized by discovery flow: how a developer gradually learns MLX.

Usage:
    uv run python bench/bench_mlx.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rex.storage import SearchResult, get_symbol, search, get_members


@dataclass
class Query:
    op: str  # "find", "show", "members"
    query: str
    expect: list[str]  # substrings to match in results
    description: str = ""


@dataclass
class QueryResult:
    op: str
    query: str
    expect: list[str]
    found: list[str]
    hits: list[str]
    misses: list[str]
    elapsed_ms: float
    description: str = ""

    @property
    def score(self) -> float:
        return len(self.hits) / len(self.expect) if self.expect else 1.0


@dataclass
class SuiteResult:
    name: str
    queries: list[QueryResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        return (
            sum(q.score for q in self.queries) / len(self.queries)
            if self.queries
            else 0.0
        )

    @property
    def total_ms(self) -> float:
        return sum(q.elapsed_ms for q in self.queries)


# ===================================================================
# Suite 1: nn.layers — every layer class
# ===================================================================
NN_LAYERS = [
    # Linear & basic
    Query("find", "Linear", ["Linear"],
          "fully connected layer"),
    Query("find", "Bilinear", ["Bilinear"],
          "bilinear transformation"),
    Query("find", "Identity", ["Identity"],
          "no-op identity layer"),
    # Convolutions
    Query("find", "Conv1d", ["Conv1d"],
          "1D convolution"),
    Query("find", "Conv2d", ["Conv2d"],
          "2D convolution"),
    Query("find", "Conv3d", ["Conv3d"],
          "3D convolution"),
    Query("find", "ConvTranspose1d", ["ConvTranspose1d"],
          "1D transposed convolution"),
    Query("find", "ConvTranspose2d", ["ConvTranspose2d"],
          "2D transposed convolution"),
    Query("find", "ConvTranspose3d", ["ConvTranspose3d"],
          "3D transposed convolution"),
    # Recurrent
    Query("find", "RNN", ["RNN"],
          "Elman recurrent unit"),
    Query("find", "LSTM", ["LSTM"],
          "long short-term memory"),
    Query("find", "GRU", ["GRU"],
          "gated recurrent unit"),
    # Normalization
    Query("find", "LayerNorm", ["LayerNorm"],
          "layer normalization"),
    Query("find", "BatchNorm", ["BatchNorm"],
          "batch normalization"),
    Query("find", "GroupNorm", ["GroupNorm"],
          "group normalization"),
    Query("find", "InstanceNorm", ["InstanceNorm"],
          "instance normalization"),
    Query("find", "RMSNorm", ["RMSNorm"],
          "RMS normalization"),
    # Dropout
    Query("find", "Dropout", ["Dropout"],
          "standard dropout"),
    Query("find", "Dropout2d", ["Dropout2d"],
          "channel-wise dropout 2D"),
    Query("find", "Dropout3d", ["Dropout3d"],
          "channel-wise dropout 3D"),
    # Embedding
    Query("find", "Embedding", ["Embedding"],
          "token embedding"),
    Query("find", "QuantizedEmbedding", ["QuantizedEmbedding"],
          "quantized embedding"),
    # Transformer & attention
    Query("find", "MultiHeadAttention", ["MultiHeadAttention"],
          "scaled dot-product attention"),
    Query("find", "TransformerEncoder", ["TransformerEncoder"],
          "encoder stack"),
    Query("find", "TransformerDecoder", ["TransformerDecoder"],
          "decoder stack"),
    Query("find", "Transformer", ["Transformer"],
          "complete transformer"),
    # Positional encoding
    Query("find", "RoPE", ["RoPE"],
          "rotary position embedding"),
    Query("find", "SinusoidalPositionalEncoding",
          ["SinusoidalPositionalEncoding"],
          "sinusoidal positional encoding"),
    Query("find", "ALiBi", ["ALiBi"],
          "attention linear biases"),
    # Pooling
    Query("find", "MaxPool1d", ["MaxPool1d"],
          "1D max pooling"),
    Query("find", "MaxPool2d", ["MaxPool2d"],
          "2D max pooling"),
    Query("find", "AvgPool1d", ["AvgPool1d"],
          "1D average pooling"),
    Query("find", "AvgPool2d", ["AvgPool2d"],
          "2D average pooling"),
    # Other
    Query("find", "Upsample", ["Upsample"],
          "spatial upsampling"),
    Query("find", "Sequential", ["Sequential"],
          "compose layers"),
    Query("find", "QuantizedLinear", ["QuantizedLinear"],
          "quantized linear layer"),
    Query("find", "Module", ["Module"],
          "base class for models"),
]

# ===================================================================
# Suite 2: nn.activations — every activation function
# ===================================================================
NN_ACTIVATIONS = [
    # ReLU family
    Query("find", "ReLU", ["ReLU"], "basic ReLU"),
    Query("find", "LeakyReLU", ["LeakyReLU"], "leaky ReLU"),
    Query("find", "PReLU", ["PReLU"], "parametric ReLU"),
    Query("find", "ELU", ["ELU"], "exponential LU"),
    Query("find", "CELU", ["CELU"], "continuously diff ELU"),
    Query("find", "SELU", ["SELU"], "scaled ELU"),
    # Gating / modern
    Query("find", "GELU", ["GELU"], "Gaussian error LU"),
    Query("find", "SiLU", ["SiLU"], "sigmoid LU / Swish"),
    Query("find", "Mish", ["Mish"], "Mish activation"),
    Query("find", "GLU", ["GLU"], "gated linear unit"),
    # Sigmoid / tanh
    Query("find", "Sigmoid", ["Sigmoid"], "sigmoid"),
    Query("find", "Tanh", ["Tanh"], "hyperbolic tangent"),
    Query("find", "LogSigmoid", ["LogSigmoid"], "log sigmoid"),
    # Softmax
    Query("find", "Softmax", ["Softmax"], "softmax"),
    Query("find", "LogSoftmax", ["LogSoftmax"], "log softmax"),
    Query("find", "Softmin", ["Softmin"], "softmin"),
    # Hard variants
    Query("find", "HardTanh", ["HardTanh"], "hard tanh"),
    Query("find", "Hardswish", ["Hardswish"], "hard swish"),
    Query("find", "HardShrink", ["HardShrink"], "hard shrink"),
    # Misc
    Query("find", "Softplus", ["Softplus"], "softplus"),
    Query("find", "Softsign", ["Softsign"], "softsign"),
    Query("find", "Softshrink", ["Softshrink"], "softshrink"),
    Query("find", "Step", ["Step"], "binary step"),
]

# ===================================================================
# Suite 3: nn.losses — every loss function
# ===================================================================
NN_LOSSES = [
    Query("find", "cross_entropy", ["cross_entropy"],
          "classification loss"),
    Query("find", "binary_cross_entropy", ["binary_cross_entropy"],
          "binary classification"),
    Query("find", "mse_loss", ["mse_loss"],
          "mean squared error"),
    Query("find", "l1_loss", ["l1_loss"],
          "L1 / MAE loss"),
    Query("find", "nll_loss", ["nll_loss"],
          "negative log likelihood"),
    Query("find", "kl_div_loss", ["kl_div_loss"],
          "KL divergence"),
    Query("find", "smooth_l1_loss", ["smooth_l1_loss"],
          "smooth L1 / Huber-like"),
    Query("find", "huber_loss", ["huber_loss"],
          "Huber loss"),
    Query("find", "triplet_loss", ["triplet_loss"],
          "metric learning"),
    Query("find", "hinge_loss", ["hinge_loss"],
          "SVM-style loss"),
    Query("find", "cosine_similarity_loss",
          ["cosine_similarity_loss"],
          "cosine similarity"),
    Query("find", "log_cosh_loss", ["log_cosh_loss"],
          "log-cosh loss"),
    Query("find", "margin_ranking_loss", ["margin_ranking_loss"],
          "margin ranking"),
    Query("find", "gaussian_nll_loss", ["gaussian_nll_loss"],
          "Gaussian NLL"),
]

# ===================================================================
# Suite 4: nn.init — every initializer
# ===================================================================
NN_INIT = [
    Query("find", "glorot_normal", ["glorot_normal"],
          "Xavier/Glorot normal"),
    Query("find", "glorot_uniform", ["glorot_uniform"],
          "Xavier/Glorot uniform"),
    Query("find", "he_normal", ["he_normal"],
          "Kaiming/He normal"),
    Query("find", "he_uniform", ["he_uniform"],
          "Kaiming/He uniform"),
    Query("find", "orthogonal", ["orthogonal"],
          "orthogonal init"),
]

# ===================================================================
# Suite 5: optimizers — every optimizer and schedule
# ===================================================================
OPTIMIZERS = [
    Query("find", "SGD", ["SGD"], "stochastic gradient descent"),
    Query("find", "Adam", ["Adam"], "Adam optimizer"),
    Query("find", "AdamW", ["AdamW"], "Adam + weight decay"),
    Query("find", "Adamax", ["Adamax"], "Adam infinity norm"),
    Query("find", "Adagrad", ["Adagrad"], "adaptive gradient"),
    Query("find", "AdaDelta", ["AdaDelta"], "adaptive delta"),
    Query("find", "RMSprop", ["RMSprop"], "RMS propagation"),
    Query("find", "Lion", ["Lion"], "sign-based optimizer"),
    Query("find", "Adafactor", ["Adafactor"], "sublinear memory"),
    # Schedules
    Query("find", "cosine_decay", ["cosine_decay"],
          "cosine annealing LR"),
    Query("find", "exponential_decay", ["exponential_decay"],
          "exponential LR decay"),
    Query("find", "step_decay", ["step_decay"],
          "step-wise LR decay"),
    Query("find", "linear_schedule", ["linear_schedule"],
          "linear LR interpolation"),
    # Utilities
    Query("find", "clip_grad_norm", ["clip_grad_norm"],
          "gradient clipping"),
]

# ===================================================================
# Suite 6: core array ops — from compiled .pyi stubs
# ===================================================================
CORE_ARRAY_OPS = [
    # Creation
    Query("find", "zeros", ["zeros"], "create zero array"),
    Query("find", "ones", ["ones"], "create ones array"),
    Query("find", "arange", ["arange"], "range of values"),
    Query("find", "linspace", ["linspace"], "linearly spaced"),
    Query("find", "eye", ["eye"], "identity matrix"),
    # Shape manipulation
    Query("find", "reshape", ["reshape"], "change shape"),
    Query("find", "transpose", ["transpose"], "transpose axes"),
    Query("find", "flatten", ["flatten"], "flatten to 1D"),
    Query("find", "squeeze", ["squeeze"], "remove size-1 dims"),
    Query("find", "expand_dims", ["expand_dims"], "add dimension"),
    Query("find", "concatenate", ["concatenate"], "join arrays"),
    Query("find", "stack", ["stack"], "stack along new axis"),
    Query("find", "split", ["split"], "split array"),
    Query("find", "tile", ["tile"], "repeat array"),
    Query("find", "pad", ["pad"], "pad array"),
    # Math
    Query("find", "matmul", ["matmul"], "matrix multiply"),
    Query("find", "einsum", ["einsum"], "Einstein summation"),
    # Reductions
    Query("find", "logsumexp", ["logsumexp"],
          "numerically stable log-sum-exp"),
    # Selection
    Query("find", "where", ["where"], "conditional select"),
    Query("find", "take", ["take"], "gather elements"),
    Query("find", "sort", ["sort"], "sort array"),
    Query("find", "argmax", ["argmax"], "index of maximum"),
    # Compilation & evaluation
    Query("find", "compile", ["compile"], "JIT compilation"),
]

# ===================================================================
# Suite 7: core.linalg — linear algebra from stubs
# ===================================================================
CORE_LINALG = [
    Query("find", "qr", ["qr"], "QR decomposition"),
    Query("find", "svd", ["svd"], "singular value decomp"),
    Query("find", "eigvals", ["eigvals"], "eigenvalues"),
    Query("find", "cholesky", ["cholesky"], "Cholesky decomp"),
    Query("find", "solve", ["solve"], "solve linear system"),
]

# ===================================================================
# Suite 8: core.fft — Fourier transforms from stubs
# ===================================================================
CORE_FFT = [
    Query("find", "fft", ["fft"], "1D Fourier transform"),
    Query("find", "rfft", ["rfft"], "real FFT"),
    Query("find", "fft2", ["fft2"], "2D FFT"),
    Query("find", "fftshift", ["fftshift"], "shift zero-freq"),
]

# ===================================================================
# Suite 9: core.random — random number generation
# ===================================================================
CORE_RANDOM = [
    Query("find", "categorical", ["categorical"],
          "sample from categorical"),
    Query("find", "bernoulli", ["bernoulli"],
          "Bernoulli samples"),
    Query("find", "truncated_normal", ["truncated_normal"],
          "truncated normal dist"),
    Query("find", "gumbel", ["gumbel"],
          "Gumbel distribution"),
    Query("find", "laplace", ["laplace"],
          "Laplace distribution"),
]

# ===================================================================
# Suite 10: utils — tree operations for parameter handling
# ===================================================================
UTILS = [
    Query("find", "tree_map", ["tree_map"],
          "map over parameter tree"),
    Query("find", "tree_flatten", ["tree_flatten"],
          "flatten nested params"),
    Query("find", "tree_unflatten", ["tree_unflatten"],
          "reconstruct from flat"),
    Query("find", "tree_reduce", ["tree_reduce"],
          "reduce parameter tree"),
]

# ===================================================================
# Suite 11: core save/load — model serialization
# ===================================================================
CORE_IO = [
    Query("find", "save_safetensors", ["save_safetensors"],
          "HuggingFace format"),
    Query("find", "save_gguf", ["save_gguf"],
          "GGML format"),
    Query("find", "savez", ["savez"],
          "numpy npz format"),
]

# ===================================================================
# Suite 12: training workflow — gradient & eval functions
# ===================================================================
TRAINING = [
    Query("find", "value_and_grad", ["value_and_grad"],
          "compute loss + gradients"),
    Query("find", "checkpoint", ["checkpoint"],
          "gradient checkpointing"),
]

# ===================================================================
# Suite 13: name resolution — show specific qualified names
# ===================================================================
NAME_RESOLUTION = [
    Query("show", "mlx.nn.Linear", ["Linear"],
          "nn.Linear by qname"),
    Query("show", "Linear", ["Linear"],
          "bare class name"),
    Query("show", "mlx.nn.Module", ["Module"],
          "model base class"),
    Query("show", "mlx.nn.losses.cross_entropy",
          ["cross_entropy"],
          "loss function by full path"),
    Query("show", "mlx.optimizers.Adam", ["Adam"],
          "optimizer by full path"),
    Query("show", "mlx.nn.RoPE", ["RoPE"],
          "positional encoding"),
    Query("show", "mlx.nn.MultiHeadAttention",
          ["MultiHeadAttention"],
          "attention layer"),
    Query("show", "mlx.nn.layers.activations.gelu",
          ["gelu"],
          "activation function by path"),
]

# ===================================================================
# Suite 14: comparing approaches — side-by-side alternatives
# ===================================================================
COMPARING = [
    # Activation comparison
    Query("show", "mlx.nn.layers.activations.gelu",
          ["gelu", "Gaussian"],
          "GELU details"),
    Query("show", "mlx.nn.layers.activations.silu",
          ["silu", "Sigmoid"],
          "SiLU/Swish details"),
    # Optimizer comparison
    Query("show", "mlx.optimizers.Adam",
          ["Adam", "learning_rate"],
          "Adam signature"),
    Query("show", "mlx.optimizers.Lion",
          ["Lion", "learning_rate"],
          "Lion (newer alternative)"),
    # Normalization comparison
    Query("show", "mlx.nn.LayerNorm",
          ["LayerNorm"],
          "standard normalization"),
    Query("show", "mlx.nn.RMSNorm",
          ["RMSNorm"],
          "RMS norm (faster for LLMs)"),
    # Init comparison
    Query("show", "mlx.nn.init.glorot_normal",
          ["glorot"],
          "Xavier/Glorot init"),
    Query("show", "mlx.nn.init.he_normal",
          ["he", "fan"],
          "He/Kaiming init"),
]

# ===================================================================
# Suite 15: members inspection — explore class internals
# ===================================================================
MEMBERS_INSPECTION = [
    Query("members", "mlx.nn.Module",
          ["parameters", "trainable_parameters", "freeze"],
          "model base class methods"),
    Query("members", "mlx.nn.MultiHeadAttention",
          ["create_additive_causal_mask", "__call__"],
          "attention methods"),
    Query("members", "mlx.optimizers.Adam",
          ["apply_single", "init_single"],
          "optimizer methods"),
    Query("members", "mlx.nn.LSTM",
          ["__call__", "__init__"],
          "LSTM methods"),
]


SUITES = {
    "nn.layers": NN_LAYERS,
    "nn.activations": NN_ACTIVATIONS,
    "nn.losses": NN_LOSSES,
    "nn.init": NN_INIT,
    "optimizers": OPTIMIZERS,
    "core array ops": CORE_ARRAY_OPS,
    "core.linalg": CORE_LINALG,
    "core.fft": CORE_FFT,
    "core.random": CORE_RANDOM,
    "utils": UTILS,
    "core I/O": CORE_IO,
    "training": TRAINING,
    "name resolution": NAME_RESOLUTION,
    "comparing approaches": COMPARING,
    "members inspection": MEMBERS_INSPECTION,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_query(q: Query) -> QueryResult:
    t0 = time.perf_counter()
    found_names: list[str] = []

    if q.op == "find":
        result: SearchResult = search(q.query, limit=15)
        found_names = [s.qualified_name for s in result.symbols]
    elif q.op == "show":
        sym = get_symbol(q.query)
        if sym is not None:
            text = (f"{sym.qualified_name} "
                    f"{sym.signature or ''} "
                    f"{sym.docstring or ''}")
            found_names = [text]
    elif q.op == "members":
        found_names = [m.name for m in get_members(q.query)]

    elapsed = (time.perf_counter() - t0) * 1000

    hits, misses = [], []
    for exp in q.expect:
        if any(exp.lower() in f.lower() for f in found_names):
            hits.append(exp)
        else:
            misses.append(exp)

    return QueryResult(
        op=q.op, query=q.query, expect=q.expect,
        found=found_names[:5], hits=hits, misses=misses,
        elapsed_ms=elapsed, description=q.description,
    )


def run_benchmark() -> dict[str, SuiteResult]:
    results = {}
    for name, queries in SUITES.items():
        sr = SuiteResult(name=name)
        for q in queries:
            sr.queries.append(run_query(q))
        results[name] = sr
    return results


def print_report(results: dict[str, SuiteResult]):
    total_queries = 0
    total_hits = 0
    total_expect = 0
    total_ms = 0.0

    print("=" * 72)
    print("REX BENCHMARK — mlx")
    print("=" * 72)

    for name, sr in results.items():
        print(f"\n{'─' * 72}")
        print(f"{name}  |  score: {sr.score:.0%}  |"
              f"  {sr.total_ms:.0f}ms")
        print(f"{'─' * 72}")

        for qr in sr.queries:
            icon = "✓" if qr.score == 1.0 else (
                "◐" if qr.score > 0 else "✗")
            print(f"  {icon} {qr.op:7} {qr.query:<30} "
                  f"{qr.score:.0%}  {qr.elapsed_ms:6.1f}ms"
                  f"  {qr.description}")
            if qr.misses:
                print(f"    MISS: {qr.misses}")
            if qr.found and qr.score < 1.0:
                print(f"    GOT:  {[f[:60] for f in qr.found[:3]]}")

        total_queries += len(sr.queries)
        total_hits += sum(len(qr.hits) for qr in sr.queries)
        total_expect += sum(len(qr.expect) for qr in sr.queries)
        total_ms += sr.total_ms

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Suites:       {len(results)}")
    print(f"  Queries:      {total_queries}")
    print(f"  Expectations: {total_expect}")
    if total_expect:
        print(f"  Hits:         {total_hits} / {total_expect}"
              f"  ({total_hits / total_expect:.0%})")
    print(f"  Total time:   {total_ms:.0f}ms")
    if total_queries:
        print(f"  Avg per query:{total_ms / total_queries:.1f}ms")

    print(f"\n  {'Suite':<30} {'Score':>6} {'Time':>8}")
    print(f"  {'─' * 30} {'─' * 6} {'─' * 8}")
    for name, sr in results.items():
        print(f"  {name:<30} {sr.score:>5.0%}"
              f" {sr.total_ms:>7.0f}ms")


if __name__ == "__main__":
    results = run_benchmark()
    print_report(results)
