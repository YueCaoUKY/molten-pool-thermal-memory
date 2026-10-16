"""Torch-free reading of torch.save archives and a numpy port of the inference-time layers used by
sec6_experiments.part_3b (AlignedNODE, StructureDecoder, ResidualVQ.decode_indices).

PyTorch cannot be installed in this container, so the forward pass is re-implemented in float32
numpy and checked against arrays the original torch code already wrote (gen_WAAM_video_3.npz).
"""
from __future__ import annotations

import collections
import io
import pickle
import zipfile

import numpy as np

_DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
       "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
       "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_}


class _StorageType:
    def __init__(self, name):
        self.dtype = _DT[name]


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, meta=None):
    item = storage.itemsize
    if len(size) == 0:
        return storage[offset:offset + 1].reshape(()).copy()
    v = np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                        strides=tuple(s * item for s in stride))
    return np.array(v, copy=True)


def _rebuild_parameter(data, requires_grad=False, hooks=None):
    return data


def load_pt(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    pkl = [n for n in names if n.endswith("/data.pkl")][0]
    root = pkl[:-len("data.pkl")]

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if mod == "torch._utils" and name == "_rebuild_parameter":
                return _rebuild_parameter
            if mod == "torch" and name in _DT:
                return _StorageType(name)
            if mod == "torch" and name == "Size":
                return tuple
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, stype, key, _loc, numel = pid
            buf = zf.read(f"{root}data/{key}")
            return np.frombuffer(buf, dtype=stype.dtype, count=numel)

    return U(io.BytesIO(zf.read(pkl))).load()


# --------------------------------------------------------------------------- elementwise
def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20)))).astype(np.float32)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def silu(x):
    return (x * sigmoid(x)).astype(np.float32)


# --------------------------------------------------------------------------- AlignedNODE
class TanhMLP:
    """Section 5 experiment.MLP: Linear-Tanh-Linear-Tanh-Linear."""

    def __init__(self, sd, prefix):
        self.w = [sd[f"{prefix}.net.{i}.weight"].astype(np.float32) for i in (0, 2, 4)]
        self.b = [sd[f"{prefix}.net.{i}.bias"].astype(np.float32) for i in (0, 2, 4)]

    def __call__(self, x):
        h = np.tanh(x @ self.w[0].T + self.b[0])
        h = np.tanh(h @ self.w[1].T + self.b[1])
        return (h @ self.w[2].T + self.b[2]).astype(np.float32)


class AlignedNODE:
    """node_aligned.AlignedNODE with use_delta=True, delta_dim=1, lam_bands=40, gamma_min=1/4."""

    def __init__(self, sd, tau0=4.0, tau_m=40.0):
        self.gamma_min = np.float32(1.0 / tau0)
        self.z_eq_net = TanhMLP(sd, "z_eq_net")
        self.gamma_net = TanhMLP(sd, "gamma_net")
        self.phi_net = TanhMLP(sd, "phi_net")
        lo, hi = 1.0 / (1.0 / tau0), float(tau_m)
        tau = lo * (hi / lo) ** sigmoid(sd["raw_lam"].astype(np.float32))
        self.lam = (1.0 / tau).astype(np.float32)

    def terms(self, z, delta, u):
        dfeat = delta / (1.0 + delta)
        z_eq = self.z_eq_net(np.concatenate([u, dfeat]))
        gamma = self.gamma_min + softplus(self.gamma_net(np.concatenate([z, u, dfeat])))
        target = softplus(self.phi_net(np.concatenate([u, z])))
        return z_eq, gamma, self.lam * (target - delta)

    def rollout(self, z0, u, dt):
        z = z0.astype(np.float32)
        delta = np.zeros(1, np.float32)
        zs, ds = [], []
        for k in range(u.shape[0]):
            zs.append(z); ds.append(delta)
            if k == u.shape[0] - 1:
                break
            z_eq, gamma, dd = self.terms(z, delta, u[k])
            z = (z_eq + (z - z_eq) * np.exp(-gamma * np.float32(dt))).astype(np.float32)
            delta = (delta + np.float32(dt) * dd).astype(np.float32)
        return np.stack(zs), np.stack(ds)

    def rollout_forced(self, z0, u, delta, dt):
        z, out = z0.astype(np.float32), []
        for k in range(u.shape[0]):
            out.append(z)
            if k == u.shape[0] - 1:
                break
            z_eq, gamma, _ = self.terms(z, delta[k], u[k])
            z = (z_eq + (z - z_eq) * np.exp(-gamma * np.float32(dt))).astype(np.float32)
        return np.stack(out)


# --------------------------------------------------------------------------- conv layers
def conv2d(x, w, b, pad):
    """x (B,C,H,W), w (O,C,k,k), stride 1."""
    B, C, H, W = x.shape
    O, _, kh, kw = w.shape
    if kh == 1:
        out = np.tensordot(w[:, :, 0, 0], x, axes=([1], [1]))           # (O,B,H,W)
    else:
        xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        out = np.zeros((O, B, H, W), np.float32)
        for i in range(kh):
            for j in range(kw):
                out += np.tensordot(w[:, :, i, j], xp[:, :, i:i + H, j:j + W], axes=([1], [1]))
    out = out.transpose(1, 0, 2, 3)
    return (out + b[None, :, None, None]).astype(np.float32)


def group_norm(x, g, weight, bias, eps=1e-5):
    B, C, H, W = x.shape
    xr = x.reshape(B, g, -1).astype(np.float64)
    m = xr.mean(-1, keepdims=True)
    v = xr.var(-1, keepdims=True)
    xr = ((xr - m) / np.sqrt(v + eps)).reshape(B, C, H, W).astype(np.float32)
    return xr * weight[None, :, None, None] + bias[None, :, None, None]


def upsample2(x):
    return x.repeat(2, axis=2).repeat(2, axis=3)


def _p(sd, k):
    return sd[k].astype(np.float32)


def up_block(sd, prefix, x):
    """nn.Sequential(Upsample, Conv2d 3x3, GroupNorm(8), SiLU) at index prefix."""
    x = upsample2(x)
    x = conv2d(x, _p(sd, f"{prefix}.1.weight"), _p(sd, f"{prefix}.1.bias"), 1)
    x = group_norm(x, 8, _p(sd, f"{prefix}.2.weight"), _p(sd, f"{prefix}.2.bias"))
    return silu(x)


class StructureDecoder:
    def __init__(self, sd, mean_frame, size=192, base=256):
        self.sd, self.mean_frame = sd, mean_frame.astype(np.float32)
        self.start, self.base = size // 32, base

    def __call__(self, z):
        sd = self.sd
        h = z.astype(np.float32) @ _p(sd, "fc.weight").T + _p(sd, "fc.bias")
        h = h.reshape(-1, self.base, self.start, self.start)
        for i in range(5):
            h = up_block(sd, f"net.{i}", h)
        h = conv2d(h, _p(sd, "net.5.weight"), _p(sd, "net.5.bias"), 1)
        return self.mean_frame + h                                       # (B,1,S,S)


def res_block(sd, prefix, x):
    h = group_norm(x, 8, _p(sd, f"{prefix}.net.0.weight"), _p(sd, f"{prefix}.net.0.bias"))
    h = silu(h)
    h = conv2d(h, _p(sd, f"{prefix}.net.2.weight"), _p(sd, f"{prefix}.net.2.bias"), 1)
    h = group_norm(h, 8, _p(sd, f"{prefix}.net.3.weight"), _p(sd, f"{prefix}.net.3.bias"))
    h = silu(h)
    h = conv2d(h, _p(sd, f"{prefix}.net.5.weight"), _p(sd, f"{prefix}.net.5.bias"), 0)
    return x + h


class ResidualVQDecoder:
    def __init__(self, sd):
        self.sd = sd
        self.emb = _p(sd, "vq.embedding")

    def decode_indices(self, idx):
        sd = self.sd
        q = self.emb[idx].transpose(0, 3, 1, 2)                         # (B,D,H,W)
        h = conv2d(q, _p(sd, "dec.0.weight"), _p(sd, "dec.0.bias"), 1)
        h = res_block(sd, "dec.1", h)
        h = res_block(sd, "dec.2", h)
        h = up_block(sd, "dec.3", h)
        h = up_block(sd, "dec.4", h)
        h = up_block(sd, "dec.5", h)
        return conv2d(h, _p(sd, "dec.6.weight"), _p(sd, "dec.6.bias"), 1)
