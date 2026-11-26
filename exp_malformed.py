"""Malformed-input handling: Extract must return a rejection, never a partial cover or text, for
random stego images, truncated/oversized images, corrupted RR length headers and wrong-size inputs."""
import json, numpy as np
from core import *
from exp_rr_fill import fill
R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"])
I = np.load("results/images.npz")["camera"].astype(np.int64)[::4, ::4]          # 128x128 for speed
rng = np.random.default_rng(1); out = {}
cfgs = [("(2,2)", embed, extract, {}, 10, 2), ("(1,8)", embed, extract, dict(kb=8, scale=1), 16, 1), ("RR", embed_rr, extract_rr, {}, 8, 1)]
for name, emb, ext, kw, w, sc in cfgs:
    txt = fill(rr_capacity(I) // 2) if name == "RR" else "malformed-input test"
    C, nu = emb(I, txt, K, **kw); side = C.shape[0]; acc = 0; tot = 0; partial = 0
    cases = [rng.integers(0, 1 << w, C.shape) for _ in range(200)]                  # random images
    for t in range(200):                                                           # bit flips anywhere
        T = C.copy(); y, x = rng.integers(0, side, 2); T[y, x] ^= 1 << int(rng.integers(0, w)); cases.append(T)
    for T in cases:
        try: cov, tx, rep = ext(T, nu, K, **kw)
        except Exception as e: rep = dict(authentic=False, crashed=str(e)[:40]); cov, tx = None, None
        tot += 1; acc += bool(rep["authentic"]); partial += (not rep["authentic"]) and (tx is not None and tx != "")
    nonce_bad = not ext(C, bytes([nu[0] ^ 1]) + nu[1:], K, **kw)[2]["authentic"]
    out[name] = dict(cases=tot, accepted=acc, wrong_nonce_rejected=nonce_bad); print(name, out[name], flush=True)
json.dump(out, open("results/malformed.json", "w"), indent=1)
