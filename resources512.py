"""Exact gate census of the implemented QC-RDH circuit for the 512x512 'camera' cover
(1024x1024 stego, n = 10), from the bits the circuit would write, combined with the
measured elementary cost of one multi-controlled NOT (mcx_cost.py)."""
import json, math, numpy as np
from core import *
R = json.load(open("results/results.json")); K_QKD = bytes.fromhex(R["bb84"]["key_hex"])
IM = np.load("results/images.npz"); cam = IM["camera"].astype(np.int64)
cap = capacity(512, 512)
alph = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:-/()%+", np.uint8)
rng = np.random.default_rng(7); n_chars = 340000
while True:
    max_text = bytes(rng.choice(alph, n_chars)).decode()
    if len(encode_payload(max_text.encode())) <= cap: break
    n_chars -= 1000
C, nonce = embed(cam, max_text, K_QKD)
n = 10; side = 1024
S = session(master_key(K_QKD), nonce); ks = keystream(S, side)
U = C ^ ks; Y, X = np.indices((side, side)); Y2, X2 = scramble(Y, X, round_constants(S, n), n)
D = U[Y2, X2] & 3
cnt = dict(prep=int(np.unpackbits(cam.astype(np.uint8)).sum()),           # 18 controls
           writeD=int((D & 1).sum() + ((D >> 1) & 1).sum()),              # 20 controls
           key=int(sum(((ks >> k) & 1).sum() for k in range(10))))        # 20 controls
cnt = {k: int(v) for k, v in cnt.items()}
assert cnt["prep"] == R["resources"]["ones_cover"] and cnt["key"] == R["resources"]["ones_key"]
P = json.load(open("results/mcx_cost.json"))["per_gate"]
qft = n * (n - 1) // 2 + n; shear = 2 * qft + n * n + n; R2_swaps = 18
out = dict(counts=cnt, mcx_total=sum(cnt.values()), shear_ops=8 * shear, other_ops=18 + 2 + R2_swaps)          # 18 H (prep), 2 H (scaling), 18 SWAP
for mode in ("one_ancilla", "no_ancilla"):
    c18, c20 = P["18"][mode], P["20"][mode]
    tot = lambda f: cnt["prep"] * c18[f] + (cnt["writeD"] + cnt["key"]) * c20[f]
    out[mode] = dict(cx=tot("cx"), elementary=tot("total"), depth_serial=tot("depth"))
out["logical_ops"] = out["mcx_total"] + out["shear_ops"] + out["other_ops"]
json.dump(out, open("results/resources512.json", "w"), indent=1); print(json.dumps(out, indent=1))
