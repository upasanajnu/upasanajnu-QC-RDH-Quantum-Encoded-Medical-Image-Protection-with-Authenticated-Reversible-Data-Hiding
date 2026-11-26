"""Exact logical-gate execution of the parameterised circuits QC-RDH(s, k) at full size, with the branch
simulator of circsim.py. s = 1 omits the scaling Hadamards; the intensity register has 8 + k qubits, is
rotated by k places (|c> -> |2^k c>), and W_D writes k bits. Output is compared with core.embed/extract."""
import math, json, time, sys, numpy as np
from qiskit import QuantumRegister
from core import *
from qcirc import draper_affine
from circsim import BranchSim

def build_and_run(cover, text, K, kb, sc, qb=8):
    M = cover.shape[0]; side = sc * M; n = int(math.log2(side)); N = 1 << n; nc = qb + kb
    C_ref, nonce = embed(cover, text, K, kb=kb, scale=sc)
    Km = master_key(K); S = session(Km, nonce); bits = encode_payload(text.encode()); cap = kb * side * side
    pad = np.unpackbits(np.frombuffer(xof(S, b"pad", (cap - len(bits) + 7) // 8), np.uint8))[: cap - len(bits)]
    b = np.concatenate([np.frombuffer(bits.encode(), np.uint8) - 48, pad]).astype(np.int64)
    D = (b.reshape(-1, kb) << np.arange(kb - 1, -1, -1)).sum(1).reshape(side, side)
    consts = round_constants(S, n); ks = keystream(S, side, nc)
    y, x, c = QuantumRegister(n, "y"), QuantumRegister(n, "x"), QuantumRegister(nc, "c")
    sim = BranchSim([y, x, c]); sim.npos = 2 * n; t0 = time.time()
    up = (list(y[1:]) + list(x[1:])) if sc == 2 else (list(y) + list(x)); m = n - (1 if sc == 2 else 0)
    sim.h(up)                                                            # U_prep
    for Y in range(M):
        for X in range(M):
            v = int(cover[Y, X]); st = Y + (X << m)
            for j in range(qb):
                if (v >> j) & 1: sim.mcx(up, c[j], ctrl_state=st)
    if sc == 2: sim.h(y[0]); sim.h(x[0])                                 # U_sc
    for _ in range(kb):                                                  # R_k: rotate intensity register by k places
        for j in range(nc - 1, 0, -1): sim.swap(c[j], c[j - 1])
    ctrl = list(y) + list(x)
    for Y in range(side):                                                # W_D
        for X in range(side):
            d = int(D[Y, X]); st = Y + (X << n)
            for j in range(kb):
                if (d >> j) & 1: sim.mcx(ctrl, c[j], ctrl_state=st)
    for al, be, ga, de in consts: draper_affine(sim, x, y, al, be); draper_affine(sim, y, x, ga, de)
    for Y in range(side):                                                # U_key
        for X in range(side):
            k = int(ks[Y, X]); st = Y + (X << n)
            for j in range(nc):
                if (k >> j) & 1: sim.mcx(ctrl, c[j], ctrl_state=st)
    C_q, nbr = sim.image(n); cnt = dict(sim.counts)
    return dict(qubits=2 * n + nc, branches=int(nbr), stego_equals_emulation=bool(np.array_equal(C_q, C_ref)),
                mcx=cnt["mcx"], swaps=cnt["swap"], hadamards=cnt["h"], max_phase_err=sim.max_phase_err, seconds=time.time() - t0)

if __name__ == "__main__":
    R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"])
    A = np.load("results/images.npz"); B = np.load("results/medical_imgs.npz")
    imgs = [(k, A[k].astype(np.int64)) for k in A] + [(k, B[k].astype(np.int64)) for k in B if not k.endswith(("_s", "_r"))]
    alph = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:-/()%+", np.uint8)
    kb, sc = int(sys.argv[1]), int(sys.argv[2]); out = {}
    for name, I in imgs:
        cap = kb * (sc * I.shape[0]) ** 2; rng = np.random.default_rng(7); nch = int(cap / 6.25)
        while True:
            t = bytes(rng.choice(alph, nch)).decode()
            if len(encode_payload(t.encode())) <= cap: break
            nch -= max(100, nch // 300)
        out[name] = build_and_run(I, t, K, kb, sc); print(name, out[name], flush=True)
    json.dump(out, open(f"results/circsim_s{sc}_k{kb}.json", "w"), indent=1)
