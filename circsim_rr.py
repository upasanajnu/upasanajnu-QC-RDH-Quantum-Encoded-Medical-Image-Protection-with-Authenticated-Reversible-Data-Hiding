"""Exact logical-gate execution of the room-reserving (RR) circuits at full size with the branch simulator:
U_prep writes the container V (header, JPEG-LS cover, payload, pad) into an NEQR state of 2n + w qubits,
followed by the Draper-adder scrambling and the key oracle. No scaling, rotation or W_D is needed."""
import math, json, time, sys, numpy as np
from qiskit import QuantumRegister
from core import *
from core import _jls_encode, _bits_to_words, RR_HDR
from qcirc import draper_affine
from circsim import BranchSim

def container(cover, text, K, qb=8, extra=0):
    M = cover.shape[0]; w = qb + extra; data = text.encode()
    Km = master_key(K); nonce = siv_nonce(Km, cover, data); S = session(Km, nonce)
    z = _jls_encode(cover, qb); zb = np.unpackbits(np.frombuffer(z, np.uint8)).astype(np.int64)
    hdr = np.array([int(c) for c in format(len(z), "032b")], np.int64)
    pb = np.frombuffer(encode_payload(data).encode(), np.uint8).astype(np.int64) - 48
    tot = M * M * w; used = RR_HDR + zb.size + pb.size
    pad = np.unpackbits(np.frombuffer(xof(S, b"pad", (tot - used + 7) // 8), np.uint8))[: tot - used].astype(np.int64)
    return _bits_to_words(np.concatenate([hdr, zb, pb, pad]), w).reshape(M, M), S, nonce

def run(cover, text, K, qb=8, extra=0):
    M = cover.shape[0]; n = int(math.log2(M)); w = qb + extra
    C_ref, nonce_ref = embed_rr(cover, text, K, qb=qb, extra=extra)
    V, S, nonce = container(cover, text, K, qb, extra); assert nonce == nonce_ref
    consts = round_constants(S, n); ks = keystream(S, M, w)
    y, x, c = QuantumRegister(n, "y"), QuantumRegister(n, "x"), QuantumRegister(w, "c")
    sim = BranchSim([y, x, c]); sim.npos = 2 * n; t0 = time.time(); ctrl = list(y) + list(x)
    sim.h(ctrl)                                                          # U_prep: uniform superposition ...
    for Y in range(M):                                                   # ... and one MCX per set container bit
        for X in range(M):
            v = int(V[Y, X]); st = Y + (X << n)
            for j in range(w):
                if (v >> j) & 1: sim.mcx(ctrl, c[j], ctrl_state=st)
    prep = sim.counts["mcx"]
    for al, be, ga, de in consts: draper_affine(sim, x, y, al, be); draper_affine(sim, y, x, ga, de)
    for Y in range(M):                                                   # U_key
        for X in range(M):
            k = int(ks[Y, X]); st = Y + (X << n)
            for j in range(w):
                if (k >> j) & 1: sim.mcx(ctrl, c[j], ctrl_state=st)
    C_q, nb = sim.image(n); cnt = dict(sim.counts)
    return dict(qubits=2 * n + w, branches=int(nb), stego_equals_emulation=bool(np.array_equal(C_q, C_ref)),
                mcx=cnt["mcx"], mcx_prep=prep, mcx_key=cnt["mcx"] - prep, hadamards=cnt["h"], shear_ops=cnt["block_gates"],
                max_phase_err=sim.max_phase_err, seconds=time.time() - t0)

if __name__ == "__main__":
    from exp_rr_fill import fill
    R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"])
    A = np.load("results/images.npz"); B = np.load("results/medical_imgs.npz")
    imgs = [(k, A[k].astype(np.int64)) for k in A] + [(k, B[k].astype(np.int64)) for k in B if not k.endswith(("_s", "_r"))]
    out = {}
    for name, I in imgs:
        t = fill(rr_capacity(I)); out[name] = run(I, t, K); print(name, out[name], flush=True)
    json.dump(out, open("results/circsim_rr.json", "w"), indent=1)
