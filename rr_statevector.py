"""Gate-level statevector check of the RR circuit at 18 qubits (32x32 cover, w = 8), both as logical circuit
and transpiled to {CX, u}, against the emulation embed_rr."""
import json, math, numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from core import *
from qcirc import draper_affine
from circsim_rr import container
R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"])
y0, x0 = np.indices((32, 32)); I = np.rint(np.load("results/images.npz")["camera"].reshape(32, 16, 32, 16).mean((1, 3))).astype(np.int64)
text = "RR"; M = 32; n = 5; w = 8
C_ref, _ = embed_rr(I, text, K); V, S, _ = container(I, text, K)
y, x, c = QuantumRegister(n, "y"), QuantumRegister(n, "x"), QuantumRegister(w, "c"); qc = QuantumCircuit(y, x, c)
ctrl = list(y) + list(x); qc.h(ctrl)
for Y in range(M):
    for X in range(M):
        for j in range(w):
            if (int(V[Y, X]) >> j) & 1: qc.mcx(ctrl, c[j], ctrl_state=Y + (X << n))
for al, be, ga, de in round_constants(S, n): draper_affine(qc, x, y, al, be); draper_affine(qc, y, x, ga, de)
ks = keystream(S, M, w)
for Y in range(M):
    for X in range(M):
        for j in range(w):
            if (int(ks[Y, X]) >> j) & 1: qc.mcx(ctrl, c[j], ctrl_state=Y + (X << n))
SIM = AerSimulator(method="statevector"); out = {}
for name, circ in (("logical", qc), ("elementary", transpile(qc, basis_gates=["cx", "u"], optimization_level=1, seed_transpiler=1))):
    c2 = circ.copy(); c2.save_statevector()
    sv = np.asarray(SIM.run(transpile(c2, SIM, optimization_level=0)).result().get_statevector())
    idx = np.flatnonzero(np.abs(sv) > 1e-6); img = -np.ones((M, M), np.int64)
    img[idx & (M - 1), (idx >> n) & (M - 1)] = idx >> (2 * n)
    out[name] = dict(gates=circ.size(), cx=int(circ.count_ops().get("cx", 0)), branches=int(idx.size),
                     amp_spread=float(np.abs(np.abs(sv[idx]) - 1 / M).max()), equals_emulation=bool(np.array_equal(img, C_ref)))
    print(name, out[name], flush=True)
json.dump(out, open("results/rr_statevector.json", "w"), indent=1)
