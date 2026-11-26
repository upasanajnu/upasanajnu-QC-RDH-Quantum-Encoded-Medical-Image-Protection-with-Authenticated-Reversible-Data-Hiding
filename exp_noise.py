"""Noise study: the complete sender circuit for a 4x4 cover (8x8 stego image, 16 qubits), transpiled to
{CX, u}, run under depolarising noise (two-qubit error p, single-qubit error p/10) with 300 shots per error rate.
A shot is correct if the measured (position, value) pair equals the ideal stego image at that position.
The ideal circuit gives 100 %; random outcomes give 2^-10 = 0.1 %."""
import json, numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from core import *
from qcirc import sender_circuits
R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"])
I = np.array([[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 8], [9, 7, 9, 3]]); txt = "Q"
emb, enc, nu = sender_circuits(I, txt, K, DEFAULT_CHAOS); C, _ = embed(I, txt, K)
qc = emb.compose(enc); qc.measure_all()
t = transpile(qc, basis_gates=["cx", "u"], optimization_level=1, seed_transpiler=1)
ops = t.count_ops(); ncx, nu1 = int(ops.get("cx", 0)), int(ops.get("u", 0)); print("gates", t.size(), "cx", ncx, "u", nu1, "depth", t.depth())
SH = int(__import__("os").environ.get("SH", 300))
n = 3; out = dict(cx=ncx, u=nu1, depth=t.depth(), runs=[])
for p in [float(x) for x in __import__("os").environ.get("PS", "0,1e-5,1e-4,1e-3").split(",")]:
    nm = NoiseModel()
    if p:
        nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"]); nm.add_all_qubit_quantum_error(depolarizing_error(p / 10, 1), ["u"])
    sim = AerSimulator(noise_model=nm if p else None, seed_simulator=3, max_parallel_threads=0)
    counts = sim.run(t, shots=SH).result().get_counts()
    ok = 0
    for k, v in counts.items():
        idx = int(k.replace(" ", ""), 2); Y, X, val = idx & 7, (idx >> 3) & 7, idx >> 6
        ok += v * (C[Y, X] == val)
    est = (1 - p) ** ncx * (1 - p / 10) ** nu1
    out["runs"].append(dict(p=p, correct=ok / SH, no_fault_probability=est)); print(p, ok / SH, round(est, 4), flush=True)
json.dump(out, open(__import__("os").environ.get("OUT", "results/noise.json"), "w"), indent=1)
