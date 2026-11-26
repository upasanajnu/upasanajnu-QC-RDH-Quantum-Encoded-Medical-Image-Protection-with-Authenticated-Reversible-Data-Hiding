"""Measures the elementary cost of the multi-controlled NOTs used by QC-RDH and
derives the 512x512 resource counts from the implemented circuit's gate census."""
import json, numpy as np
from collections import Counter
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from skimage import data, transform
from core import *; from qcirc import *

def cost(m, anc):
    qc = QuantumCircuit(m + 1 + anc)
    qc.append(MCXGate(m), list(range(m + 1)))
    if anc:   # let the synthesis use the dirty/clean ancilla
        qc = QuantumCircuit(m + 2); qc.mcx(list(range(m)), m, ancilla_qubits=[m + 1], mode="recursion")
    t = transpile(qc, basis_gates=["cx", "u"], optimization_level=1, seed_transpiler=1)
    o = t.count_ops(); return dict(cx=int(o.get("cx", 0)), total=t.size(), depth=t.depth())

R = json.load(open("results/results.json")); K_QKD = bytes.fromhex(R["bb84"]["key_hex"])
small = transform.resize(data.camera(), (8, 8), preserve_range=True, anti_aliasing=True).round().astype(np.int64)
emb, enc, _ = sender_circuits(small, "HbA1c 8.1%", K_QKD, DEFAULT_CHAOS)
census = Counter()
for ins in emb.compose(enc).data:
    if ins.operation.name.startswith("mcx") or ins.operation.name == "mcx":
        census[ins.operation.num_ctrl_qubits] += 1
out = dict(census_8x8={str(k): v for k, v in sorted(census.items())}, per_gate={})
for m in (6, 7, 8, 18, 19, 20):
    out["per_gate"][m] = dict(no_ancilla=cost(m, 0), one_ancilla=cost(m, 1))
json.dump(out, open("results/mcx_cost.json", "w"), indent=1); print(json.dumps(out, indent=1))
