"""Exact check of the elementary-gate (CX + u) decompositions of the multi-controlled NOTs used at full
size (18 and 20 controls), with one ancilla and without: the transpiled circuit is applied to a statevector
whose 2^q amplitudes carry distinct phases, and the output must be exactly the MCX permutation of the input
(identity on the ancilla, whatever its state), up to numerical error. Open controls are checked too."""
import json, numpy as np, time
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
SIM = AerSimulator(method="statevector")
def check(m, anc, ctrl_state):
    q = m + 1 + anc; qc = QuantumCircuit(q)
    if anc: qc.mcx(list(range(m)), m, ancilla_qubits=[m + 1], mode="recursion", ctrl_state=ctrl_state)
    else: qc.mcx(list(range(m)), m, ctrl_state=ctrl_state)
    t = transpile(qc, basis_gates=["cx", "u"], optimization_level=1, seed_transpiler=1)
    N = 1 << q; idx = np.arange(N); ph = np.exp(2j * np.pi * idx / N) / np.sqrt(N)
    c = QuantumCircuit(q); c.initialize(ph); c.compose(t, inplace=True); c.save_statevector()
    out = np.asarray(SIM.run(transpile(c, SIM)).result().get_statevector())
    mask = (1 << m) - 1; hit = (idx & mask) == ctrl_state
    perm = np.where(hit, idx ^ (1 << m), idx)              # expected: flip target iff controls match
    exp = np.empty(N, complex); exp[perm] = ph
    gphase = np.vdot(exp, out)                              # global phase allowed
    err = float(np.abs(out - exp * gphase / abs(gphase)).max() * np.sqrt(N))
    o = t.count_ops()
    return dict(controls=m, ancilla=anc, ctrl_state=ctrl_state, qubits=q, cx=int(o.get("cx", 0)), gates=t.size(),
                max_amp_error=err, exact=err < 1e-6)
rng = np.random.default_rng(5); out = []
for m in (18, 20):
    for anc in (1, 0):
        for cs in ((1 << m) - 1, int(rng.integers(0, 1 << m))):
            t0 = time.time(); r = check(m, anc, cs); r["seconds"] = time.time() - t0; out.append(r); print(r, flush=True)
json.dump(out, open("results/mcx_verify.json", "w"), indent=1)
