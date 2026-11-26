"""Depth and elementary-gate counts of the 18-qubit gate-level circuit used in Section 8."""
import json, numpy as np
from skimage import data, transform
from qiskit import transpile
from core import *; from qcirc import *
R = json.load(open("results/results.json")); K_QKD = bytes.fromhex(R["bb84"]["key_hex"])
small = transform.resize(data.camera(), (8, 8), preserve_range=True, anti_aliasing=True).round().astype(np.int64)
emb_c, enc_c, _ = sender_circuits(small, "HbA1c 8.1%", K_QKD, DEFAULT_CHAOS)
full = emb_c.compose(enc_c)
out = dict(qubits=full.num_qubits, ops_logical=full.size(), depth_logical=full.depth(),
           depth_prep_embed=emb_c.depth(), depth_enc=enc_c.depth())
t = transpile(full, basis_gates=["cx", "u"], optimization_level=1, seed_transpiler=1)
ops = t.count_ops(); out.update(depth_elementary=t.depth(), cx=int(ops.get("cx", 0)), u=int(ops.get("u", 0)), elementary=t.size())
print(out); json.dump(out, open("results/depth.json", "w"), indent=1)
