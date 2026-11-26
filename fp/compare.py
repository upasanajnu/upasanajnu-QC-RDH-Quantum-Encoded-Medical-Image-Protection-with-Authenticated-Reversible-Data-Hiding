"""Compares the chaotic-digest trajectories printed by the implementations in fp/ (256 post-transient states,
IEEE-754 doubles as hex) with the Python reference, and the integer-only (Q32.32) trajectories with theirs.
Regenerate an output with e.g.  gcc -O0 fp/traj.c -o t && ./t > fp/out_O0.txt  (see the commands in this file).
Updates results/fp.json."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
FLOAT = {"out_py.txt": "Python 3.13 float", "out_np.txt": "NumPy float64", "out_O0.txt": "C, gcc 13 -O0",
         "out_O2.txt": "C, gcc 13 -O2 -ffp-contract=off", "out_js.txt": "JavaScript, Node 22", "out_java.txt": "Java 21",
         "out_go.txt": "Go (GOAMD64=v1)", "out_gov3.txt": "Go (GOAMD64=v3)", "out_rust.txt": "Rust -O -C target-cpu=native",
         "out_O3fma.txt": "C, gcc 13 -O3 -march=native -ffp-contract=fast (FMA)", "out_fastmath.txt": "C, gcc 13 -O3 -ffast-math"}
FIXED = {"fx_py.txt": "Python (Q32.32)", "fx_O0.txt": "C fixed.c -O0", "fx_fast.txt": "C fixed.c -O3 -ffast-math"}
def load(f): return open(os.path.join(H, f)).read().split("\n")
def cmp(table, ref):
    r = load(ref); out = {}
    for f, name in table.items():
        s = load(f); d = sum(a != b for a, b in zip(r, s)) + abs(len(r) - len(s))
        out[name] = {"identical": d == 0, "states_differing": d}; print(f"{name:55s} {'identical' if d == 0 else f'{d} states differ'}")
    return out
P = os.path.join(H, "..", "results", "fp.json")
R = json.load(open(P)) if os.path.exists(P) else {}
R["implementations"] = cmp(FLOAT, "out_py.txt"); R["fixed_point"] = cmp(FIXED, "fx_py.txt")
json.dump(R, open(P, "w"), indent=1)
# Build commands used:  gcc -O0 traj.c; gcc -O2 -ffp-contract=off traj.c; gcc -O3 -march=native -ffp-contract=fast traj.c;
# gcc -O3 -ffast-math traj.c; node traj.js; javac T.java && java T; (cd gt && go run .); GOAMD64=v3 go run .;
# rustc -O -C target-cpu=native r.rs; gcc -O0 fixed.c; gcc -O3 -ffast-math fixed.c
