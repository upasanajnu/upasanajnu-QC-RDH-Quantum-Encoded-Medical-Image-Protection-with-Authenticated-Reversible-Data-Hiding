"""Room-reserving configuration (RR): lossless JPEG-LS compression of the cover before quantum encoding.
Runs on the 16 main images (container width w = 8 and w = 16), on all 157 slices of the four clinical volumes
(8-bit windowed and native 12-bit) and on the 29 further DICOM images at native depth. Full payloads.
Reports exact recovery, wrong-key and tamper rejection, rate (coded payload bits per transmitted bit) and
stego statistics against their ideal values."""
import json, math, os, sys, numpy as np, nrrd
from scipy.stats import chi2 as CHI2
from core import *
from core import _jls_encode
R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"]); Kw = bytes([K[0] ^ 1]) + K[1:]
alph = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:-/()%+", np.uint8)
def fill(cap, seed=7):
    rng = np.random.default_rng(seed); n = max(10, int(cap / 6.25))
    while True:
        t = bytes(rng.choice(alph, n)).decode()
        if len(encode_payload(t.encode())) <= cap: return t
        n -= max(20, n // 400)
def run(I, qb, extra=0, stats=True):
    M = I.shape[0]; w = qb + extra; L = 1 << w; cap = rr_capacity(I, qb, extra); txt = fill(cap)
    C, nu = embed_rr(I, txt, K, qb=qb, extra=extra); rc, rt, rep = extract_rr(C, nu, K, qb=qb, extra=extra)
    used = len(encode_payload(txt.encode()))
    r = dict(size=M, bits=qb, w=w, cap=cap, payload=used, load=100 * used / cap, rate=used / (w * I.size),
             compressed_bpp=8 * len(_jls_encode(I, qb)) / I.size,
             roundtrip=bool(np.array_equal(rc, I) and rt == txt and rep["authentic"]),
             wrongkey=not extract_rr(C, nu, Kw, qb=qb, extra=extra)[2]["authentic"])
    T = C.copy(); T[3, 5] ^= 1; r["tamper"] = not extract_rr(T, nu, K, qb=qb, extra=extra)[2]["authentic"]
    if stats:
        I1 = I.copy(); I1[M // 2, M // 2] ^= 1
        try: C1 = embed_rr(I1, txt, K, qb=qb, extra=extra)[0]; r["npcr"], r["uaci"] = npcr_uaci(C, C1, L)
        except ValueError: r["npcr"] = r["uaci"] = None              # changed cover may compress worse
        ref = np.random.default_rng(M).integers(0, L, C.shape)
        r.update(H=entropy(C, L), H_ref=entropy(ref, L), chi2=chi_square(C, L), chi2_crit=float(CHI2.ppf(.95, L - 1)),
                 rmax=max(abs(corr(C, 1, 0)), abs(corr(C, 0, 1)), abs(corr(C, 1, 1))),
                 ideal_npcr=100 * (1 - 1 / L), ideal_uaci=100 * (L + 1) / (3 * L))
    return r
out = {}
A = np.load("results/images.npz"); B = np.load("results/medical_imgs.npz")
main = [(k, A[k].astype(np.int64)) for k in A] + [(k, B[k].astype(np.int64)) for k in B if not k.endswith(("_s", "_r"))]
for extra in (0, 8):
    out[f"main_w{8 + extra}"] = []
    for name, I in main:
        r = run(I, 8, extra); r["image"] = name; out[f"main_w{8 + extra}"].append(r); print(extra, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
def to8(a, lo, hi): return np.clip(np.rint((a.astype(float) - lo) / (hi - lo) * 255), 0, 255).astype(np.int64)
out["slices"] = []
for v in ["lung1", "lung2", "brain1", "brain2"]:
    vol = nrrd.read(f"data_med/{v}_image.nrrd")[0]; mod = "CT" if v.startswith("lung") else "MR"
    for z in range(vol.shape[2]):
        a = vol[:, :, z].T
        if mod == "CT": I8 = to8(a, -1350, 150); I12 = np.clip(a.astype(np.int64) + 1024, 0, 4095)
        else:
            lo, hi = np.percentile(a, [.5, 99.5]); I8 = to8(a, lo, hi) if hi > lo else np.zeros_like(a, np.int64)
            I12 = np.clip(a.astype(np.int64), 0, 4095)
        for qb, I in ((8, I8), (12, I12)):
            r = run(I, qb, 0, stats=False); r.update(volume=v, modality=mod, slice=z); out["slices"].append(r)
    print(v, "done", flush=True)
import os, pydicom
from pydicom.data import get_testdata_file
D1 = os.environ.get("PYDICOM_DATA", "data_dicom/data_store/data/").rstrip("/") + "/"
SEL = [("CT", D1 + "693_UNCR.dcm", None), ("CT", D1 + "bad_sequence.dcm", None), ("CT", D1 + "explicit_VR-UN.dcm", None),
       ("CT", D1 + "eCT_Supplemental.dcm", 0), ("CT", D1 + "eCT_Supplemental.dcm", 1), ("CT", get_testdata_file("CT_small.dcm"), None),
       ("MR", D1 + "MR2_UNCR.dcm", None), ("MR", D1 + "MR-SIEMENS-DICOM-WithOverlays.dcm", None), ("MR", get_testdata_file("MR_small.dcm"), None)] + \
      [("MR", D1 + "emri_small.dcm", z) for z in range(10)] + \
      [("CR", D1 + "RG1_UNCR.dcm", None), ("CR", D1 + "RG3_UNCR.dcm", None),
       ("US", D1 + "JPGLosslessP14SV1_1s_1f_8b.dcm", None), ("US", D1 + "OBXXXX1A.dcm", None), ("US", get_testdata_file("examples_palette.dcm"), None),
       ("US", D1 + "gdcm-US-ALOKA-16.dcm", None), ("US", D1 + "US1_UNCR.dcm", None), ("US", get_testdata_file("examples_rgb_color.dcm"), None),
       ("US", D1 + "color-px.dcm", None), ("NM", D1 + "JPEG-LL.dcm", None)]
def planes(path, frame):
    d = pydicom.dcmread(path); a = d.pixel_array
    if frame is not None: a = a[frame]
    return [a[..., c] for c in range(3)] if a.ndim == 3 else [a]
out["dicom"] = []
for mod, path, fr in SEL:
    for ci, a in enumerate(planes(path, fr)):
        a = a.astype(np.int64); off = int(a.min()); v = a - off; qb = max(8, int(math.ceil(math.log2(int(v.max()) + 1))))
        side0 = 1 << int(math.ceil(math.log2(max(v.shape)))); I = np.zeros((side0, side0), np.int64); I[:v.shape[0], :v.shape[1]] = v
        r = run(I, qb, 0, stats=False); r.update(modality=mod, file=os.path.basename(path), frame=fr, plane=ci, shape=list(a.shape))
        out["dicom"].append(r); print(r, flush=True)
json.dump(out, open("results/rr.json", "w"), indent=1)
