"""Further modalities from the public DICOM test collections of pydicom (pydicom-data on GitHub and the
files bundled with pydicom; MIT licence). Each image is used at its native bit depth: the pixel values are
shifted by their minimum (a public offset) to non-negative integers of q_b = max(8, ceil(log2(range+1))) bits,
zero-padded to a power-of-two square, and embedded with a (q_b+2)-qubit intensity register. Colour images are
processed plane by plane. Checks: exact recovery of every original pixel, wrong-key and tamper rejection, NPCR/UACI."""
import json, math, glob, os, numpy as np, pydicom
from pydicom.data import get_testdata_file
from core import *
R = json.load(open("results/results.json")); K = bytes.fromhex(R["bb84"]["key_hex"]); Kw = bytes([K[0] ^ 1]) + K[1:]
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
rows = []
for mod, path, fr in SEL:
    for ci, a in enumerate(planes(path, fr)):
        a = a.astype(np.int64); off = int(a.min()); v = a - off; qb = max(8, int(math.ceil(math.log2(int(v.max()) + 1))))
        side0 = 1 << int(math.ceil(math.log2(max(v.shape)))); I = np.zeros((side0, side0), np.int64); I[:v.shape[0], :v.shape[1]] = v
        txt = "".join(chr(65 + (i * 7) % 26) for i in range(min(200000, side0 * side0 // 4)))   # payload text
        C, nu = embed(I, txt, K, qb=qb); rc, rt, rep = extract(C, nu, K, qb=qb)
        rec = rc[:v.shape[0], :v.shape[1]] + off
        ok = bool(np.array_equal(rec, a) and rt == txt and rep["authentic"])
        wk = not extract(C, nu, Kw, qb=qb)[2]["authentic"]
        T = C.copy(); T[3, 5] ^= 1; tp = not extract(T, nu, K, qb=qb)[2]["authentic"]
        I1 = I.copy(); I1[v.shape[0] // 2, v.shape[1] // 2] ^= 1; C1 = embed(I1, txt, K, qb=qb)[0]
        np1, ua1 = npcr_uaci(C, C1, 1 << (qb + 2))
        rows.append(dict(modality=mod, file=os.path.basename(path), frame=fr, plane=ci, shape=list(a.shape), bits=qb, cover=side0,
                         stego=2 * side0, qubits=2 * int(math.log2(2 * side0)) + qb + 2, roundtrip=ok, wrongkey=wk, tamper=tp,
                         npcr=np1, uaci=ua1, ideal_npcr=100 * (1 - 2.0 ** -(qb + 2)), ideal_uaci=100 * ((1 << (qb + 2)) + 1) / (3 * (1 << (qb + 2)))))
        print(rows[-1], flush=True)
json.dump(rows, open("results/dicom_modalities.json", "w"), indent=1)
