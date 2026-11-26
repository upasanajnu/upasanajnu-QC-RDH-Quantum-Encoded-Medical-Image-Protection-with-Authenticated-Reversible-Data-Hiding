"""Identical-data reference for classical RDHEI: on exactly the covers used for QC-RDH, measure the lossless
bit rate R of JPEG-LS (ISO/IEC 14495-1, CharLS via imagecodecs). A reserve-room-before-encryption scheme that
compresses the cover losslessly before encryption can vacate at most about 8 - R bits per pixel (side information
not counted), i.e. (8 - R)/8 payload bits per transmitted bit. QC-RDH: 2 bits per 10-bit stego pixel = 0.2."""
import json, numpy as np, nrrd, imagecodecs as ic
def rate(I):
    a = I.astype(np.uint8); e = ic.jpegls_encode(a); assert np.array_equal(ic.jpegls_decode(e), a)
    return 8 * len(e) / a.size
def to8(a, lo, hi): return np.clip(np.rint((a.astype(float) - lo) / (hi - lo) * 255), 0, 255).astype(np.int64)
A = np.load("results/images.npz"); B = np.load("results/medical_imgs.npz")
groups = {"natural (8)": [A[k] for k in A], "clinical (8)": [B[k] for k in B if not k.endswith(("_s", "_r"))]}
ct, mr = [], []
for v in ["lung1", "lung2", "brain1", "brain2"]:
    vol = nrrd.read(f"data_med/{v}_image.nrrd")[0]
    for z in range(vol.shape[2]):
        a = vol[:, :, z].T
        if v.startswith("lung"): ct.append(to8(a, -1350, 150))
        else:
            lo, hi = np.percentile(a, [.5, 99.5]); mr.append(to8(a, lo, hi) if hi > lo else np.zeros_like(a, np.int64))
groups["CT, all slices (109)"] = ct; groups["MRI, all slices (48)"] = mr
out = {}
for g, L in groups.items():
    R = np.array([rate(I) for I in L]); room = 8 - R
    out[g] = dict(n=len(L), R_mean=float(R.mean()), R_min=float(R.min()), R_max=float(R.max()),
                  room_mean=float(room.mean()), per_bit_mean=float(room.mean() / 8), per_bit_min=float(room.min() / 8), per_bit_max=float(room.max() / 8))
    print(g, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[g].items()})
out["QC-RDH"] = dict(bpp_stego=2.0, per_bit=0.2)
json.dump(out, open("results/baseline_jpegls.json", "w"), indent=1)
