import json, math, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from core import *
plt.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral"],"mathtext.fontset":"stix","pdf.fonttype":42})
K_QKD = bytes.fromhex(json.load(open("results/results.json"))["bb84"]["key_hex"])
cover = np.array([[236,215],[239,203]]); text = "HbA1c 8.1%"
M=2; side=4; n=2; data=text.encode()
K = master_key(K_QKD); nonce = siv_nonce(K, cover, data); S = session(K, nonce)
cap = capacity(M, M)
b = np.unpackbits(np.frombuffer(xof(S,b"pad",cap//8),np.uint8))[:cap].astype(np.int64)   # illustrative payload bits (a 2x2 cover is too small for a Huffman header)
D = (2*b[0::2]+b[1::2]).reshape(side,side)
E = np.repeat(np.repeat(cover,2,0),2,1); V = 4*E + D
Y,X = np.indices((side,side)); Y2,X2 = scramble(Y,X,round_constants(S,n),n)
Cs = np.empty_like(V); Cs[Y2,X2] = V[Y,X]; ks = keystream(S,side); C = Cs ^ ks
np.savez("results/example.npz", cover=cover, E=E, D=D, V=V, Cs=Cs, ks=ks, C=C)
# colour of every stego cell = colour of the cover pixel it came from (shows the permutation pi_S)
pal = ["#cfe3fa","#fde7a1","#d3f2dc","#f9cfe6"]
blk = [[pal[(i//2)*2+(j//2)] for j in range(4)] for i in range(4)]
Ccol = [[None]*4 for _ in range(4)]
for yy in range(4):
    for xx in range(4): Ccol[Y2[yy,xx]][X2[yy,xx]] = blk[yy][xx]
fig, ax = plt.subplots(figsize=(6.3, 1.85)); ax.axis("off")
cell = 1.0; H4 = 4*cell
def grid(A, x0, title, colors=None, fs=7.2):
    h, w = A.shape; c = H4/h; y0 = H4/2
    for i in range(h):
        for j in range(w):
            fc = colors[i][j] if colors is not None else "white"
            ax.add_patch(Rectangle((x0+j*c, y0-(i+1)*c), c, c, fc=fc, ec="#333", lw=.6, clip_on=False))
            ax.text(x0+(j+.5)*c, y0-(i+.5)*c, str(A[i,j]), ha="center", va="center", fontsize=fs if h == 4 else 9, clip_on=False)
    ax.text(x0+w*c/2, -y0-.22, title, ha="center", va="top", fontsize=8.2, clip_on=False)
    return x0, x0+w*c
def arr(x1, x2, lab=""):
    ax.add_patch(FancyArrowPatch((x1,0),(x2,0), arrowstyle="-|>", mutation_scale=8, lw=.8, color="#333", clip_on=False))
    if lab: ax.text((x1+x2)/2, .18, lab, ha="center", va="bottom", fontsize=7, clip_on=False)
xs = 0.0
a0, a1 = grid(cover, xs, r"(a) cover $I$, $2\times2$", [[pal[0],pal[1]],[pal[2],pal[3]]])
b0, b1 = grid(E, a1+1.9, r"(b) $E$: NN scaling", blk)
arr(a1+.15, b0-.15, r"$H_{y_0}H_{x_0}$")
c0, c1 = grid(D, b1+1.0, r"(c) payload $D$")
ax.text((b1+c0)/2, 0, "+", fontsize=12, ha="center", va="center")
d0, d1 = grid(V, c1+1.9, r"(d) $V=4E+D$", blk)
arr(c1+.15, d0-.15)
e0, e1 = grid(C, d1+2.3, r"(e) $C=\pi_S(V)\oplus k_S$", Ccol)
arr(d1+.15, e0-.15, r"$\pi_S$, $\oplus k_S$")
ax.set_xlim(-.1, e1+.1); ax.set_ylim(-H4/2-.8, H4/2+.1); ax.set_aspect("equal")
fig.savefig("figs/fig_example.pdf", bbox_inches="tight", pad_inches=.04); print(cover, D, V, C, sep="\n")
