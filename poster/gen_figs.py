from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Imatinib (a real kinase-inhibitor drug); fall back to caffeine if parsing fails
smi = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
name = "Imatinib"
mol = Chem.MolFromSmiles(smi)
if mol is None:
    smi = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"; name = "Caffeine"
    mol = Chem.MolFromSmiles(smi)
print("molecule:", name)

d = rdMolDraw2D.MolDraw2DCairo(1400, 900)
o = d.drawOptions(); o.bondLineWidth = 5; o.padding = 0.08
d.DrawMolecule(mol); d.FinishDrawing()
open("mol.png","wb").write(d.GetDrawingText())

# Morgan fingerprint, folded to 256 bits, shown as a 16x16 grid (red = set bit)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
bits = np.array(fp, dtype=int).reshape(16,16)
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(bits, cmap=ListedColormap(["white","#C40D1E"]), interpolation="nearest")
ax.set_xticks(np.arange(-.5,16,1), minor=True); ax.set_yticks(np.arange(-.5,16,1), minor=True)
ax.grid(which="minor", color="#cccccc", linewidth=0.8); ax.tick_params(which="both", length=0)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_edgecolor("#222222"); s.set_linewidth(1.5)
plt.tight_layout(pad=0.2); plt.savefig("fp.png", dpi=150, bbox_inches="tight"); print("wrote mol.png, fp.png")
print("on-bits:", int(bits.sum()), "of 256")
