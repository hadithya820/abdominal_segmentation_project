import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Change these to your file paths ---
unet_csv = "results/unet_dice.csv"
attn_csv = "results/attention_unet_dice.csv"

# --- Load CSVs ---
df_unet = pd.read_csv(unet_csv)
df_attn = pd.read_csv(attn_csv)

# --- Set 'Organ' as the index for easy lookup ---
df_unet.set_index("Organ", inplace=True)
df_attn.set_index("Organ", inplace=True)

organs = ['Liver', 'Kidneys', 'Spleen']
mean_unet = [df_unet.loc[org, "Mean Dice"] for org in organs]
mean_attn = [df_attn.loc[org, "Mean Dice"] for org in organs]

# --- Build comparison DataFrame ---
df_compare = pd.DataFrame({
    'Organ': organs,
    'U-Net': mean_unet,
    'Attention U-Net': mean_attn
})

print(df_compare)

# --- Plot ---
ax = df_compare.plot.bar(
    x='Organ',
    figsize=(8, 5),
    ylabel='Mean Dice',
    title='Per-Organ Dice Score Comparison',
    rot=0
)
plt.tight_layout()
outdir = Path("results/model_comparison_plots")
outdir.mkdir(parents=True, exist_ok=True)
plt.savefig(outdir / "per_organ_dice.png", dpi=150)
plt.show()
