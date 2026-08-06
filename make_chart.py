"""Chart for the prompt-vs-train result. Outputs prompt_vs_train.png at 1200x1000."""

import matplotlib.pyplot as plt
import numpy as np

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_MUTED = "#52514e"
PROMPTED  = "#2a78d6"   # categorical slot 1
TRAINED   = "#eb6834"   # categorical slot 2

# sorted by gap, largest benefit from training at the top
rows = [
    ("chair",      0.439, 0.547),
    ("umbrella",   0.641, 0.745),
    ("toothbrush", 0.410, 0.491),
    ("person",     0.688, 0.743),
    ("scissors",   0.528, 0.549),
    ("backpack",   0.429, 0.403),
]
labels   = [r[0] for r in rows]
prompted = np.array([r[1] for r in rows])
trained  = np.array([r[2] for r in rows])

y = np.arange(len(rows))
h = 0.26
gap = 0.03          # surface gap between the paired bars

fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.barh(y - (h / 2 + gap / 2), prompted, height=h, color=PROMPTED, label="Prompted (YOLO-World S)")
ax.barh(y + (h / 2 + gap / 2), trained,  height=h, color=TRAINED,  label="Trained (YOLOv8s)")

for yi, (p, t) in enumerate(zip(prompted, trained)):
    ax.text(p + 0.008, yi - (h / 2 + gap / 2), f"{p:.3f}", va="center", ha="left",
            fontsize=13, color=INK_MUTED)
    ax.text(t + 0.008, yi + (h / 2 + gap / 2), f"{t:.3f}", va="center", ha="left",
            fontsize=13, color=INK_MUTED)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=16, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 0.85)
ax.set_xlabel("F1 score", fontsize=14, color=INK_MUTED, labelpad=12)
ax.tick_params(axis="x", labelsize=13, colors=INK_MUTED, length=0)
ax.tick_params(axis="y", length=0)

ax.xaxis.grid(True, color="#e4e3df", linewidth=1)
ax.set_axisbelow(True)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)

ax.text(0, 1.135, "Prompting a detector gets ~90% of training one",
        transform=ax.transAxes, fontsize=25, color=INK, va="bottom", weight="bold")
ax.text(0, 1.075, "Mean F1 0.523 prompted vs 0.580 trained. Same backbone, same images, "
                  "100 with the object plus 100 without.",
        transform=ax.transAxes, fontsize=14, color=INK_MUTED, va="bottom")

ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, fontsize=14,
          frameon=False, labelcolor=INK_MUTED, handlelength=1.2, columnspacing=1.8)

ax.text(0, -0.155, "COCO val2017  ·  github.com/IsrarAhmed919/yoloworld_testing",
        transform=ax.transAxes, fontsize=12, color=INK_MUTED, va="top")

plt.tight_layout()
plt.savefig("prompt_vs_train.png", facecolor=SURFACE, bbox_inches="tight", pad_inches=0.45)
print("wrote prompt_vs_train.png")
