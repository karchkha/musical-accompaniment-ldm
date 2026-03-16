import json
import os
import numpy as np
import matplotlib
import matplotlib.ticker
import matplotlib.gridspec
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SWEEPS = {
    "diffusion": "qn032kx2",
    "cd":        "3e970woo",
}
HERE           = os.path.dirname(os.path.abspath(__file__))
ENTITY_PROJECT = "tornike_karchkha/stream-music-gen"
CACHE_FILE     = os.path.join(HERE, "sweep_data.json")
OUT            = os.path.join(HERE, "sweep_analysis.png")
T_s            = 264600 / 44100  # ~6 s

# -----------------------------------------------------------------------
# Load data — from local cache if available, otherwise fetch from W&B
# -----------------------------------------------------------------------
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} runs from cache ({CACHE_FILE})")
else:
    import wandb
    api  = wandb.Api(timeout=120)
    data = []
    for model_name, sweep_id in SWEEPS.items():
        runs = api.runs(ENTITY_PROJECT, filters={"sweep": sweep_id}, per_page=100)
        for run in runs:
            s = run.summary
            if "fad/score" not in s:
                print(f"SKIP {run.id} {run.name} (no metrics)")
                continue
            cfg = run.config
            data.append({
                "model":       model_name,
                "r":           cfg.get("r"),
                "w":           cfg.get("w"),
                "fad":         s["fad/score"],
                "beat":        s["beat_alignment/pred_f_measure/mean"],
                "beat_std":    s["beat_alignment/pred_f_measure/std"],
                "gt_beat":     s["beat_alignment/gt_f_measure/mean"],
                "cocola_both": s["cocola/both/pred_scores/mean"],
                "cocola_harm": s["cocola/harmonic/pred_scores/mean"],
                "cocola_perc": s["cocola/percussive/pred_scores/mean"],
                "gt_cocola":   s["cocola/both/gt_scores/mean"],
            })
            print(f"OK  [{model_name}]  {run.id}  r={cfg.get('r')}  w={cfg.get('w')}  FAD={s['fad/score']:.3f}")
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(data)} runs to {CACHE_FILE}")

data = [d for d in data if d["r"] > 0.0625]
data.sort(key=lambda x: (x["r"], x["w"]))
data_diff = [d for d in data if d.get("model", "diffusion") == "diffusion"]
data_cd   = [d for d in data if d.get("model") == "cd"]
print(f"{len(data_diff)} diffusion runs, {len(data_cd)} CD runs (r >= 0.125)")

# -----------------------------------------------------------------------
# Load paper baselines
# -----------------------------------------------------------------------
with open(os.path.join(HERE, "baselines.json")) as f:
    baselines = json.load(f)

# -----------------------------------------------------------------------
# Plot config
# -----------------------------------------------------------------------
w_vals  = sorted(set(d["w"] for d in data))
colors  = {-1: "tab:blue", 0: "tab:orange", 1: "tab:green"}
markers = {-1: "^", 0: "o", 1: "s"}
labels  = {-1: "w=-1 (offline)", 0: "w=0 (causal)", 1: "w=1 (lookahead)"}

def r_to_pct(r): return r * 100

# -----------------------------------------------------------------------
# Single figure: bottom 1×3 comparison plots
# (top 1×3 sweep plots commented out)
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(18, 5.5))
# fig.suptitle("Comparison with Paper Baselines",
#              fontsize=14, fontweight="bold", y=1.01)

# # --- top sweep plots (commented out) ---
# gs_top = matplotlib.gridspec.GridSpec(1, 3, figure=fig,
#                                       top=0.93, bottom=0.56,
#                                       hspace=0.40, wspace=0.30)
# ax_fad, ax_beat, ax_cocola = [fig.add_subplot(gs_top[0, c]) for c in range(3)]
#
# def setup_r_axis(ax):
#     """Log x-axis with ticks labeled as % of window and step size in seconds."""
#     r_vals = sorted(set(d["r"] for d in data))
#     ticks  = [r_to_pct(r) for r in r_vals]
#     ax.set_xscale("log")
#     ax.set_xticks(ticks)
#     ax.set_xticklabels([f"{r_to_pct(r):.2f}%\n({r*T_s:.2f}s)" for r in r_vals], fontsize=7)
#     ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
#     ax.set_xlabel("Step size  [% of window | seconds]")
#
# # --- FAD vs step size ---
# ax = ax_fad
# for w in w_vals:
#     pts = sorted([(r_to_pct(d["r"]), d["fad"]) for d in data if d["w"] == w])
#     if pts:
#         xs, ys = zip(*pts)
#         ax.plot(xs, ys, marker=markers[w], color=colors[w], label=labels[w], linewidth=2, markersize=8)
# setup_r_axis(ax)
# ax.set_ylabel("FAD ↓"); ax.set_title("FAD Score vs step size")
# ax.legend(); ax.grid(True, alpha=0.3)
#
# # --- Beat alignment vs step size ---
# ax = ax_beat
# for w in w_vals:
#     pts = sorted([(r_to_pct(d["r"]), d["beat"]) for d in data if d["w"] == w])
#     if pts:
#         xs, ys = zip(*pts)
#         ax.plot(xs, ys, marker=markers[w], color=colors[w], label=labels[w], linewidth=2, markersize=8)
# b = baselines["beat_alignment_f1"]
# ax.axhline(b["ground_truth"],   color="k",    linestyle="--", linewidth=2.0, alpha=0.9, label=f"GT {b['ground_truth']:.3f}")
# ax.axhline(b["random_pairing"], color="gray", linestyle="--", linewidth=2.0, alpha=0.9, label=f"Random Pairing {b['random_pairing']:.3f}")
# setup_r_axis(ax)
# ax.set_ylabel("Beat F-measure ↑"); ax.set_title("Beat Alignment vs step size")
# ax.legend(); ax.grid(True, alpha=0.3)
#
# # --- COCOLA vs step size ---
# ax = ax_cocola
# for w in w_vals:
#     pts = sorted([(r_to_pct(d["r"]), d["cocola_both"]) for d in data if d["w"] == w])
#     if pts:
#         xs, ys = zip(*pts)
#         ax.plot(xs, ys, marker=markers[w], color=colors[w], label=labels[w], linewidth=2, markersize=8)
# c = baselines["cocola_overall"]
# ax.axhline(c["ground_truth"],   color="k",    linestyle="--", linewidth=2.0, alpha=0.9, label=f"GT {c['ground_truth']:.1f}")
# ax.axhline(c["random_pairing"], color="gray", linestyle="--", linewidth=2.0, alpha=0.9, label=f"Random Pairing {c['random_pairing']:.1f}")
# setup_r_axis(ax)
# ax.set_ylabel("COCOLA ↑"); ax.set_title("COCOLA (both stems) vs step size")
# ax.legend(); ax.grid(True, alpha=0.3)

gs_bot = matplotlib.gridspec.GridSpec(1, 3, figure=fig,
                                      top=0.88, bottom=0.30,
                                      hspace=0.40, wspace=0.32)
axes2 = [fig.add_subplot(gs_bot[0, c]) for c in range(3)]

# -----------------------------------------------------------------------
# Bottom row: comparison with paper baselines
# x-axis convention (ours): net_lookahead = T_s * r * w
#   x < 0 → Generate Behind (more future visible, w=-1)
#   x > 0 → Generate Ahead  (less future visible, w=+1)
# Paper t_f is flipped: x_paper = -t_f  (paper's positive t_f = behind = our negative x)
# Paper offline (∞ future) → plotted at OFFLINE_X = -6.8 (left of axis)
# -----------------------------------------------------------------------
SPREAD_W0 = True   # spread w=0 points across x=[-IMM_W, +IMM_W] as an area; set False to stack at x=0
IMM_W     = 2.5     # half-width of Immediate zone
OFFLINE_X = -(IMM_W + 7.5)  # stand-in for −∞ (offline / max future visible)
X_MIN     = -(IMM_W + 8.0)
X_MAX     =  (IMM_W + 7.0)

pc  = baselines["paper_curve"]
osg = baselines["offline_stemgen"]
opd = baselines["offline_prefix_decoder"]
c   = baselines["cocola_overall"]
b   = baselines["beat_alignment_f1"]

comparison_metrics = [
    ("cocola_both", "COCOLA score↑",  pc["cocola"], osg["cocola"], opd["cocola"],
     c["our_ground_truth"], c["baseline_ground_truth"], c["random_pairing"]),
    ("beat",        "Beat alignment F1 ↑", pc["beat"],   osg["beat"],   opd["beat"],
     b["our_ground_truth"], b["baseline_ground_truth"], b["random_pairing"]),
    ("fad",         "FAD score↓",     pc["fad"],    osg["fad"],    opd["fad"],
     None, None, None),
]

# paper t_f → our zone-offset coordinate system:
#   t_f > 0 (paper "behind") → our Generate Behind zone: -(IMM_W + t_f)
#   t_f < 0 (paper "ahead")  → our Generate Ahead zone:  +(IMM_W + |t_f|)
#   t_f = 0 (causal)         → 0 (center of Immediate zone)
paper_x = [-(IMM_W + t) if t > 0 else (IMM_W + abs(t)) if t < 0 else 0
           for t in pc["t_f"]]

# Pre-compute w=0 x positions (same for all metrics) — r-proportional, padded inside zone
w0_r_vals = sorted(set(d["r"] for d in data_diff if d["w"] == 0))
r_max_w0  = max(w0_r_vals)
# x = r/r_max * IMM_W * 0.82, so max r sits at 82% of zone half-width (never at edge)
w0_xs_by_r = {r: (r / r_max_w0) * IMM_W * 0.82 * (1 if len(w0_r_vals) > 1 else 0)
              for r in w0_r_vals}
# For single-r case fall back to 0; for multi-r spread positively from ~0
if len(w0_r_vals) > 1:
    r_min_w0 = min(w0_r_vals)
    span = r_max_w0 - r_min_w0
    w0_xs_by_r = {r: ((r - r_min_w0) / span * 2 - 1) * IMM_W * 0.80
                  for r in w0_r_vals}

for ax, (key, ylabel, paper_vals, osg_val, opd_val, our_gt_val, baseline_gt_val, rand_val) in zip(axes2, comparison_metrics):
    # Background shading: Generate Behind | Immediate | Generate Ahead
    ax.axvspan(X_MIN,  -IMM_W, alpha=0.12, color="green",      zorder=0)
    ax.axvspan(-IMM_W,  IMM_W, alpha=0.15, color="lightgray",  zorder=0)
    ax.axvspan( IMM_W,  X_MAX, alpha=0.12, color="salmon",     zorder=0)
    ax.axvline(-IMM_W, color="gray", linewidth=0.6, linestyle=":", zorder=1)
    ax.axvline( IMM_W, color="gray", linewidth=0.6, linestyle=":", zorder=1)

    # Baseline Online Decoder curve (t_f flipped to our coords)
    ax.plot(paper_x, paper_vals, color="royalblue", linewidth=2.0,
            marker="D", markersize=5, label="Baseline Online Decoder", zorder=4)

    # Baseline offline markers
    ax.scatter([OFFLINE_X], [osg_val], marker="*", s=200, color="royalblue",
               zorder=5, label="Baseline StemGen (offline)")
    ax.scatter([OFFLINE_X], [opd_val], marker="P", s=150, color="steelblue",
               zorder=5, label="Baseline Prefix Dec (offline)")

    if our_gt_val is not None:
        ax.axhline(our_gt_val,      color="k",          linestyle="--", linewidth=1.8, alpha=0.85,
                   label="GT (Ours)", zorder=3)
        ax.axhline(baseline_gt_val, color="royalblue",  linestyle="--", linewidth=1.8, alpha=0.85,
                   label="GT (Baseline)", zorder=3)
        ax.axhline(rand_val,        color="gray",        linestyle="--", linewidth=1.8, alpha=0.85,
                   label="Random Pairing", zorder=3)

    # Our model curve — zone-offset coordinates:
    #   w=-1: x = -(IMM_W + T_s*r)  (starts at left boundary, extends left)
    #   w=0:  spread across [-IMM_W, +IMM_W] by r ascending
    #   w=+1: x = +(IMM_W + T_s*r)  (starts at right boundary, extends right)
    def plot_model_curve(ax, mdata, color, label, marker="o"):
        if SPREAD_W0:
            w0_pts = sorted([(d["r"], d[key]) for d in mdata if d["w"] == 0])
            w0_xs  = [w0_xs_by_r[r] for r, _ in w0_pts if r in w0_xs_by_r]
            w0_ys  = [y for r, y in w0_pts if r in w0_xs_by_r]
            neg = sorted([(-(IMM_W + T_s*d["r"]), d[key]) for d in mdata if d["w"] == -1])
            pos = sorted([ (+(IMM_W + T_s*d["r"]), d[key]) for d in mdata if d["w"] ==  1])
            xs = [x for x,_ in neg] + w0_xs + [x for x,_ in pos]
            ys = [y for _,y in neg] + w0_ys + [y for _,y in pos]
        else:
            neg = sorted([(-(IMM_W + T_s*d["r"]), d[key]) for d in mdata if d["w"] == -1])
            w0  = sorted([(0.0, d[key]) for d in mdata if d["w"] == 0])
            pos = sorted([ (+(IMM_W + T_s*d["r"]), d[key]) for d in mdata if d["w"] ==  1])
            xs, ys = zip(*neg + w0 + pos)
        ax.plot(xs, ys, color=color, marker=marker, markersize=5,
                linewidth=1.8, zorder=6, label=label)

    plot_model_curve(ax, data_diff, "crimson",    "Diffusion (ours)", marker="o")
    if data_cd:
        plot_model_curve(ax, data_cd,   "darkorange", "CD (ours)",        marker="s")

    # Ticks: OFFLINE on left; zone-boundary at ±IMM_W; steps at ±(IMM_W+Δ); r labels in middle
    raw_ticks   = [OFFLINE_X,
                   -(IMM_W+6), -(IMM_W+3), -(IMM_W+1.5), -(IMM_W+0.75),
                   -IMM_W,
                   +(IMM_W+0.75), +(IMM_W+1.5), +(IMM_W+3), +(IMM_W+6)]
    tick_labels = ["offline",
                   "-6s", "-3s", "-1.5s", "-0.75s",
                   "0",
                   "0.75s", "1.5s", "3s", "6s"]
    # Add r-value ticks in the Immediate zone
    w0_tick_xs  = list(w0_xs_by_r.values())
    w0_tick_lbs = [f"{r:.3g}" for r in w0_xs_by_r.keys()]
    all_ticks  = raw_ticks + w0_tick_xs + [IMM_W]
    all_labels = tick_labels + w0_tick_lbs + ["0"]
    # Sort by position
    sorted_pairs = sorted(zip(all_ticks, all_labels))
    ax.set_xticks([p[0] for p in sorted_pairs])
    ax.set_xticklabels([p[1] for p in sorted_pairs], rotation=40, ha="right")
    ax.tick_params(axis='x', labelsize=9)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xlabel("← net lookahead  T · r · w  (seconds) →", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    # ax.set_title(f"{ylabel.split(' ')[0]}")
    ax.grid(True, alpha=0.3)

    # Zone labels: extend y-axis bottom to create blank space, place labels there
    ylo, yhi = ax.get_ylim()
    yrange   = yhi - ylo
    ylo_new  = ylo - yrange * 0.05   # 15% extra space at bottom
    ax.set_ylim(ylo_new, yhi)
    y_label  = ylo_new + yrange * 0.01  # just above the new bottom
    x_behind    = (X_MIN + (-IMM_W)) / 2
    x_immediate = 0.0
    x_ahead     = (IMM_W + X_MAX) / 2
    ax.text(x_behind,    y_label, "Retrospective", fontsize=9, color="darkgreen",
            ha="center", va="bottom", alpha=0.85)
    ax.text(x_immediate, y_label, "Immediate",       fontsize=9, color="gray",
            ha="center", va="bottom", alpha=0.95)
    ax.text(x_ahead,     y_label, "Look-ahead",  fontsize=9, color="firebrick",
            ha="center", va="bottom", alpha=0.85)

# Shared legend below the bottom row
handles, lbls = axes2[0].get_legend_handles_labels()
seen = {}
for h, l in zip(handles, lbls):
    if l not in seen:
        seen[l] = h
fig.legend(seen.values(), seen.keys(),
           loc="lower center", ncol=7, fontsize=10,
           bbox_to_anchor=(0.5, 0.09), frameon=True)

plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
