"""
3D Semantic Word Space Visualizer
----------------------------------
Pick three words. See them plotted in reduced 3D embedding space with their
nearest neighbours fanned out around them. Lines connect each pair of anchors
labelled with their pairwise cosine similarities.

Renders as an interactive HTML page — open the saved file in your Windows browser.

First run downloads GloVe embeddings (~66 MB, cached to ~/gensim-data/).
Usage:
    python word_visualizer.py
    python word_visualizer.py king queen man     # pass words as args
"""

import sys
from itertools import combinations
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import gensim.downloader as api

from relations import cosine_similarity


# ── configuration ──────────────────────────────────────────────────────────────
MODEL_NAME  = "glove-wiki-gigaword-50"
N_NEIGHBORS = 8

# anchor colours (one per word)
ANCHOR_COLORS = ["#2563EB", "#DC2626", "#16A34A"]   # blue, red, green
NB_COLORS     = ["#93C5FD", "#FCA5A5", "#86EFAC"]   # light blue/red/green
SHARED_COLOR  = "#7C3AED"                            # purple – shared by 2+ anchors


def load_model():
    print(f"Loading '{MODEL_NAME}' embeddings (downloaded once, then cached)…")
    return api.load(MODEL_NAME)


def build_word_set(model, anchors: list[str], n: int):
    """
    Returns:
        unique  – ordered word list (anchors first)
        vectors – corresponding embedding matrix
        nb_sets – list of sets, one per anchor, containing that anchor's neighbours
    """
    nb_sets = [[w for w, _ in model.most_similar(a, topn=n)] for a in anchors]

    all_words = list(anchors)
    for nbs in nb_sets:
        all_words += nbs

    seen, unique = set(), []
    for w in all_words:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    vectors = np.array([model[w] for w in unique])
    return unique, vectors, [set(nbs) for nbs in nb_sets]


def reduce_3d(vectors: np.ndarray):
    pca = PCA(n_components=3)
    coords = pca.fit_transform(vectors)
    return coords, pca.explained_variance_ratio_


def build_figure(anchors, unique, coords, nb_sets, sims, variance):
    n_anchors = len(anchors)
    traces = []

    anchor_idx = [unique.index(a) for a in anchors]
    anchor_pts  = [coords[i] for i in anchor_idx]

    # ── spokes: each neighbour → its anchor(s) ────────────────────────────────
    for i, word in enumerate(unique):
        if word in anchors:
            continue
        for ai, (a, nb_set, color) in enumerate(zip(anchors, nb_sets, ANCHOR_COLORS)):
            if word in nb_set:
                p0 = anchor_pts[ai]
                p1 = coords[i]
                traces.append(go.Scatter3d(
                    x=[p0[0], p1[0], None],
                    y=[p0[1], p1[1], None],
                    z=[p0[2], p1[2], None],
                    mode="lines",
                    line=dict(color=color, width=1),
                    opacity=0.25,
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # ── bridge lines between every pair of anchors ────────────────────────────
    for (ai, aj), sim in zip(combinations(range(n_anchors), 2), sims):
        p0, p1 = anchor_pts[ai], anchor_pts[aj]
        mid = (np.array(p0) + np.array(p1)) / 2
        label = f"{anchors[ai]} ↔ {anchors[aj]}: {sim:.3f}"
        traces.append(go.Scatter3d(
            x=[p0[0], mid[0], p1[0]],
            y=[p0[1], mid[1], p1[1]],
            z=[p0[2], mid[2], p1[2]],
            mode="lines+text",
            line=dict(color="#111827", width=3),
            text=["", f"cos={sim:.3f}", ""],
            textposition="middle center",
            textfont=dict(size=11, color="#111827"),
            name=label,
            hoverinfo="name",
        ))

    # ── scatter: anchors ──────────────────────────────────────────────────────
    for ai, (a, color) in enumerate(zip(anchors, ANCHOR_COLORS)):
        p = anchor_pts[ai]
        traces.append(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]],
            mode="markers+text",
            marker=dict(size=18, color=color, symbol="diamond",
                        line=dict(color="white", width=1.5), opacity=1.0),
            text=[a],
            textposition="top center",
            textfont=dict(size=13, color="#1E293B"),
            name=f'"{a}" (anchor)',
            hovertemplate=f"<b>{a}</b><extra></extra>",
        ))

    # ── scatter: neighbours, bucketed by membership ───────────────────────────
    # Each non-anchor word gets a colour based on which anchors claim it.
    # Shared by 2+  → purple.  Exclusive → that anchor's light colour.
    for i, word in enumerate(unique):
        if word in anchors:
            continue
        membership = [ai for ai, nb_set in enumerate(nb_sets) if word in nb_set]
        if len(membership) > 1:
            color = SHARED_COLOR
            size  = 11
            group = "Shared neighbours"
        else:
            ai    = membership[0]
            color = NB_COLORS[ai]
            size  = 9
            group = f'Neighbours of "{anchors[ai]}"'

        p = coords[i]
        traces.append(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]],
            mode="markers+text",
            marker=dict(size=size, color=color, symbol="circle",
                        line=dict(color="white", width=0.8), opacity=0.88),
            text=[word],
            textposition="top center",
            textfont=dict(size=9, color="#1E293B"),
            name=group,
            legendgroup=group,
            showlegend=(word == next(
                w for w in unique
                if w not in anchors and (
                    (len(membership) > 1 and SHARED_COLOR == color) or
                    (len(membership) == 1 and membership[0] == ai)
                )
            )),
            hovertemplate=f"<b>{word}</b><extra></extra>",
        ))

    # ── layout ────────────────────────────────────────────────────────────────
    pair_sims = " · ".join(
        f"{anchors[ai]}↔{anchors[aj]}={s:.3f}"
        for (ai, aj), s in zip(combinations(range(n_anchors), 2), sims)
    )
    layout = go.Layout(
        title=dict(
            text=(f'Semantic Space · '
                  + " · ".join(f"<b>{a}</b>" for a in anchors)
                  + f'<br><sup>{pair_sims} · {MODEL_NAME}'
                  f' · PC variance: {variance[0]:.1%}/{variance[1]:.1%}/{variance[2]:.1%}</sup>'),
            x=0.5,
            font=dict(size=15),
        ),
        scene=dict(
            xaxis=dict(title=f"PC1 ({variance[0]:.1%})", backgroundcolor="#F1F5F9",
                       gridcolor="white", showbackground=True),
            yaxis=dict(title=f"PC2 ({variance[1]:.1%})", backgroundcolor="#E2E8F0",
                       gridcolor="white", showbackground=True),
            zaxis=dict(title=f"PC3 ({variance[2]:.1%})", backgroundcolor="#CBD5E1",
                       gridcolor="white", showbackground=True),
        ),
        paper_bgcolor="#F8FAFC",
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=90, b=0),
    )

    return go.Figure(data=traces, layout=layout)


def main():
    model = load_model()

    if len(sys.argv) >= 4:
        anchors = [w.lower() for w in sys.argv[1:4]]
    else:
        anchors = [input(f"Word {i+1}: ").strip().lower() for i in range(3)]

    for w in anchors:
        if w not in model:
            print(f"  '{w}' not found in vocabulary.")
            sys.exit(1)

    print(f"Building neighbourhoods for {anchors}…")
    unique, vectors, nb_sets = build_word_set(model, anchors, N_NEIGHBORS)
    coords, variance = reduce_3d(vectors)

    # pairwise cosine similarities
    sims = [
        cosine_similarity(model[anchors[ai]], model[anchors[aj]])[0]
        for ai, aj in combinations(range(len(anchors)), 2)
    ]
    for (ai, aj), s in zip(combinations(range(len(anchors)), 2), sims):
        print(f"  {anchors[ai]} ↔ {anchors[aj]}: {s:.4f}")

    shared = set.intersection(*nb_sets)
    if shared:
        print(f"  Shared by all three: {', '.join(shared)}")

    fig = build_figure(anchors, unique, coords, nb_sets, sims, variance)

    out = f"semantic_{'_'.join(anchors)}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\nSaved → {out}")
    print(f"Windows path: \\\\wsl.localhost\\Ubuntu{out.replace('/', chr(92)) if out.startswith('/') else '\\\\wsl.localhost\\Ubuntu\\home\\jvardi\\research\\bm25\\' + out}")

if __name__ == "__main__":
    main()
