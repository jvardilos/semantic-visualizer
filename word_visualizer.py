"""
3D Semantic Word Space Visualizer
----------------------------------
Pick two words. See them plotted in reduced 3D embedding space with their
nearest neighbours fanned out around them. A line connects the two anchor
words and is labelled with their cosine similarity.

Renders as an interactive HTML page in your browser — no X11 or Qt needed.

First run downloads GloVe embeddings (~66 MB, cached to ~/gensim-data/).
Usage:
    python word_visualizer.py
    python word_visualizer.py king queen      # pass words as args
"""

import sys
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import gensim.downloader as api

from relations import cosine_similarity


# ── configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = "glove-wiki-gigaword-50"   # ~66 MB; swap -100/-200/-300 for higher quality
N_NEIGHBORS = 8                          # similar words shown around each anchor


# ── colour palette ─────────────────────────────────────────────────────────────
C_WORD1  = "#2563EB"   # blue        – anchor 1
C_WORD2  = "#DC2626"   # red         – anchor 2
C_NB1    = "#93C5FD"   # light blue  – neighbours of word 1 only
C_NB2    = "#FCA5A5"   # light red   – neighbours of word 2 only
C_SHARED = "#7C3AED"   # purple      – shared neighbours
C_EDGE1  = "#3B82F6"   # spoke from anchor 1
C_EDGE2  = "#EF4444"   # spoke from anchor 2
C_BRIDGE = "#111827"   # line between anchors


def load_model():
    print(f"Loading '{MODEL_NAME}' embeddings (downloaded once, then cached)…")
    return api.load(MODEL_NAME)


def build_word_set(model, word1: str, word2: str, n: int):
    nb1 = [w for w, _ in model.most_similar(word1, topn=n)]
    nb2 = [w for w, _ in model.most_similar(word2, topn=n)]

    all_words = [word1, word2] + nb1 + nb2
    seen, unique = set(), []
    for w in all_words:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    vectors = np.array([model[w] for w in unique])
    return unique, vectors, set(nb1), set(nb2)


def reduce_3d(vectors: np.ndarray):
    pca = PCA(n_components=3)
    coords = pca.fit_transform(vectors)
    return coords, pca.explained_variance_ratio_


def classify(word, word1, word2, nb1_set, nb2_set):
    if word == word1:   return "anchor1"
    if word == word2:   return "anchor2"
    in1, in2 = word in nb1_set, word in nb2_set
    if in1 and in2:     return "shared"
    if in1:             return "nb1"
    return "nb2"


def build_figure(word1, word2, unique, coords, nb1_set, nb2_set, sim, variance):
    idx1 = unique.index(word1)
    idx2 = unique.index(word2)
    p1, p2 = coords[idx1], coords[idx2]

    traces = []

    # ── spokes: neighbours → anchors ───────────────────────────────────────────
    for i, word in enumerate(unique):
        kind = classify(word, word1, word2, nb1_set, nb2_set)
        if kind in ("nb1", "shared"):
            x0, y0, z0 = p1
            x1, y1, z1 = coords[i]
            traces.append(go.Scatter3d(
                x=[x0, x1, None], y=[y0, y1, None], z=[z0, z1, None],
                mode="lines",
                line=dict(color=C_EDGE1, width=1),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            ))
        if kind in ("nb2", "shared"):
            x0, y0, z0 = p2
            x1, y1, z1 = coords[i]
            traces.append(go.Scatter3d(
                x=[x0, x1, None], y=[y0, y1, None], z=[z0, z1, None],
                mode="lines",
                line=dict(color=C_EDGE2, width=1),
                opacity=0.3,
                showlegend=False,
                hoverinfo="skip",
            ))

    # ── bridge line between the two anchors ────────────────────────────────────
    traces.append(go.Scatter3d(
        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
        mode="lines+text",
        line=dict(color=C_BRIDGE, width=4),
        text=["", f"cos sim = {sim:.3f}"],
        textposition="middle center",
        textfont=dict(size=13, color=C_BRIDGE),
        name=f"similarity = {sim:.4f}",
        hoverinfo="name",
    ))

    # ── scatter groups ─────────────────────────────────────────────────────────
    groups = {
        "anchor1": dict(color=C_WORD1,  size=18, symbol="diamond", label=f'"{word1}" (anchor)'),
        "anchor2": dict(color=C_WORD2,  size=18, symbol="diamond", label=f'"{word2}" (anchor)'),
        "nb1":     dict(color=C_NB1,    size=9,  symbol="circle",  label=f'Neighbours of "{word1}"'),
        "nb2":     dict(color=C_NB2,    size=9,  symbol="circle",  label=f'Neighbours of "{word2}"'),
        "shared":  dict(color=C_SHARED, size=11, symbol="square",  label="Shared neighbours"),
    }

    bucket: dict[str, dict] = {k: {"x": [], "y": [], "z": [], "text": [], "hover": []}
                                for k in groups}

    for i, word in enumerate(unique):
        kind = classify(word, word1, word2, nb1_set, nb2_set)
        x, y, z = coords[i]
        nb_sims = cosine_similarity(np.array([model[word]]), np.array([[model[word1][0]]]))[0] \
            if False else None  # hover text built below
        bucket[kind]["x"].append(x)
        bucket[kind]["y"].append(y)
        bucket[kind]["z"].append(z)
        bucket[kind]["text"].append(word)

    for kind, grp in groups.items():
        b = bucket[kind]
        if not b["x"]:
            continue
        traces.append(go.Scatter3d(
            x=b["x"], y=b["y"], z=b["z"],
            mode="markers+text",
            marker=dict(
                size=grp["size"],
                color=grp["color"],
                symbol=grp["symbol"],
                line=dict(color="white", width=1),
                opacity=0.9,
            ),
            text=b["text"],
            textposition="top center",
            textfont=dict(size=10 if kind.startswith("anchor") else 9,
                          color="#1E293B"),
            name=grp["label"],
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))

    # ── layout ─────────────────────────────────────────────────────────────────
    layout = go.Layout(
        title=dict(
            text=(f'Semantic Space · "<b>{word1}</b>" ↔ "<b>{word2}</b>"'
                  f'<br><sup>Cosine Similarity = {sim:.4f} · {MODEL_NAME}'
                  f' · PC variance: {variance[0]:.1%} / {variance[1]:.1%} / {variance[2]:.1%}</sup>'),
            x=0.5,
            font=dict(size=16),
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
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#CBD5E1",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    return go.Figure(data=traces, layout=layout)


def main():
    global model
    model = load_model()

    if len(sys.argv) >= 3:
        word1, word2 = sys.argv[1].lower(), sys.argv[2].lower()
    else:
        word1 = input("First word : ").strip().lower()
        word2 = input("Second word: ").strip().lower()

    for w in (word1, word2):
        if w not in model:
            print(f"  '{w}' not found in vocabulary. Try a simpler / more common word.")
            sys.exit(1)

    print(f"Building neighbourhood for '{word1}' and '{word2}'…")
    unique, vectors, nb1_set, nb2_set = build_word_set(model, word1, word2, N_NEIGHBORS)

    coords, variance = reduce_3d(vectors)

    v1, v2 = model[word1], model[word2]
    sim = cosine_similarity(v1, v2)[0]

    print(f"Cosine similarity: {sim:.4f}")
    shared = nb1_set & nb2_set
    if shared:
        print(f"Shared neighbours: {', '.join(shared)}")

    fig = build_figure(word1, word2, unique, coords, nb1_set, nb2_set, sim, variance)

    out = f"semantic_{word1}_{word2}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"\nPlot saved → {out}")
    print(f"Open in browser: file:///$(wslpath -w $(realpath {out}))")


if __name__ == "__main__":
    main()
