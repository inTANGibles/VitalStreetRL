"""
Plotting: graph structure, node types, mask, pred vs true, error distribution, training curves.
All matplotlib, no seaborn.
"""
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


def _pos_from_gdf_data(data: Data, gdf: Any) -> Optional[Dict[int, Tuple[float, float]]]:
    """从 Data.node_id 与 gdf 行索引对应得到节点位置。返回 {node_idx: (x, y)}。"""
    if not hasattr(data, "node_id"):
        return None
    pos = {}
    try:
        centroids = gdf.geometry.centroid
        for i in range(data.num_nodes):
            nid = int(data.node_id[i].item())
            if nid >= len(centroids):
                continue
            p = centroids.iloc[nid]
            if hasattr(p, "is_empty") and p.is_empty:
                continue
            try:
                x, y = float(p.x), float(p.y)
                if not (math.isnan(x) or math.isnan(y)):
                    pos[i] = (x, y)
            except Exception:
                continue
        return pos if len(pos) > 0 else None
    except Exception:
        return None


def _data_to_nx(data: Data) -> nx.Graph:
    """同构 Data -> NetworkX，节点 id 为 0..N-1。"""
    G = nx.Graph()
    n = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
    G.add_nodes_from(range(n))
    if hasattr(data, "edge_index") and data.edge_index is not None and data.edge_index.numel() > 0:
        ei = data.edge_index
        for j in range(ei.size(1)):
            u, v = int(ei[0, j].item()), int(ei[1, j].item())
            G.add_edge(u, v)
    return G


def plot_graph(
    data: Data,
    out_path: Optional[Path] = None,
    title: str = "Graph",
    center_node: Optional[int] = None,
    gdf: Optional[Any] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    同构图可视化。center_node: 子图内中心节点局部索引，高亮显示。
    gdf: 与 nodes.csv 顺序一致时可用于底图与节点位置。
    """
    G = _data_to_nx(data)
    pos_dict = _pos_from_gdf_data(data, gdf) if gdf is not None else None
    if pos_dict is not None and len(pos_dict) == G.number_of_nodes():
        pos = pos_dict
    else:
        pos = nx.spring_layout(G, seed=42, k=0.5)
    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    if gdf is not None and pos_dict is not None:
        gdf_valid = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
        if not gdf_valid.empty:
            try:
                gdf_valid.plot(ax=ax, facecolor="lightgray", edgecolor="gray", alpha=0.5, linewidth=0.5, aspect="equal")
            except ValueError:
                pass
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
    nodelist = [n for n in G.nodes() if n != center_node]
    # shop 与 public_space 分开着色（需 gdf 与节点顺序一致且含 unit_type）
    if gdf is not None and len(gdf) == G.number_of_nodes() and "unit_type" in gdf.columns:
        shop_nodes = [n for n in nodelist if n < len(gdf) and gdf.iloc[n]["unit_type"] == "shop"]
        public_nodes = [n for n in nodelist if n < len(gdf) and gdf.iloc[n]["unit_type"] == "public_space"]
        other_nodes = [n for n in nodelist if n not in shop_nodes and n not in public_nodes]
        if shop_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=shop_nodes, node_color="tab:orange", ax=ax, node_size=80, label="shop")
        if public_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=public_nodes, node_color="tab:blue", ax=ax, node_size=80, label="public_space")
        if other_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color="gray", ax=ax, node_size=80)
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="tab:blue", ax=ax, node_size=80)
    if center_node is not None and center_node in G:
        nx.draw_networkx_nodes(G, pos, nodelist=[center_node], node_color="red", label="center", ax=ax, node_size=200)
        nx.draw_networkx_labels(G, pos, {center_node: "C"}, font_size=10, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    if pos_dict is not None and len(pos_dict) == G.number_of_nodes():
        ax.set_aspect("equal")
    if ax is None:
        plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_subgraph_comparison(
    data_full: Data,
    sub_data: Data,
    center_global: int,
    center_label: Optional[float] = None,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """并排展示：整图（中心高亮）与 2-hop 子图。center_global: 整图中中心节点全局索引。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    G = _data_to_nx(data_full)
    pos = nx.spring_layout(G, seed=42, k=0.5)
    ax = axes[0]
    nodelist = [n for n in G.nodes() if n != center_global]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color="tab:blue", ax=ax, node_size=60)
    if center_global in G:
        nx.draw_networkx_nodes(G, pos, nodelist=[center_global], node_color="red", ax=ax, node_size=180)
        nx.draw_networkx_labels(G, pos, {center_global: "C"}, font_size=10, ax=ax)
    ax.set_title("Full graph (center highlighted)")
    ax.axis("off")
    G_sub = _data_to_nx(sub_data)
    pos_sub = nx.spring_layout(G_sub, seed=42, k=0.6)
    ax = axes[1]
    center_local = getattr(sub_data, "center_idx", 0)
    nx.draw_networkx_edges(G_sub, pos_sub, ax=ax, alpha=0.4)
    nx.draw_networkx_nodes(G_sub, pos_sub, nodelist=[n for n in G_sub.nodes() if n != center_local], node_color="tab:blue", ax=ax, node_size=80)
    if center_local in G_sub:
        nx.draw_networkx_nodes(G_sub, pos_sub, nodelist=[center_local], node_color="red", ax=ax, node_size=200)
        nx.draw_networkx_labels(G_sub, pos_sub, {center_local: "C"}, font_size=10, ax=ax)
    lbl = f", label={center_label:.1f}" if center_label is not None else ""
    ax.set_title(f"2-hop subgraph (n={sub_data.num_nodes}{lbl})")
    ax.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_histograms(
    x: torch.Tensor,
    feature_names: List[str],
    title: str = "Node features",
    out_path: Optional[Path] = None,
    n_cols: int = 4,
) -> plt.Figure:
    """Histograms for each of the 11 continuous features."""
    n_feat = x.size(1)
    n_cols = min(n_cols, n_feat)
    n_rows = (n_feat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_feat > 1 else [axes]
    for i in range(n_feat):
        axes[i].hist(x[:, i].numpy().ravel(), bins=30, edgecolor="black", alpha=0.7)
        axes[i].set_title(feature_names[i] if i < len(feature_names) else f"dim_{i}")
    for j in range(n_feat, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_time_feat_values(
    time_slices: List[Tuple[int, int]],
    use_slot: bool = False,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot 4 time features across time slices (e.g. 42 or 220). use_slot: 第二维为 slot_idx。"""
    sin_h = []
    cos_h = []
    sin_d = []
    cos_d = []
    for day, val in time_slices:
        if use_slot:
            hour_frac = 12 + val // 4 + (val % 4) * 0.25
        else:
            hour_frac = val
        sin_h.append(math.sin(2 * math.pi * hour_frac / 24))
        cos_h.append(math.cos(2 * math.pi * hour_frac / 24))
        sin_d.append(math.sin(2 * math.pi * day / 7))
        cos_d.append(math.cos(2 * math.pi * day / 7))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(sin_h, ".-")
    axes[0, 0].set_title("sin(2π*hour/24)")
    axes[0, 1].plot(cos_h, ".-")
    axes[0, 1].set_title("cos(2π*hour/24)")
    axes[1, 0].plot(sin_d, ".-")
    axes[1, 0].set_title("sin(2π*day/7)")
    axes[1, 1].plot(cos_d, ".-")
    axes[1, 1].set_title("cos(2π*day/7)")
    plt.suptitle("Time features across time slices")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_labeled_count_per_slice(
    values: List[float],
    out_path: Optional[Path] = None,
    ylabel: str = "Number of labeled public nodes",
    title: str = "Labeled public nodes per time slice",
) -> plt.Figure:
    """Bar or line: values per time slice（如客流量、有标签节点数等）。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values, ".-")
    ax.set_xlabel("Time slice index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_flow_distribution_log1p(y_list: List[torch.Tensor], titles: List[str], out_path: Optional[Path] = None) -> plt.Figure:
    """Histograms of log1p(flow) for 3 time slices."""
    n = len(y_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, y, t in zip(axes, y_list, titles):
        vals = y.numpy().ravel()
        vals = vals[vals > 0]  # only labeled
        ax.hist(vals, bins=20, edgecolor="black", alpha=0.7)
        ax.set_title(t)
        ax.set_xlabel("log1p(flow)")
    plt.suptitle("Public node flow (log1p) distribution")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_train_curves(
    epochs: List[int],
    train_loss: List[float],
    val_mae: List[float],
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Train loss and val MAE vs epoch."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_loss, "b.-")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax1.set_title("Train loss vs epoch")
    ax2.plot(epochs, val_mae, "g.-")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val MAE")
    ax2.set_title("Val MAE vs epoch")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pred_vs_true(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: torch.Tensor,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Scatter plot: prediction vs ground truth (masked points)."""
    yt = y_true[mask].numpy().ravel()
    yp = y_pred[mask].numpy().ravel()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, alpha=0.6)
    mn = min(yt.min(), yp.min())
    mx = max(yt.max(), yp.max())
    ax.plot([mn, mx], [mn, mx], "r--", label="y=x")
    ax.set_xlabel("True (log1p flow)")
    ax.set_ylabel("Predicted (log1p flow)")
    ax.set_title("Prediction vs True (val public labeled)")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pred_vs_true_by_subgraph(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    center_ids: torch.Tensor,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    按静态子图聚合：同一 center 的多个时间片取 True/Pred 的平均值，再画 Pred vs True。
    用于查看「同一子图对应一个平均标签、一个平均预测」时预测是否合理。
    center_ids: 与 y_true/y_pred 同长的整数张量，表示每个样本的全局 center 索引。
    """
    yt = y_true.numpy().ravel()
    yp = y_pred.numpy().ravel()
    cid = center_ids.numpy().ravel()
    uniq = np.unique(cid)
    mean_trues = np.array([yt[cid == c].mean() for c in uniq])
    mean_preds = np.array([yp[cid == c].mean() for c in uniq])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(mean_trues, mean_preds, alpha=0.8, label=f"by subgraph (n={len(uniq)})")
    mn = min(mean_trues.min(), mean_preds.min())
    mx = max(mean_trues.max(), mean_preds.max())
    ax.plot([mn, mx], [mn, mx], "r--", label="y=x")
    # 数据自身的拟合曲线：pred = a * true + b（线性回归）
    if len(mean_trues) >= 2:
        coef = np.polyfit(mean_trues, mean_preds, 1)
        x_fit = np.linspace(mn, mx, 100)
        y_fit = np.polyval(coef, x_fit)
        ax.plot(x_fit, y_fit, "g-", lw=2, label=f"fit: pred={coef[0]:.3f}*true+{coef[1]:.3f}")
    ax.set_xlabel("True mean (log1p flow, per subgraph)")
    ax.set_ylabel("Predicted mean (log1p flow, per subgraph)")
    ax.set_title("Prediction vs True (val, aggregated by static subgraph)")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_histogram(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: torch.Tensor,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Histogram of prediction errors on masked nodes."""
    err = (y_pred - y_true)[mask].numpy().ravel()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(err, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Error (pred - true)")
    ax.set_ylabel("Count")
    ax.set_title("Error distribution (val)")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_nodes_by_value(
    data: Data,
    values: torch.Tensor,
    title: str = "Node values",
    out_path: Optional[Path] = None,
    value_label: str = "yhat_flow",
    gdf: Optional[Any] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """同构 Data：节点着色为 values。values 可为 [n_public]（仅 public）或 [N]。"""
    G = _data_to_nx(data)
    pos_dict = _pos_from_gdf_data(data, gdf) if gdf is not None else None
    pos = pos_dict if (pos_dict is not None and len(pos_dict) == G.number_of_nodes()) else nx.spring_layout(G, seed=42, k=0.5)
    shop_n = getattr(data, "num_shop", 0)
    public_n = getattr(data, "num_public", data.num_nodes - shop_n)
    if values.numel() == public_n:
        node_vals = {i: float("nan") for i in range(shop_n)}
        for i, v in enumerate(values.tolist()):
            node_vals[shop_n + i] = v
    else:
        node_vals = {i: values[i].item() for i in range(data.num_nodes)}
    fig, ax = plt.subplots(figsize=(8, 6))
    if gdf is not None and pos_dict is not None and len(pos_dict) == G.number_of_nodes():
        gdf_valid = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
        if not gdf_valid.empty:
            try:
                gdf_valid.plot(ax=ax, facecolor="lightgray", edgecolor="gray", alpha=0.5, linewidth=0.5, aspect="equal")
            except ValueError:
                pass
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
    valid = [n for n in G.nodes() if not math.isnan(node_vals.get(n, float("nan")))]
    invalid = [n for n in G.nodes() if n not in valid]
    if invalid:
        nx.draw_networkx_nodes(G, pos, nodelist=invalid, node_color="lightgray", ax=ax, node_size=60)
    if valid:
        if vmin is None:
            vmin = min(node_vals[n] for n in valid)
        if vmax is None:
            vmax = max(node_vals[n] for n in valid)
        sc = nx.draw_networkx_nodes(G, pos, nodelist=valid, node_color=[node_vals[n] for n in valid],
                                    ax=ax, node_size=80, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label=value_label)
    ax.set_title(title)
    plt.axis("off")
    if pos_dict is not None and len(pos_dict) == G.number_of_nodes():
        ax.set_aspect("equal")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def _build_polygon_neighbors(gdf: Any) -> List[List[int]]:
    """Build adjacency from spatial relations (touches/intersects)."""
    try:
        import geopandas as gpd
    except ImportError:
        return [[] for _ in range(len(gdf))]
    n = len(gdf)
    neighbors = [[] for _ in range(n)]
    geoms = gdf.geometry
    for i in range(n):
        geom_i = geoms.iloc[i]
        if geom_i is None or (hasattr(geom_i, "is_empty") and geom_i.is_empty):
            continue
        try:
            for j in gdf.sindex.intersection(geom_i.bounds):
                if j == i:
                    continue
                geom_j = geoms.iloc[j]
                if geom_j is None or (hasattr(geom_j, "is_empty") and geom_j.is_empty):
                    continue
                if geom_i.touches(geom_j) or geom_i.intersects(geom_j):
                    neighbors[i].append(j)
        except Exception:
            pass
    return neighbors


def plot_heatmap_choropleth(
    gdf: Any,
    values: Any,
    title: str = "Flow heatmap",
    value_label: str = "yhat_flow",
    smooth_with_neighbors: bool = True,
    out_path: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """热力图：仅 public_space 图块有颜色，shop 为灰色；颜色受周边 public 节点影响呈渐变。
    gdf: GeoDataFrame，与 nodes.csv 顺序一致，需含 unit_type 列。
    values: 每块的值 [n]，可为 tensor 或 array。"""
    if hasattr(values, "cpu"):
        values = values.cpu().numpy()
    values = np.asarray(values, dtype=float)
    if len(values) != len(gdf):
        raise ValueError(f"values shape {len(values)} != gdf len {len(gdf)}")
    gdf = gdf.copy()
    public_mask = gdf["unit_type"] == "public_space" if "unit_type" in gdf.columns else np.ones(len(gdf), dtype=bool)
    n_public = int(public_mask.sum())
    if n_public == 0:
        public_mask = np.ones(len(gdf), dtype=bool)
    gdf["_value"] = np.nan
    if smooth_with_neighbors:
        neighbors = _build_polygon_neighbors(gdf)
        for i in range(len(values)):
            if not public_mask[i]:
                continue
            nbs = [j for j in neighbors[i] if public_mask[j]]
            vi = values[i] if not math.isnan(values[i]) else 0.0
            if not nbs:
                gdf.loc[gdf.index[i], "_value"] = vi
            else:
                nb_vals = [values[j] for j in nbs if not math.isnan(values[j])]
                smoothed = (vi + sum(nb_vals)) / (1.0 + len(nb_vals)) if nb_vals else vi
                gdf.loc[gdf.index[i], "_value"] = smoothed
    else:
        for i in np.where(public_mask)[0]:
            vi = values[i] if not math.isnan(values[i]) else np.nan
            gdf.loc[gdf.index[i], "_value"] = vi
    gdf_all = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    if gdf_all.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(title)
        return fig
    fig, ax = plt.subplots(figsize=(8, 6))
    if "unit_type" in gdf_all.columns:
        gdf_shop = gdf_all[gdf_all["unit_type"] == "shop"]
        gdf_public = gdf_all[gdf_all["unit_type"] == "public_space"]
    else:
        gdf_shop = gdf_all.iloc[0:0].copy()
        gdf_public = gdf_all
    if not gdf_shop.empty:
        gdf_shop.plot(ax=ax, facecolor="lightgray", edgecolor="gray", linewidth=0.5, alpha=0.7)
    if not gdf_public.empty:
        if vmin is None:
            vmin = float(gdf_public["_value"].min())
        if vmax is None:
            vmax = float(gdf_public["_value"].max())
        if math.isnan(vmin) or math.isnan(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        gdf_public.plot(ax=ax, column="_value", cmap="jet", legend=True, legend_kwds={"label": value_label},
                        edgecolor="gray", linewidth=0.5, alpha=0.9, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_edge_toggle_effect(
    node_deltas: List[float],
    node_ids: List[int],
    top_k: int = 10,
    title: str = "Top-k public node change (edge toggle)",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart: top-k public nodes by absolute prediction change."""
    pairs = list(zip(node_ids, node_deltas))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = pairs[:top_k]
    ids = [x[0] for x in top]
    deltas = [x[1] for x in top]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(ids)), deltas, color=["tab:green" if d >= 0 else "tab:red" for d in deltas])
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids)
    ax.set_xlabel("Public node_id")
    ax.set_ylabel("Prediction change (yhat_flow)")
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_graph_with_highlighted_edges(
    data: Data,
    edges_highlight: List[Tuple[int, int]],
    node_values: Optional[torch.Tensor] = None,
    title: str = "Graph with toggled edges highlighted",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """同构 Data：高亮边。edges_highlight: [(u, v), ...] 为全局节点索引。"""
    G = _data_to_nx(data)
    pos = nx.spring_layout(G, seed=42, k=0.5)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1)
    for u, v in edges_highlight:
        if G.has_edge(u, v):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, width=3, edge_color="red", style="--")
    nx.draw_networkx_nodes(G, pos, node_color="tab:blue", ax=ax, node_size=80)
    ax.set_title(title)
    plt.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
