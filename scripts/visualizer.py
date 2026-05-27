# scripts/visualizer.py
import io
import os
import colorsys
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import base64

# 경고 끄기
plt.rcParams["figure.max_open_warning"] = 100

DEFAULT_THEME = "#4e73df"


def _configure_matplotlib_fonts() -> None:
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    preferred = [
        "Apple SD Gothic Neo",
        "Nanum Gothic",
        "Pretendard",
        "Noto Sans KR",
        "Malgun Gothic",
        "DejaVu Sans",
    ]
    matched = [f for f in preferred if f in available]
    if not matched:
        matched = ["DejaVu Sans"]

    plt.rcParams["font.family"] = matched
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"


_configure_matplotlib_fonts()


def _normalize_hex(hex_color: str) -> str:
    if not hex_color:
        return DEFAULT_THEME
    color = hex_color.strip().lower()
    if color.startswith("#"):
        color = color[1:]
    if len(color) == 3:
        color = "".join([c * 2 for c in color])
    if len(color) != 6 or any(c not in "0123456789abcdef" for c in color):
        return DEFAULT_THEME
    return "#" + color


def _hex_to_rgb01(hex_color: str) -> tuple:
    hex_color = _normalize_hex(hex_color)[1:]
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b


def _rgb01_to_hex(r: float, g: float, b: float) -> str:
    r_i = int(max(0.0, min(1.0, r)) * 255)
    g_i = int(max(0.0, min(1.0, g)) * 255)
    b_i = int(max(0.0, min(1.0, b)) * 255)
    return "#{:02x}{:02x}{:02x}".format(r_i, g_i, b_i)


def _adjust_lightness(hex_color: str, delta: float) -> str:
    r, g, b = _hex_to_rgb01(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + delta))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb01_to_hex(r, g, b)


def complementary_hex(hex_color: str) -> str:
    """hue를 180도 회전한 보색 반환"""
    r, g, b = _hex_to_rgb01(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + 0.5) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb01_to_hex(r, g, b)


def _adjust_hls(hex_color: str, delta_l: float, delta_s: float) -> str:
    r, g, b = _hex_to_rgb01(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + delta_l))
    s = max(0.0, min(1.0, s + delta_s))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb01_to_hex(r, g, b)


def build_color_map(theme_color: str) -> Dict[str, Any]:
    base = _normalize_hex(theme_color)
    light = _adjust_lightness(base, 0.22)
    lighter = _adjust_lightness(base, 0.38)
    dark = _adjust_lightness(base, -0.18)
    darker = _adjust_lightness(base, -0.32)
    # 채도 낮은 테마에서도 색이 살아있도록 최소 채도 보장
    _, _, s0 = colorsys.rgb_to_hls(*_hex_to_rgb01(base))
    header = _adjust_hls(base, 0.20, max(-0.45, 0.15 - s0))
    highlight = _adjust_hls(base, 0.65, max(-0.40, 0.20 - s0))  # 형광펜용: 매우 밝고 채도 낮춤
    series = [base, dark, light, darker]
    return {
        "base": base,
        "light": light,
        "lighter": lighter,
        "dark": dark,
        "darker": darker,
        "header": header,
        "highlight": highlight,
        "series": series,
        "grid": "#9b9b9b",
        "text": "#111111",
        "muted": "#666666",
    }


def relative_luminance(hex_color: str) -> float:
    r, g, b = _hex_to_rgb01(hex_color)

    def to_lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_l, g_l, b_l = map(to_lin, (r, g, b))
    return 0.2126 * r_l + 0.7152 * g_l + 0.0722 * b_l


def is_dark_color(hex_color: str) -> bool:
    return relative_luminance(hex_color) < 0.5


_svg_counter = 0


def _fig_to_svg(fig) -> str:
    global _svg_counter
    _svg_counter += 1
    prefix = f"svg{_svg_counter}_"

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    idx = svg.find("<svg")
    if idx != -1:
        svg = svg[idx:]

    import re

    id_map: dict[str, str] = {}
    for m in re.finditer(r'\bid="([^"]+)"', svg):
        old_id = m.group(1)
        id_map[old_id] = prefix + old_id

    for old_id, new_id in id_map.items():
        svg = svg.replace(f'id="{old_id}"', f'id="{new_id}"')
        svg = svg.replace(f'href="#{old_id}"', f'href="#{new_id}"')
        svg = svg.replace(f'xlink:href="#{old_id}"', f'xlink:href="#{new_id}"')

    return svg


def _style_axes(ax, color_map: Dict[str, Any], grid_axis: Optional[str] = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")
    ax.tick_params(colors="#666666", labelsize=10)
    if grid_axis in ("x", "y", "both"):
        ax.grid(True, axis=grid_axis, color=color_map["grid"], linewidth=0.8)
    else:
        ax.grid(False)


def _value_colors(
    values: List[float],
    color_map: Dict[str, Any],
    palette: Optional[List[str]] = None,
):
    if not values:
        return [color_map["base"]]
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [color_map["base"] for _ in values]
    if palette:
        cmap = LinearSegmentedColormap.from_list("theme", palette, N=256).reversed()
    else:
        cmap = LinearSegmentedColormap.from_list(
            "theme",
            [color_map["lighter"], color_map["light"], color_map["base"], color_map["dark"]],
        )
    return [cmap((v - vmin) / (vmax - vmin)) for v in values]


def _contrast_text_color(rgba, threshold: float = 0.45) -> str:
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < threshold else "#1a1a1a"


def _extract_month_spans(labels: List[Any]) -> List[Dict[str, Any]]:
    parsed = pd.to_datetime(pd.Series(labels), errors="coerce")
    if parsed.isna().all():
        return []

    month_keys = parsed.dt.to_period("M")
    spans: List[Dict[str, Any]] = []
    current_month = None
    start_idx = None

    for idx, month_key in enumerate(month_keys):
        if pd.isna(month_key):
            if current_month is not None and start_idx is not None:
                spans.append({"month": current_month, "start": start_idx, "end": idx - 1})
            current_month = None
            start_idx = None
            continue

        if current_month is None:
            current_month = month_key
            start_idx = idx
            continue

        if month_key != current_month and start_idx is not None:
            spans.append({"month": current_month, "start": start_idx, "end": idx - 1})
            current_month = month_key
            start_idx = idx

    if current_month is not None and start_idx is not None:
        spans.append({"month": current_month, "start": start_idx, "end": len(month_keys) - 1})

    return spans


def _line_label_indices(values: List[Any]) -> List[int]:
    valid_points: List[tuple[int, float]] = []
    for idx, value in enumerate(values):
        if pd.isna(value):
            continue
        valid_points.append((idx, float(value)))

    if not valid_points:
        return []

    first_idx, first_value = valid_points[0]
    label_idxs: List[int] = [first_idx]
    ref_peak_idx = first_idx
    ref_peak_value = first_value
    prev_value = first_value
    decline_found = False

    for idx, value in valid_points[1:]:
        if not decline_found:
            if value >= ref_peak_value:
                ref_peak_value = value
                ref_peak_idx = idx
            if value < prev_value:
                decline_found = True
                if ref_peak_idx not in label_idxs:
                    label_idxs.append(ref_peak_idx)
        elif value > ref_peak_value:
            ref_peak_value = value
            label_idxs.append(idx)

        prev_value = value

    if not decline_found and ref_peak_idx not in label_idxs:
        label_idxs.append(ref_peak_idx)

    label_idxs.sort()
    return label_idxs

def render_line_chart(dataset: Dict[str, Any], color_map: Dict[str, Any], compact: bool = False) -> str:
    if not dataset:
        return ""

    labels = dataset.get("labels") or []
    series = dataset.get("series") or []

    if not labels or not series or len(labels) <= 1:
        return ""

    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8.0, 4.4))

    unit = str(dataset.get("unit") or "").strip()
    plotted_values: List[float] = []
    show_legend = bool(dataset.get("show_legend"))

    # 제목용 라벨 계산
    title_text = str(dataset.get("title") or "").strip()
    chart_label = "주별" if "주별" in title_text else "월별" if "월별" in title_text else None

    label_map = {
        "spend": "광고비",
        "revenue": "매출발생",
    }

    color_map_line = {
        "spend": "#1565C0",
        "revenue": "#C62828",
    }

    # 데이터 라인 그리기
    for idx, s in enumerate(series):
        raw_data = s.get("data") or []
        if not raw_data:
            continue

        data = pd.to_numeric(pd.Series(raw_data), errors="coerce").tolist()
        x_values = x[: len(data)]
        if not x_values:
            continue

        series_name = s.get("name") or f"series_{idx}"

        color = (
            color_map_line.get(series_name, color_map["series"][idx % len(color_map["series"])])
            if show_legend
            else color_map["series"][idx % len(color_map["series"])]
        )

        plot_label = label_map.get(series_name, series_name)

        ax.plot(
            x_values,
            data,
            color=color,
            linewidth=2,
            marker="o",
            markersize=3.8,
            label=plot_label if show_legend else None,
        )

        label_idx_set = set(_line_label_indices(data))
        last_labeled_x = None
        last_labeled_y = None

        valid_values = [float(v) for v in data if pd.notna(v)]
        series_y_min = min(valid_values)
        series_y_max = max(valid_values)
        series_y_span = max(series_y_max - series_y_min, 1)

        for point_idx, (x_val, y_val) in enumerate(zip(x_values, data)):
            if pd.isna(y_val):
                continue

            y_num = float(y_val)
            plotted_values.append(y_num)

            if x_val not in label_idx_set:
                continue

            # ===== 라벨 겹침 방지 =====
            if last_labeled_x is not None and last_labeled_y is not None:
                x_gap = abs(x_val - last_labeled_x)
                y_gap = abs(y_num - last_labeled_y)

                if x_gap <= 2 and y_gap < series_y_span * 0.12:
                    continue
            

            # 광고비/매출 첫 값 겹침 방지
            for idx, (x_val, y_num) in enumerate(zip(x_values, data)):
                if pd.isna(y_num):
                    continue

                y_num = float(y_num)
                plotted_values.append(y_num)
                if x_val not in label_idx_set:
                    continue

                # 기본값
                if show_legend:
                    if series_name == "spend":
                        base_offset = 6
                        va = "bottom"
                    elif series_name == "revenue":
                        base_offset = 6
                        va = "bottom"
                    else:
                        base_offset = 6
                        va = "bottom"
                else:
                    base_offset = 6
                    va = "bottom"

                # 첫 번째 값만 강제 위치
                if idx == 0:
                    if series_name == "spend":
                        base_offset = 6   # 광고비는 아래
                        va = "bottom"
                    elif series_name == "revenue":
                        base_offset = 17    # 매출은 위
                        va = "bottom"

                xytext = (0, base_offset)
                ha = "center"

                ax.annotate(
                    f"{_format_chart_value(y_num)}{unit}" if unit else _format_chart_value(y_num),
                    (x_val, y_num),
                    textcoords="offset points",
                    xytext=xytext,
                    ha=ha,
                    va=va,
                    fontsize=8,
                    color=color,
                    clip_on=False,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.85
                    ),
                )

                last_labeled_x = x_val
                last_labeled_y = y_num
            
    if not plotted_values:
        return ""

    # Y축 범위 설정
    y_min, y_max = min(plotted_values), max(plotted_values)
    y_span = y_max - y_min
    y_pad = max(y_span * 0.22, 0.3)

    y_low = 0 if unit == "%" else y_min - y_pad * 0.2
    y_high = y_max + y_pad

    if abs(y_high - y_low) < 1e-12:
        y_high = y_low + 1.0

    ax.set_ylim(y_low, y_high)

    # 월 밴드
    month_spans = _extract_month_spans(labels)

    if month_spans:
        ax.set_xlim(-0.55, len(labels) - 0.5)

        month_band_ymin, month_band_ymax = -0.06, 0.0

        for span_idx, span in enumerate(month_spans):
            start, end, period = span["start"], span["end"], span["month"]

            ax.add_patch(
                Rectangle(
                    (start - 0.5, month_band_ymin),
                    (end - start) + 1.0,
                    month_band_ymax - month_band_ymin,
                    transform=ax.get_xaxis_transform(),
                    facecolor="#f5f5f5" if span_idx % 2 == 0 else "#fafafa",
                    edgecolor="none",
                    zorder=0,
                    clip_on=False,
                )
            )

            ax.text(
                (start + end) / 2,
                (month_band_ymin + month_band_ymax) / 2,
                f"{period.year}.{period.month:02d}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=7.5,
                color="#7a7a7a",
                zorder=1,
                clip_on=False,
            )

        for span in month_spans[1:]:
            ax.axvline(
                span["start"] - 0.5,
                color="#d9d9d9",
                linestyle=(0, (2, 3)),
                linewidth=0.9,
                zorder=1.5,
            )

    # 스타일
    ax.set_xticks([])

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: f"{int(round(v)):,}" if abs(v - round(v)) < 1e-9 else f"{v:,.2f}"
        )
    )

    _style_axes(ax, color_map, grid_axis=None)
    ax.tick_params(axis="x", length=0, labelbottom=False)

    # 제목
    if chart_label:
        ax.set_title(
            chart_label,
            fontsize=17,
            color="#2E2E2E",
            fontweight="bold",
            pad=0,
            y=1.08
        )

    # 범례
    if show_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=len(series),
            frameon=False,
            fontsize=9,
        )

    # 위치 조정
    return _fig_to_svg(fig)

def render_bar_h_chart(
    dataset: Dict[str, Any],
    color_map: Dict[str, Any],
    compact: bool = False,
    chart_width: float = None,
    chart_height: float = None,
    palette: Optional[List[str]] = None,
) -> str:
    if not dataset:
        return ""
    labels = dataset.get("labels") or []
    series = dataset.get("series") or []
    if not labels or not series:
        return ""

    values = series[0].get("data") or []
    if not values:
        return ""

    labels = labels[: len(values)]
    y = list(range(len(labels)))
    if compact:
        width, height = 3.2, 1.8
    else:
        # Match example.py barh ratio (4:6) so charts are quarter-width and vertically long.
        width = chart_width if isinstance(chart_width, (int, float)) else 4.4
        height = chart_height if isinstance(chart_height, (int, float)) else 6.2
    fig, ax = plt.subplots(figsize=(width, height))

    colors = _value_colors(values, color_map, palette=palette)
    ax.barh(y, values, color=colors)
    ax.invert_yaxis()

    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5)
        unit = dataset.get("unit", "")
        if unit:
            ax.set_xlabel(unit, fontsize=9.5, color=color_map["muted"])
        _style_axes(ax, color_map, grid_axis=None)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=0.6)
    return _fig_to_svg(fig)

def _format_chart_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def render_bar_v_chart(
    dataset: Dict[str, Any],
    color_map: Dict[str, Any],
    compact: bool = False,
    show_labels: bool = False,
    show_values: bool = False,
) -> str:
    if not dataset:
        return ""
    labels = dataset.get("labels") or []
    series = dataset.get("series") or []
    if not labels or not series:
        return ""

    values = pd.Series(pd.to_numeric(series[0].get("data") or [], errors="coerce")).fillna(0.0).tolist()
    if not values:
        return ""

    labels = labels[: len(values)]
    x = list(range(len(labels)))
    if compact and (show_labels or show_values):
        fig_size = (3.4, 2.7)
    else:
        fig_size = (6.8, 3.8) if not compact else (3.4, 2.0)
    fig, ax = plt.subplots(figsize=fig_size)

    colors = _value_colors(values, color_map)
    bars = ax.bar(x, values, color=colors)

    max_val = max(values) if values else 0.0
    y_pad = max(max_val * 0.2, 0.4)
    y_top = max(max_val + y_pad, 1.0)
    ax.set_ylim(0, y_top)

    if show_values:
        unit = str(dataset.get("unit") or "").strip()
        suffix = unit if unit in {"%", "회", "명"} else ""
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_pad * 0.08,
                f"{_format_chart_value(float(value))}{suffix}",
                ha="center",
                va="bottom",
                fontsize=7 if compact else 8.5,
                color=color_map["muted"],
            )

    if compact:
        if show_labels:
            display_labels = [str(label).replace("<br>", "\n") for label in labels]
            ax.set_xticks(x)
            ax.set_xticklabels(display_labels, fontsize=6.5, rotation=0, ha="center")
            ax.tick_params(axis="x", length=0, pad=1, colors="#666666")
        else:
            ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5, rotation=30, ha="right")
        unit = dataset.get("unit", "")
        if unit:
            ax.set_ylabel(unit, fontsize=9.5, color=color_map["muted"])
        _style_axes(ax, color_map, grid_axis=None)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=0.6)
    return _fig_to_svg(fig)


def _render_heatmap(rows: List[Dict[str, Any]], metric: str, color_map: Dict[str, Any]) -> str:
    df = pd.DataFrame(rows)
    if df.empty or metric not in df.columns:
        return ""
    if "age" not in df.columns or "gender" not in df.columns:
        return ""

    pivot = df.pivot_table(index="gender", columns="age", values=metric, aggfunc="mean")
    imp_pivot = None
    if metric == "ctr" and "impressions" in df.columns:
        imp_pivot = df.pivot_table(index="gender", columns="age", values="impressions", aggfunc="mean")

    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    gender_order = ["female", "male"]
    pivot = pivot.reindex(
        index=[g for g in gender_order if g in pivot.index],
        columns=[a for a in age_order if a in pivot.columns],
    )
    if imp_pivot is not None:
        imp_pivot = imp_pivot.reindex(index=pivot.index, columns=pivot.columns)
    if pivot.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    cmap = LinearSegmentedColormap.from_list(
        "theme",
        [color_map["lighter"], color_map["light"], color_map["base"], color_map["dark"]],
    )
    heat_values = pivot.values.astype(float)
    vmin = float(np.nanmin(heat_values))
    vmax = float(np.nanmax(heat_values))
    im = ax.imshow(heat_values, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=10, colors="#666666")
    if metric == "impressions":
        cbar.formatter = FuncFormatter(lambda x, _: f"{int(round(float(x))):,}")
        cbar.update_ticks()

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                continue
            norm = 0.5 if abs(vmax - vmin) < 1e-12 else (float(val) - vmin) / (vmax - vmin)
            cell_color = cmap(norm)
            if metric == "impressions":
                label = f"{int(round(float(val))):,}"
            else:
                label = f"{float(val):.2f}"
                if imp_pivot is not None:
                    imp_val = imp_pivot.iloc[i, j]
                    if pd.notna(imp_val):
                        label += f"\n({int(round(float(imp_val))):,})"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=11,
                color=_contrast_text_color(cell_color, threshold=0.45),
            )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(c) for c in pivot.index], fontsize=11)
    ax.tick_params(axis="x", bottom=True, top=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.6)
    return _fig_to_svg(fig)


def _render_simple_table(rows: List[Dict[str, Any]]) -> str:
    df = pd.DataFrame(rows)
    if df.empty:
        return ""

    max_rows = 12
    if len(df) > max_rows:
        df = df.head(max_rows)

    fig_height = 0.35 * len(df) + 1.2
    fig, ax = plt.subplots(figsize=(6, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.2)

    fig.tight_layout(pad=0.6)
    return _fig_to_svg(fig)


def render_table_chart(dataset: Dict[str, Any], color_map: Dict[str, Any], metric: str = None) -> str:
    if not dataset:
        return ""
    rows = dataset.get("rows") or []
    if not rows:
        return ""

    if metric == "purchases":
        heatmap_svg = _render_purchase_conversion_heatmap(rows, color_map)
        if heatmap_svg:
            return heatmap_svg

    if metric:
        heatmap_svg = _render_heatmap(rows, metric, color_map)
        if heatmap_svg:
            return heatmap_svg

    if "age" in rows[0] and "gender" in rows[0] and "ctr" in rows[0]:
        heatmap_svg = _render_heatmap(rows, "ctr", color_map)
        if heatmap_svg:
            return heatmap_svg

    return _render_simple_table(rows)


def render_content_card(dataset: Dict[str, Any], color_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not dataset:
        return []

    items = dataset.get("items") or []
    rendered = []

    for item in items:
        new_item = dict(item)
        details = item.get("target_details") or []
        chart_svg = ""

        if details:
            detail_df = pd.DataFrame(details)
            if (
                not detail_df.empty
                and {"age", "gender", "ctr"}.issubset(detail_df.columns)
            ):
                detail_df["gender"] = detail_df["gender"].astype(str).str.strip()
                detail_df = detail_df[detail_df["gender"].str.lower() != "unknown"]
                detail_df["ctr"] = pd.to_numeric(detail_df["ctr"], errors="coerce")
                detail_df = detail_df.dropna(subset=["ctr"])
                detail_df = detail_df[detail_df["ctr"] > 0]
                if detail_df.empty:
                    new_item["chart"] = chart_svg
                    rendered.append(new_item)
                    continue

                detail_df = detail_df.sort_values("ctr", ascending=False).head(6)
                labels = []
                for _, row in detail_df.iterrows():
                    age_text = str(row["age"]).strip()
                    gender_text = str(row["gender"]).strip()
                    gender_low = gender_text.lower()
                    if gender_low == "female":
                        gender_text = "여성"
                    elif gender_low == "male":
                        gender_text = "남성"
                    labels.append(f"{age_text}<br>{gender_text}")
                values = detail_df["ctr"].tolist()
                mini_ds = {
                    "kind": "bar_v",
                    "labels": labels,
                    "series": [{"name": "ctr", "data": values}],
                    "unit": "%",
                }
                chart_svg = render_bar_v_chart(
                    mini_ds,
                    color_map,
                    compact=True,
                    show_labels=True,
                    show_values=True,
                )

        new_item["chart"] = chart_svg
        rendered.append(new_item)

    return rendered


def render_dataset(dataset: Dict[str, Any], color_map: Dict[str, Any], **kwargs):
    if not dataset:
        return ""

    kind = dataset.get("kind")
    renderers = {
        "line": render_line_chart,
        "bar_h": render_bar_h_chart,
        "bar_v": render_bar_v_chart,
        "bubble": render_bubble_chart,
        "table": render_table_chart,
        "content_card": render_content_card,
    }

    renderer = renderers.get(kind)
    if not renderer:
        return ""

    return renderer(dataset, color_map, **kwargs)

def render_bubble_chart(
    dataset: Dict[str, Any],
    color_map: Dict[str, Any],
    compact: bool = True,
    palette: Optional[List[str]] = None,
) -> str:
    labels = dataset.get("labels") or []
    series = dataset.get("series") or []
    if not labels or not series:
        return ""

    ctr_raw = pd.to_numeric(series[0].get("data") or [], errors="coerce")
    size_raw = ctr_raw
    if len(series) > 1:
        size_raw = pd.to_numeric(series[1].get("data") or [], errors="coerce")

    n = min(len(labels), len(ctr_raw), len(size_raw))
    if n == 0:
        return ""

    rows = []
    for i in range(n):
        ctr = ctr_raw[i]
        size_val = size_raw[i]
        if pd.isna(ctr) or pd.isna(size_val):
            continue
        rows.append({
            "label": str(labels[i]),
            "ctr": float(ctr),
            "size": max(float(size_val), 1.0),
        })
    if not rows:
        return ""

    rows.sort(key=lambda x: x["size"], reverse=True)

    size_values = np.sqrt(np.array([r["size"] for r in rows], dtype=float))
    size_norm = size_values / size_values.max() if size_values.max() > 0 else np.ones(len(rows))
    max_r = 0.58 if len(rows) <= 3 else (0.45 if len(rows) <= 6 else 0.36)
    min_r = max(0.12, max_r * 0.38)
    radii = (min_r + (max_r - min_r) * size_norm).tolist()

    positions = [np.array([0.0, 0.0])]
    if len(radii) > 1:
        positions.append(np.array([radii[0] + radii[1], 0.0]))

    for i in range(2, len(radii)):
        r_new = radii[i]
        placed = False
        for j in range(len(positions)):
            for k in range(j + 1, len(positions)):
                r1, r2 = radii[j], radii[k]
                p1, p2 = positions[j], positions[k]
                dist = np.linalg.norm(p1 - p2)
                if dist <= 1e-9:
                    continue
                if dist > (r1 + r_new) + (r2 + r_new):
                    continue

                d1 = r1 + r_new
                d2 = r2 + r_new
                a = (d1**2 - d2**2 + dist**2) / (2 * dist)
                h = np.sqrt(max(0.0, d1**2 - a**2))
                p3 = p1 + a * (p2 - p1) / dist

                for sign in (-1, 1):
                    test_pos = np.array([
                        p3[0] + sign * h * (p2[1] - p1[1]) / dist,
                        p3[1] - sign * h * (p2[0] - p1[0]) / dist,
                    ])
                    if all(
                        np.linalg.norm(test_pos - p) >= (radii[idx] + r_new) * 0.99
                        for idx, p in enumerate(positions)
                    ):
                        positions.append(test_pos)
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            positions.append(np.array([radii[0] + r_new, r_new * i]))

    if palette:
        bubble_palette = palette
    else:
        bubble_palette = [
            color_map["lighter"],
            color_map["light"],
            color_map["base"],
            color_map["dark"],
        ]
    cmap = LinearSegmentedColormap.from_list("bubble_palette", bubble_palette, N=256).reversed()

    ctr_values = [r["ctr"] for r in rows]
    vmin, vmax = min(ctr_values), max(ctr_values)

    fig_size = (4, 4) if compact else (5.4, 5.4)
    fig, ax = plt.subplots(figsize=fig_size)

    for idx, pos in enumerate(positions):
        ctr = rows[idx]["ctr"]
        label = rows[idx]["label"]
        radius = radii[idx]
        norm = 0.5 if abs(vmax - vmin) < 1e-12 else (ctr - vmin) / (vmax - vmin)
        color = cmap(norm)
        circle = plt.Circle(pos, radius, facecolor=color, edgecolor="white", linewidth=1.6, alpha=1.0)
        ax.add_patch(circle)

        font_size = max(7, min(13, 5 + radius * 11))
        ax.text(
            pos[0],
            pos[1],
            f"{label}\n({ctr:.2f}%)",
            fontsize=font_size,
            ha="center",
            va="center",
            color=_contrast_text_color(color, threshold=0.45),
            fontweight="bold",
        )

    all_points = np.array(
        [p + np.array([r, r]) for p, r in zip(positions, radii)] +
        [p - np.array([r, r]) for p, r in zip(positions, radii)]
    )
    x_min, x_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    y_min, y_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    x_pad = max((x_max - x_min) * 0.05, 0.08)
    y_pad = max((y_max - y_min) * 0.05, 0.08)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal")
    ax.axis("off")

    return _fig_to_svg(fig)


# 구매 콘텐츠 파이차트
def render_purchase_pie_chart(rows: List[Dict[str, Any]], color_map: Dict[str, Any]) -> str:
    df = pd.DataFrame(rows)
    if df.empty or "purchases" not in df.columns:
        return ""

    df = df.copy()
    df["purchases"] = pd.to_numeric(df["purchases"], errors="coerce").fillna(0)
    df = df[df["purchases"] > 0]
    if df.empty:
        return ""

    def _gender_label(g):
        g = str(g).strip().lower()
        if g == "female":
            return "여성"
        if g == "male":
            return "남성"
        return str(g)

    labels = [
        f"{str(row['age']).strip()} {_gender_label(row['gender'])}"
        for _, row in df.iterrows()
    ]
    values = df["purchases"].tolist()
    total = int(df["purchases"].sum())
    percentages = [(v / total) * 100 for v in values]

    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    colors = _value_colors(
        values,
        color_map,
        palette=[
            color_map["dark"],
            color_map["base"],
            color_map["light"],
            color_map["lighter"],
        ],
    )

    pie_radius = 1.21

    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=pie_radius,
        wedgeprops=dict(edgecolor="none")
    )

    label_items = []

    for i, w in enumerate(wedges):
        angle = (w.theta2 + w.theta1) / 2.0
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))

        label_items.append({
            "idx": i,
            "x": x,
            "y": y,
            "line_start": (x * pie_radius * 0.96, y * pie_radius * 0.96),
            "label_x": 1.34 if x >= 0 else -1.34,
            "label_y": y * 1.08,
            "ha": "left" if x >= 0 else "right",
            "side": "right" if x >= 0 else "left",
        })

    # 한쪽에 라벨이 너무 몰리면 구매건수가 작은 조각부터 반대편으로 분산
    left_items = [d for d in label_items if d["side"] == "left"]
    right_items = [d for d in label_items if d["side"] == "right"]

    if len(left_items) - len(right_items) >= 2:
        move_count = (len(left_items) - len(right_items)) // 2
        left_items_sorted = sorted(left_items, key=lambda d: values[d["idx"]])  
        for item in left_items_sorted[:move_count]:
            item["side"] = "right"
            item["label_x"] = 1.34
            item["ha"] = "left"

    elif len(right_items) - len(left_items) >= 2:
        move_count = (len(right_items) - len(left_items)) // 2
        right_items_sorted = sorted(right_items, key=lambda d: values[d["idx"]]) 
        for item in right_items_sorted[:move_count]:
            item["side"] = "left"
            item["label_x"] = -1.34
            item["ha"] = "right"

    # 같은 쪽 라벨끼리 간격 벌려서 겹침 방지
    def _adjust(items, min_gap=0.32):
        items = sorted(items, key=lambda d: d["label_y"], reverse=True)
        for j in range(1, len(items)):
            upper = items[j - 1]
            cur = items[j]
            if upper["label_y"] - cur["label_y"] < min_gap:
                cur["label_y"] = upper["label_y"] - min_gap
        return items

    left_items = _adjust([d for d in label_items if d["side"] == "left"], min_gap=0.32)
    right_items = _adjust([d for d in label_items if d["side"] == "right"], min_gap=0.32)

    adjusted = {d["idx"]: d for d in left_items + right_items}

    for i in range(len(labels)):
        item = adjusted[i]
        label_x = item["label_x"]
        label_y = item["label_y"]
        ha = item["ha"]

        ax.annotate(
            "",
            xy=item["line_start"],
            xytext=(label_x, label_y),
            arrowprops=dict(
                arrowstyle="-",
                color="#8c8c8c",
                lw=0.8,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="angle3,angleA=0,angleB=90"
            )
        )

        ax.text(
            label_x,
            label_y + 0.04,
            labels[i],
            ha=ha,
            va="bottom",
            fontsize=11,
            color="#555555"
        )

        ax.text(
            label_x,
            label_y - 0.01,
            f"{percentages[i]:.1f}%",
            ha=ha,
            va="top",
            fontsize=10,
            color="#9a9a9a"
        )

    # 차트 크기 일정하게 보이도록 축 범위 고정
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.45, 1.45)

    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.08)
    return _fig_to_svg(fig)

# -----------------------------------
# 팔로워 인구통계 추가

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=350, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

# 팔로워 연령대별 누적 가로막대 차트
def render_follower_age_gender_stacked_barh_chart(chart_data, color_map):
    labels = chart_data.get("labels", [])
    series = chart_data.get("series", [])

    if not labels or not series:
        return ""

    # 시리즈 이름/데이터 정리
    parsed = []
    for s in series:
        name = str(s.get("name", "")).strip().lower()
        data = pd.to_numeric(pd.Series(s.get("data", [])), errors="coerce").fillna(0).tolist()

        if name in ["male", "남성"]:
            parsed.append(("남성", data, color_map["base"], "white"))
        elif name in ["female", "여성"]:
            parsed.append(("여성", data, color_map["lighter"], "#252525"))
        elif name in ["unknown", "알 수 없음"]:
            parsed.append(("알 수 없음", data, color_map["dark"], "white"))
        elif name in ["known", "남/여 전체"]:
            parsed.append(("남/여 전체", data, color_map["lighter"], "#252525"))

    if not parsed:
        return ""

    # 길이 맞추기
    n = len(labels)
    normalized = []
    for display_name, data, color, text_color in parsed:
        if len(data) < n:
            data = data + [0] * (n - len(data))
        elif len(data) > n:
            data = data[:n]
        normalized.append((display_name, data, color, text_color))

    df = pd.DataFrame({"age_range": labels})
    for i, (display_name, data, color, text_color) in enumerate(normalized):
        df[f"v{i}"] = data

    value_cols = [f"v{i}" for i in range(len(normalized))]
    df["total"] = df[value_cols].sum(axis=1)
    df = df[df["total"] > 0]
    if df.empty:
        return ""

    age_order = ["13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["age_order"] = df["age_range"].apply(lambda x: age_order.index(x) if x in age_order else 99)
    df = df.sort_values("age_order", ascending=False)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.5, 5))

    left = np.zeros(len(df))
    for i, (display_name, data, color, text_color) in enumerate(normalized):
        vals = df[f"v{i}"].tolist()

        ax.barh(
            y, vals, left=left,
            color=color, edgecolor="none", height=0.89,
            label=display_name
        )

        for j, v in enumerate(vals):
            if v >= max(df["total"]) * 0.08:   # 너무 작은 값은 라벨 생략
                ax.text(
                    left[j] + v / 2, j, f"{int(v):,}",
                    ha="center", va="center",
                    fontsize=13, color=text_color, fontweight="bold"
                )
        left += np.array(vals)

    ax.set_yticks(y)
    ax.set_yticklabels(df["age_range"].astype(str).tolist(), fontsize=14)

    max_total = df["total"].max()
    ax.set_xlim(0, max_total * 1.12)

    ax.tick_params(axis="x", labelsize=13, colors="#666666")
    ax.grid(axis="x", linestyle="--", alpha=0.18)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.12)
    return fig_to_base64(fig)

#--------------------------------
# 팔로워 성별 도넛 차트
def render_follower_gender_doughnut_chart(chart_data, color_map):
    labels = chart_data.get("labels", [])
    series = chart_data.get("series", [])

    if not labels or not series:
        return ""

    values = series[0].get("data", [])
    if not values:
        return ""

    values = [float(v) for v in values]

    def _pick_color(label: str) -> str:
        label = str(label).strip()
        if label == "여성":
            return color_map["lighter"]
        elif label == "남성":
            return color_map["base"]
        elif label in ["남/여 전체", "연령 확인 가능", "확인 가능", "Known"]:
            return color_map["lighter"]
        elif label in ["알 수 없음", "Unknown", "unknown"]:
            return color_map["dark"]
        return color_map["dark"]

    colors = [_pick_color(label) for label in labels]

    fig, ax = plt.subplots(figsize=(5.4, 4.8))

    fig.suptitle(
        chart_data.get("title", ""),
        y=0.94,
        fontsize=14,
        fontweight="bold"
    )

    def autopct_func(pct):
        return f"{pct:.0f}%" if pct >= 5 else ""

    wedges, texts, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=0.98,
        wedgeprops=dict(width=0.48, edgecolor="white"),
        autopct=autopct_func,
        pctdistance=0.74
    )

    for i, t in enumerate(autotexts):
        if colors[i] == color_map["lighter"]:
            t.set_color("#252525")
        else:
            t.set_color("white")

        t.set_fontsize(14)
        t.set_fontweight("bold")

    total = sum(values)

    center_text = chart_data.get("center_text")
    center_subtext = chart_data.get("center_subtext")

    if center_text is not None:
        center_main = str(center_text)
        center_sub = str(center_subtext or "")
    else:
        is_ratio_chart = 99.5 <= total <= 100.5
        center_main = "100%" if is_ratio_chart else f"{int(total):,}"
        center_sub = "비율" if is_ratio_chart else "팔로워"

    ax.text(
        0, 0.05, center_main,
        ha="center", va="center",
        fontsize=17, fontweight="bold"
    )
    ax.text(
        0, -0.16, center_sub,
        ha="center", va="center",
        fontsize=13, color="#666"
    )

    ax.legend(
        wedges, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=11,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0
    )

    ax.set(aspect="equal")
    ax.axis("off")

    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.05, right=0.95)
    return fig_to_base64(fig)

# 구매전환 히트맵
def _render_purchase_conversion_heatmap(
    rows: List[Dict[str, Any]],
    color_map: Dict[str, Any],
) -> str:
    df = pd.DataFrame(rows)
    if df.empty or "purchases" not in df.columns:
        return ""
    if "age" not in df.columns or "gender" not in df.columns:
        return ""

    pivot = df.pivot_table(index="gender", columns="age", values="purchases", aggfunc="sum")

    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    gender_order = ["female", "male"]
    pivot = pivot.reindex(
        index=[g for g in gender_order if g in pivot.index],
        columns=[a for a in age_order if a in pivot.columns],
    )

    if pivot.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    cmap = LinearSegmentedColormap.from_list(
        "theme",
        [color_map["lighter"], color_map["light"], color_map["base"], color_map["dark"]],
    )

    heat_values = pivot.values.astype(float)
    vmin = float(np.nanmin(heat_values))
    vmax = float(np.nanmax(heat_values))

    im = ax.imshow(heat_values, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=10, colors="#666666")
    cbar.formatter = FuncFormatter(lambda x, _: f"{int(round(float(x))):,}")
    cbar.update_ticks()

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                continue

            norm = 0.5 if abs(vmax - vmin) < 1e-12 else (float(val) - vmin) / (vmax - vmin)
            cell_color = cmap(norm)

            ax.text(
                j,
                i,
                f"{int(round(float(val))):,}\n ",
                ha="center",
                va="center",
                fontsize=11,
                linespacing=1.2,
                color=_contrast_text_color(cell_color, threshold=0.45),
            )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(c) for c in pivot.index], fontsize=11)
    ax.tick_params(axis="x", bottom=True, top=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.6)
    return _fig_to_svg(fig)