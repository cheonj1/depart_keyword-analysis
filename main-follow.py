"""
게시물(콘텐츠) 단위 "도달/팔로우" CSV + DB의 키워드 태깅(ad_keywords)을 결합해
팔로우 발생 기준 키워드 분석 리포트를 만드는 스크립트.

CSV: Meta 인사이트에서 내보낸 게시물 단위, 현재 시점 기준 누적 성과.
    필수 컬럼 — 게시물 ID / 도달 / 팔로우
DB: CSV의 "게시물 ID"(ig_contents.fb_ig_media_id 포맷)에 연결된 광고의
    essential_keywords/variable_keywords만 가져와 붙인다 (target_id로 계정 범위 지정,
    scripts.processor.get_content_keywords_by_account). 같은 게시물을 여러 광고가
    재사용했다면 그 광고들의 키워드를 모두 합쳐(중복 제거) 사용한다.

main.py와의 차이
    - 성과 지표: 광고 노출/클릭(CTR) 대신 게시물 도달/팔로우 발생수. 광고 성과 테이블
      (ad_performance_daily 등)은 전혀 쓰지 않는다.
    - 성별·연령 브레이크다운 없음 (CSV에 그런 구분이 없어 "전체"만 존재 — main/avoid 타겟
      섹션 자체가 없다)
    - 출력 PDF: 키워드 상하위 + 키워드 조합 분석 페이지만 표시 (template_keywords_follow.html,
      main_csv.py와 동일한 "키워드 전용" 렌더링 방식)
    - min 기준(키워드/조합 모두 등장 게시물 수 >= 1)이 사실상 필터링 의미가 없으므로
      "N개 이상 콘텐츠 등장" 문구를 표에서 제거
    - 키워드 상하위 페이지: 표 없이 차트 4개(명사/형용사 × 도달 1,000당 팔로우 발생 수 기준,
      팔로우 발생 수 기준)로 구성. 각 바 내부 왼쪽 끝에 수치를 직접 표시한다
      (도달 1,000당 차트: "0.02 (1명)", 발생 수 차트: "1명"). 콤보 카드도 동일한 기준(단위 % 없음).
      페이지 하단에 계산식 footnote를 표시한다.
"""

import itertools
import os
import time

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from scripts.processor import (
    get_account_name, get_content_keywords_by_account,
    filter_keywords_by_pos, _normalize_keyword_by_pos,
)
from scripts.visualizer import build_color_map, complementary_hex, is_dark_color, render_dataset
from main import _apply_display_predicate_suffix, _combo_cards, export_to_pdf


def _generate_keyword_html(context, output_path):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("template_keywords_follow.html")
    output = template.render(context)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    return os.path.abspath(output_path)


# ============================================================
# CSV 로딩 + DB 키워드 매핑
# ============================================================

_REQUIRED_COLUMNS = {"게시물 ID", "도달", "팔로우"}


def _load_perf_csv(path):
    df = pd.read_csv(path, dtype={"게시물 ID": str})
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {sorted(missing)}")

    df = df.rename(columns={"게시물 ID": "post_id", "도달": "reach", "팔로우": "follows"})
    df["post_id"] = df["post_id"].astype(str).str.strip()
    df["reach"] = pd.to_numeric(df["reach"], errors="coerce").fillna(0)
    df["follows"] = pd.to_numeric(df["follows"], errors="coerce").fillna(0)
    return df


def _attach_keywords(perf_df, target_id):
    kw_df = get_content_keywords_by_account(target_id)
    kw_df["post_id"] = kw_df["post_id"].astype(str).str.strip()

    merged = perf_df.merge(kw_df, on="post_id", how="left")

    unmatched = merged[merged["essential_keywords"].isna()]
    if not unmatched.empty:
        sample = unmatched["post_id"].head(5).tolist()
        print(f"⚠️  키워드 매핑을 찾지 못한 게시물 {len(unmatched)}개는 키워드 분석에서 제외됩니다 (예시: {sample})")

    merged = merged.dropna(subset=["essential_keywords"]).copy()
    merged["essential_keywords"] = merged["essential_keywords"].apply(lambda v: v if isinstance(v, list) else [])
    merged["variable_keywords"] = merged["variable_keywords"].apply(lambda v: v if isinstance(v, list) else [])
    return merged


# ============================================================
# 키워드 성과 (게시물 단위 도달/팔로우 합산, main_csv.py의 raw_keyword_performance_from_df와 동일한 구조)
# ============================================================

_RAW_KW_COLUMNS = ["keyword", "doc_freq", "total_impressions", "total_clicks", "avg_ctr", "avg_follows_per_content"]


def raw_keyword_performance_from_df(df, min_doc_freq=1):
    if df is None or df.empty:
        return pd.DataFrame(columns=_RAW_KW_COLUMNS)

    rows = []
    for r in df.itertuples(index=False):
        keywords = set(r.essential_keywords) | set(r.variable_keywords)
        for kw in keywords:
            if kw:
                rows.append((r.post_id, kw, r.reach, r.follows))
    if not rows:
        return pd.DataFrame(columns=_RAW_KW_COLUMNS)

    kdf = pd.DataFrame(rows, columns=["post_id", "keyword", "reach", "follows"])
    agg = kdf.groupby("keyword", as_index=False).agg(
        doc_freq=("post_id", "nunique"),
        total_impressions=("reach", "sum"),
        total_clicks=("follows", "sum"),
    )
    agg = agg[agg["doc_freq"] >= min_doc_freq].copy()
    # 도달 1,000당 팔로우 발생 수 = 총 팔로우 발생 / 총 도달 * 1,000
    agg["avg_ctr"] = np.where(
        agg["total_impressions"] > 0,
        np.round(agg["total_clicks"] / agg["total_impressions"] * 1000, 2),
        np.nan,
    )
    # 키워드 포함 콘텐츠의 평균 팔로우 발생 수 = 총 팔로우 발생 / 등장 콘텐츠 수(doc_freq)
    agg["avg_follows_per_content"] = np.where(
        agg["doc_freq"] > 0,
        np.round(agg["total_clicks"] / agg["doc_freq"], 2),
        np.nan,
    )
    return agg[_RAW_KW_COLUMNS]


# ============================================================
# 키워드 조합 성과 (main_csv.py의 strategic_performance_from_df와 동일한 구조)
# ============================================================

_STRAT_COLUMNS = ["ess_1", "ess_2", "combo_doc_freq", "combo_overall_ctr", "var_keyword", "with_var_ctr", "var_imps"]


def strategic_performance_from_df(df, min_doc_freq=1):
    if df is None or df.empty:
        return pd.DataFrame(columns=_STRAT_COLUMNS)

    base = df[df["essential_keywords"].apply(lambda ks: len(set(k for k in ks if k)) >= 2)]
    if base.empty:
        return pd.DataFrame(columns=_STRAT_COLUMNS)

    pair_rows = []
    for r in base.itertuples(index=False):
        ess_unique = sorted(set(k for k in r.essential_keywords if k))
        var_unique = sorted(set(k for k in r.variable_keywords if k))
        for e1, e2 in itertools.combinations(ess_unique, 2):
            pair_rows.append((r.post_id, e1, e2, r.reach, r.follows, var_unique))
    if not pair_rows:
        return pd.DataFrame(columns=_STRAT_COLUMNS)
    pairs_df = pd.DataFrame(pair_rows, columns=["post_id", "ess_1", "ess_2", "reach", "follows", "variable_keywords"])

    essential_agg = pairs_df.groupby(["ess_1", "ess_2"], as_index=False).agg(
        combo_doc_freq=("post_id", "nunique"),
        total_reach=("reach", "sum"),
        total_follows=("follows", "sum"),
    )
    essential_agg = essential_agg[essential_agg["combo_doc_freq"] >= min_doc_freq]
    # 도달 1000당 팔로우 발생 수 = 총 팔로우 발생 / 총 도달 * 1000
    essential_agg["combo_overall_ctr"] = np.where(
        essential_agg["total_reach"] > 0,
        np.round(essential_agg["total_follows"] / essential_agg["total_reach"] * 1000, 2),
        np.nan,
    )
    essential_agg = essential_agg.dropna(subset=["combo_overall_ctr"])
    if essential_agg.empty:
        return pd.DataFrame(columns=_STRAT_COLUMNS)

    top_essential = essential_agg.sort_values("combo_overall_ctr", ascending=False).head(15)
    top_keys = set(zip(top_essential["ess_1"], top_essential["ess_2"]))

    var_rows = []
    for row in pairs_df.itertuples(index=False):
        if (row.ess_1, row.ess_2) not in top_keys:
            continue
        for vk in row.variable_keywords:
            var_rows.append((row.ess_1, row.ess_2, vk, row.reach, row.follows))
    if not var_rows:
        return pd.DataFrame(columns=_STRAT_COLUMNS)

    vdf = pd.DataFrame(var_rows, columns=["ess_1", "ess_2", "var_keyword", "reach", "follows"])
    var_agg = vdf.groupby(["ess_1", "ess_2", "var_keyword"], as_index=False).agg(
        v_imps=("reach", "sum"), v_clicks=("follows", "sum"),
    )

    merged = var_agg.merge(
        top_essential[["ess_1", "ess_2", "combo_doc_freq", "combo_overall_ctr"]],
        on=["ess_1", "ess_2"], how="inner",
    )
    merged["noun_keyword"] = merged["var_keyword"].apply(lambda x: _normalize_keyword_by_pos(x, "noun"))
    merged = merged.dropna(subset=["noun_keyword"])
    if merged.empty:
        return pd.DataFrame(columns=_STRAT_COLUMNS)
    merged["var_keyword"] = merged["noun_keyword"]

    grouped = merged.groupby(
        ["ess_1", "ess_2", "combo_doc_freq", "combo_overall_ctr", "var_keyword"], as_index=False
    ).agg(v_clicks=("v_clicks", "sum"), var_imps=("v_imps", "sum"))
    grouped["with_var_ctr"] = np.where(
        grouped["var_imps"] > 0,
        np.round(grouped["v_clicks"] / grouped["var_imps"] * 1000, 2),
        np.nan,
    )
    grouped = grouped.sort_values(by=["combo_overall_ctr", "with_var_ctr"], ascending=[False, False])
    return grouped[_STRAT_COLUMNS]


# ============================================================
# datasets 조립
# ============================================================

def _bar_h_dataset_rate(title, df):
    """도달 1,000당 팔로우 발생 수 기준 차트. 바 내부 왼쪽에 '수치 (발생수명)'를 표시한다."""
    if df is None or df.empty:
        return None
    rates = df["avg_ctr"].fillna(0).tolist()
    counts = df["total_clicks"].fillna(0).tolist()
    bar_labels = [f"{rate:.2f} ({int(round(count)):,}명)" for rate, count in zip(rates, counts)]
    return {
        "kind": "bar_h",
        "title": title,
        "unit": "",
        "labels": df["keyword"].tolist(),
        "series": [{"name": "avg_ctr", "data": rates}],
        "bar_labels": bar_labels,
    }


def _bar_h_dataset_count(title, df):
    """키워드 포함 콘텐츠의 평균 팔로우 발생 수(총 팔로우 발생 / 등장 콘텐츠 수) 차트.
    바 내부 왼쪽에 평균 수치를 표시한다."""
    if df is None or df.empty:
        return None
    avg_counts = df["avg_follows_per_content"].fillna(0).tolist()
    bar_labels = [f"{avg:.2f}명" for avg in avg_counts]
    return {
        "kind": "bar_h",
        "title": title,
        "unit": "",
        "labels": df["keyword"].tolist(),
        "series": [{"name": "avg_follows_per_content", "data": avg_counts}],
        "bar_labels": bar_labels,
    }


def _table_dataset(title, df):
    if df is None or df.empty:
        return None
    return {
        "kind": "table",
        "title": title,
        "unit": "",
        "rows": df.replace({np.nan: None}).to_dict(orient="records"),
    }


def _build_combo_dataset(strat_df, label):
    if strat_df is None or strat_df.empty:
        return None
    strat_df = strat_df.copy()
    strat_df["combo_overall_ctr"] = pd.to_numeric(strat_df["combo_overall_ctr"], errors="coerce")
    strat_df["with_var_ctr"] = pd.to_numeric(strat_df["with_var_ctr"], errors="coerce")
    combo_keys = ["ess_1", "ess_2", "combo_overall_ctr"]
    combo_sizes = strat_df.groupby(combo_keys, dropna=False).size().reset_index(name="item_count")
    top_combos = (
        combo_sizes[combo_sizes["item_count"] >= 1]
        .dropna(subset=["combo_overall_ctr"])
        .sort_values(by="combo_overall_ctr", ascending=False)
        .head(6)
    )
    if top_combos.empty:
        return None
    final_df = strat_df.merge(top_combos[combo_keys], on=combo_keys, how="inner")
    final_df = final_df.sort_values(by=["combo_overall_ctr", "ess_1", "ess_2", "with_var_ctr"], ascending=[False, True, True, False])
    final_df = final_df.groupby(combo_keys, sort=False).head(8)
    return _table_dataset(f"{label} 상세 분석", final_df)


def _build_keyword_datasets(perf_df, min_doc_freq):
    datasets = {}
    raw_df = raw_keyword_performance_from_df(perf_df, min_doc_freq)

    # 팔로우=0인 콘텐츠에만 등장한 키워드가 많아(하위 10개가 대부분 0이 되어 변별력이 없음)
    # 하위 랭킹 페이지는 만들지 않고 상위 랭킹만 생성한다.
    for is_top in (True,):
        suffix = "top" if is_top else "bottom"

        # 도달 1,000당 팔로우 발생 수 기준
        rate_sorted_df = raw_df.sort_values(by=["avg_ctr", "total_impressions"], ascending=[not is_top, False])

        nouns = filter_keywords_by_pos(rate_sorted_df, "noun", exclude_zero_ctr=not is_top, sort_col="avg_ctr", rate_multiplier=1000)
        ds = _bar_h_dataset_rate(f"전체 {suffix.upper()} 10 (명사)", nouns)
        if ds:
            datasets[f"overall_{suffix}_noun"] = ds

        vas = filter_keywords_by_pos(rate_sorted_df, "verb_adj", exclude_zero_ctr=not is_top, sort_col="avg_ctr", rate_multiplier=1000)
        ds = _bar_h_dataset_rate(f"전체 {suffix.upper()} 10 (형용사)", vas)
        if ds:
            datasets[f"overall_{suffix}_va"] = ds

        # 키워드 포함 콘텐츠의 평균 팔로우 발생 수(총 팔로우 발생 / 등장 콘텐츠 수) 기준 — 표 대신 들어가는 별도 랭킹 차트
        count_sorted_df = raw_df.sort_values(by=["avg_follows_per_content", "total_impressions"], ascending=[not is_top, False])

        nouns_count = filter_keywords_by_pos(count_sorted_df, "noun", exclude_zero_ctr=not is_top, sort_col="avg_follows_per_content")
        ds = _bar_h_dataset_count(f"전체 {suffix.upper()} 10 (명사, 평균 발생 수)", nouns_count)
        if ds:
            datasets[f"overall_{suffix}_noun_count"] = ds

        vas_count = filter_keywords_by_pos(count_sorted_df, "verb_adj", exclude_zero_ctr=not is_top, sort_col="avg_follows_per_content")
        ds = _bar_h_dataset_count(f"전체 {suffix.upper()} 10 (형용사, 평균 발생 수)", vas_count)
        if ds:
            datasets[f"overall_{suffix}_va_count"] = ds

    if not raw_df.empty:
        dropped = sorted(
            kw for kw in set(raw_df["keyword"])
            if _normalize_keyword_by_pos(kw, "noun") is None and _normalize_keyword_by_pos(kw, "verb_adj") is None
        )
        if dropped:
            print(f"⚠️  명사/형용사·동사 어디로도 분류되지 않아 표에서 제외된 키워드 ({len(dropped)}개): {', '.join(dropped)}")

    strat_df = strategic_performance_from_df(perf_df, min_doc_freq)
    combo_ds = _build_combo_dataset(strat_df, "전체")
    if combo_ds:
        datasets["overall_keyword_combo_detail"] = combo_ds
    elif strat_df is None or strat_df.empty:
        print("ℹ️  키워드 조합 데이터 없음 — essential_keywords가 2개 이상인 게시물이 없거나 variable_keywords가 비어 있는지 확인하세요.")

    return datasets


def _weighted_rate(df):
    if df is None or df.empty:
        return None
    reach = df["reach"].sum()
    if reach <= 0:
        return None
    return float(df["follows"].sum()) / float(reach) * 1000.0


# 변수 지정 함수
# target_id 에 account_id
def run():
    start_time = time.time()

    config = {
        "target_id": 22,
        "csv_path": "data-follow/drr.csv",
        "brand": "De;part",
        "period_label": "",
        "min_doc_freq": 1,
        "theme_color": "#1C57AD",
        "output_html": "report_keywords_follow.html",
        "output_pdf_dir": "outputs",
    }

    target_id = config["target_id"]
    acc_name = get_account_name(target_id)

    perf_df = _load_perf_csv(config["csv_path"])
    perf_df = _attach_keywords(perf_df, target_id)

    datasets = _build_keyword_datasets(perf_df, config["min_doc_freq"])
    report_json = {"datasets": datasets}
    _apply_display_predicate_suffix(report_json)
    datasets = report_json["datasets"]

    theme_color = config["theme_color"]
    color_map = build_color_map(theme_color)
    comp_color_map = build_color_map(complementary_hex(theme_color))
    THEME_CMAP = [color_map["darker"], color_map["base"], color_map["light"]]
    theme = {
        "base": color_map["base"],
        "dark": color_map["dark"],
        "header": color_map["header"],
        "title": color_map["darker"],
        "highlight_main": color_map["highlight"],
        "highlight_avoid": comp_color_map["highlight"],
        "cover_text": "#ffffff" if is_dark_color(color_map["base"]) else "#000000",
    }

    charts = {}

    def add_chart(key, dataset_key):
        ds = datasets.get(dataset_key)
        svg = render_dataset(ds, color_map)
        if isinstance(svg, str) and svg:
            charts[key] = svg

    for pos_key, pos_label in (("noun", "noun"), ("va", "verb_adj")):
        add_chart(f"keyword_overall_top_{pos_label}", f"overall_top_{pos_key}")
        add_chart(f"keyword_overall_top_{pos_label}_count", f"overall_top_{pos_key}_count")

    # .page-note--left(폭 120mm)보다 텍스트가 길면 줄바꿈되므로 한 줄로 강제 고정
    rate_footnote = (
        '<span style="white-space:nowrap;">'
        "*도달 1,000당 팔로우 발생 수 = (총 팔로우 발생 / 총 도달) × 1,000"
        "</span>"
    )

    overall_rate_val = _weighted_rate(perf_df)
    overall_rate = f"{overall_rate_val:.2f}" if overall_rate_val is not None else "-"

    # min 기준이 1이라 doc_freq(등장 게시물 수) 필터가 사실상 의미가 없으므로
    # "N개 이상 콘텐츠 등장" 안내 문구는 넣지 않는다.
    combo_note = (
        "*업종 필수 키워드: 동일 업종의 상위 브랜드 10개의 웹사이트에서 자주 사용된 단어"
        "<br>*브랜드 변수 키워드: 필수 키워드 외 콘텐츠에 활용된 단어"
        f"<br><br>*전체 평균 도달 1,000당 팔로우 발생 수: {overall_rate}"
    )

    cards = _combo_cards(datasets.get("overall_keyword_combo_detail"), palette=THEME_CMAP, unit="")

    all_keywords = set()
    for kws in list(perf_df["essential_keywords"]) + list(perf_df["variable_keywords"]):
        all_keywords.update(k for k in kws if k)

    context = {
        "css_path": "./templates/report.css",
        "theme": theme,
        "report": {
            "title": "팔로우 키워드 분석 보고서",
            "client": acc_name,
            "quarter_label": config["period_label"],
            "year": "",
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "brand": config.get("brand", "De;part"),
            "period_label": config["period_label"],
            "ads_count": f"{perf_df['post_id'].nunique()}개",
            "keywords_count": f"{len(all_keywords)}개",
        },
        "charts": charts,
        "keywords": {
            "overall_top_note": rate_footnote,
            "overall_top_tables": None,
            "overall_combo_pages": [{"note": combo_note, "cards": cards}],
            "main_target": None,
            "main_top_tables": None,
            "main_combo_pages": None,
            "avoid_target": None,
            "avoid_top_tables": None,
            "avoid_combo_pages": None,
        },
    }

    html_path = _generate_keyword_html(context, config["output_html"])
    os.makedirs(config["output_pdf_dir"], exist_ok=True)
    output_pdf = os.path.join(config["output_pdf_dir"], f"{acc_name}_팔로우_키워드_분석_리포트.pdf")
    export_to_pdf(html_path, output_pdf)
    print(f"✅ {acc_name} 팔로우 키워드 분석 리포트 생성 완료! ({output_pdf})")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("-" * 50)
    print(f"⏳ 총 소요 시간: {elapsed_time:.2f}초")
    print("-" * 50)


if __name__ == "__main__":
    run()
