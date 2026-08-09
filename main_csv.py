"""
DB 없이 매뉴얼 CSV만으로 "타겟별 키워드" 분석 리포트를 생성하는 스크립트.

main_selected.py(DB 기반)와 달리 계정 성과(팔로워/CTR추이/구매전환/타겟 히트맵/
콘텐츠 top-bottom/별첨 키워드) 부분은 전부 제외하고, 아래 키워드 분석 부분만 생성한다.
    - 전체(overall) CTR 상위/하위 키워드 (명사 / 형용사·동사) + 상위 키워드 조합
    - 메인 타겟(main, 예: 특정 성별/연령) CTR 상위/하위 키워드 + 상위 키워드 조합 (선택)
    - 기피 타겟(avoid, 예: 특정 성별/연령) CTR 상위/하위 키워드 + 상위 키워드 조합 (선택)

키워드 추출(명사/형용사·동사 분류), 조합 카드 렌더링 등은 기존 scripts/processor.py,
main.py의 로직을 그대로 재사용하고, DB 쿼리(get_raw_keyword_performance_by_ids /
get_strategic_performance_by_ids)만 CSV 기반 pandas 연산으로 대체했다.

CSV 스키마 (sample.csv 참고)
--------------------------------
한 행 = 광고 1개 x 성별 1개 x 연령대 1개의 성과 데이터.
같은 ad_id라도 성별/연령대 조합마다 행을 나눠서 입력한다(essential_keywords /
variable_keywords는 같은 ad_id면 모든 행에 동일하게 반복 입력).

    ad_id               : 광고 식별자 (같은 광고면 동일한 값)
    gender              : "male" 또는 "female" (DB 원본과 동일하게 영문 사용, "unknown" 제외)
    age_range           : 연령대 문자열, 예) "18-24", "25-34" ... (main_age/avoid_age와 매칭됨)
    essential_keywords  : 광고의 핵심(필수) 키워드. "|"로 구분해 여러 개 입력 (최소 2개 이상이어야 조합 분석 대상)
    variable_keywords  : 브랜드/변수 키워드. "|"로 구분해 여러 개 입력 (없으면 비워둠)
    impressions         : 노출수 (정수)
    clicks              : 클릭수 (정수)

CTR 상하위 키워드는 "해당 키워드가 등장한 서로 다른 광고(ad_id) 수"가
config["min_doc_freq"] 이상인 키워드만 집계한다 (DB 버전과 동일하게 기본값 2).
"""

import itertools
import os
import re

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from scripts.processor import filter_keywords_by_pos, _normalize_keyword_by_pos
from scripts.visualizer import build_color_map, complementary_hex, is_dark_color, render_dataset
from main import _has_selector, _target_label, _combo_cards, _apply_display_predicate_suffix, export_to_pdf


def run():
    # ============================================================
    # ▼▼▼ 여기에 CSV 경로와 리포트 설정을 입력하세요 ▼▼▼
    config = {
        "csv_path": "ngt.csv",
        "account_name": "",
        "brand": "De;part",
        "period_label": "",
        # 타겟 필터 (필요 없으면 빈 문자열). 성별/연령 중 하나만 채워도 됨.
        # 예시: 성별로 메인/기피 타겟을 나누는 경우 (연령으로 나누고 싶으면 age_range 값을 사용)
        "main_age": "",
        "main_gender": "",
        "avoid_age": "",
        "avoid_gender": "",
        # CTR 상하위 키워드 최소 등장 광고(ad_id) 수
        "min_doc_freq": 2,
        "theme_color": "#182548",
        "output_html": "report_keywords.html",
        "output_pdf_dir": "outputs",
    }
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # ============================================================

    perf_df = _load_perf_csv(config["csv_path"])

    main_age, main_gender = config["main_age"], config["main_gender"]
    avoid_age, avoid_gender = config["avoid_age"], config["avoid_gender"]
    has_main_target = _has_selector(main_age) or _has_selector(main_gender)
    has_avoid_target = _has_selector(avoid_age) or _has_selector(avoid_gender)
    main_label = _target_label(main_age, main_gender)
    avoid_label = _target_label(avoid_age, avoid_gender)

    target_configs = [("overall", None, None, "전체")]
    if has_main_target:
        target_configs.append(("main", main_age, main_gender, "메인 타겟"))
    if has_avoid_target:
        target_configs.append(("avoid", avoid_age, avoid_gender, "기피 타겟"))

    datasets = _build_keyword_datasets(perf_df, target_configs, config["min_doc_freq"])
    report_json = {"datasets": datasets}
    _apply_display_predicate_suffix(report_json)
    datasets = report_json["datasets"]

    theme_color = config["theme_color"]
    color_map = build_color_map(theme_color)
    comp_color_map = build_color_map(complementary_hex(theme_color))
    THEME_CMAP = [color_map["darker"], color_map["base"], color_map["light"]]
    COMP_CMAP = [comp_color_map["darker"], comp_color_map["base"], comp_color_map["light"]]
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
    keyword_b_palette_keys = {
        "overall_bottom_noun", "overall_bottom_va",
        "main_bottom_noun", "main_bottom_va",
        "avoid_top_noun", "avoid_top_va",
    }

    def add_chart(key, dataset_key):
        ds = datasets.get(dataset_key)
        kwargs = {}
        if dataset_key in keyword_b_palette_keys and (ds or {}).get("kind") == "bar_h":
            kwargs["palette"] = COMP_CMAP
        svg = render_dataset(ds, color_map, **kwargs)
        if isinstance(svg, str) and svg:
            charts[key] = svg

    # 차트 dict 키(템플릿이 참조)는 main_selected.py와 동일하게 "_verb_adj"를 쓰고,
    # datasets 키(dataset 조회용, to_json_selected.py와 동일)는 "_va"를 쓴다 — 둘이 다르므로 섞어 쓰면 안 된다.
    for prefix in ("overall", "main", "avoid"):
        for suffix in ("top", "bottom"):
            for pos_key, pos_label in (("noun", "noun"), ("va", "verb_adj")):
                add_chart(f"keyword_{prefix}_{suffix}_{pos_label}", f"{prefix}_{suffix}_{pos_key}")

    def add_table(dataset_key, title, rank_head, kw_head):
        ds = datasets.get(dataset_key)
        if not ds or "labels" not in ds or "series" not in ds:
            return None
        labels = ds.get("labels", [])
        series_data = ds.get("series", [{}])[0].get("data", [])
        rows = []
        rank = 1
        for i, (label, value) in enumerate(zip(labels, series_data)):
            if i > 0 and value != series_data[i - 1]:
                rank = i + 1
            rows.append([f"{rank}위", label, f"{value:.2f}%"])
        if not rows:
            return None

        def _header_with_break(text):
            head = str(text)
            return head.replace("(", "<br>(") if "(" in head else head

        return {
            "title": title,
            "headers": [_header_with_break(rank_head), _header_with_break(kw_head), "평균 CTR"],
            "rows": rows,
            "footnote": "",
        }

    o_top = [
        add_table("overall_top_noun", "전체 TOP 10 (명사)", "순위(상위)", "키워드(명사)"),
        add_table("overall_top_va", "전체 TOP 10 (형용사/동사)", "순위(상위)", "키워드(형용사/동사)"),
    ]
    o_bot = [
        add_table("overall_bottom_noun", "전체 BOTTOM 10 (명사)", "순위(하위)", "키워드(명사)"),
        add_table("overall_bottom_va", "전체 BOTTOM 10 (형용사/동사)", "순위(하위)", "키워드(형용사/동사)"),
    ]

    m_top, m_bot = [], []
    if has_main_target:
        m_top = [add_table("main_top_noun", f"{main_label} TOP 10 (명사)", "순위(상위)", "키워드(명사)"), add_table("main_top_va", f"{main_label} TOP 10 (형용사/동사)", "순위(상위)", "키워드(형용사/동사)")]
        m_bot = [add_table("main_bottom_noun", f"{main_label} BOTTOM 10 (명사)", "순위(하위)", "키워드(명사)"), add_table("main_bottom_va", f"{main_label} BOTTOM 10 (형용사/동사)", "순위(하위)", "키워드(형용사/동사)")]

    a_top, a_bot = [], []
    if has_avoid_target:
        a_top = [add_table("avoid_top_noun", f"{avoid_label} TOP 10 (명사)", "순위(상위)", "키워드(명사)"), add_table("avoid_top_va", f"{avoid_label} TOP 10 (형용사/동사)", "순위(상위)", "키워드(형용사/동사)")]
        a_bot = [add_table("avoid_bottom_noun", f"{avoid_label} BOTTOM 10 (명사)", "순위(하위)", "키워드(명사)"), add_table("avoid_bottom_va", f"{avoid_label} BOTTOM 10 (형용사/동사)", "순위(하위)", "키워드(형용사/동사)")]

    filter_none = lambda lst: [t for t in lst if t is not None]

    overall_ctr_val = _weighted_ctr(_base_filtered(perf_df))
    overall_ctr = f"{overall_ctr_val:.2f}" if overall_ctr_val is not None else "-"
    main_ctr_val = _weighted_ctr(_base_filtered(perf_df, main_age, main_gender)) if has_main_target else None
    main_ctr = f"{main_ctr_val:.2f}" if main_ctr_val is not None else "-"
    avoid_ctr_val = _weighted_ctr(_base_filtered(perf_df, avoid_age, avoid_gender)) if has_avoid_target else None
    avoid_ctr = f"{avoid_ctr_val:.2f}" if avoid_ctr_val is not None else "-"

    cards = _combo_cards(datasets.get("overall_keyword_combo_detail"), palette=THEME_CMAP)
    cards_main = _combo_cards(datasets.get("main_keyword_combo_detail"), palette=THEME_CMAP) if has_main_target else []
    cards_avoid = _combo_cards(datasets.get("avoid_keyword_combo_detail"), palette=COMP_CMAP) if has_avoid_target else []

    for label_, key_, cards_ in (("전체", "overall_keyword_combo_detail", cards), ("메인 타겟", "main_keyword_combo_detail", cards_main), ("기피 타겟", "avoid_keyword_combo_detail", cards_avoid)):
        if datasets.get(key_) and not cards_:
            print(
                f"⚠️  [{label_}] 키워드 조합 표는 있지만 조합 카드가 0개입니다. "
                f"조합 카드는 동일한 필수 키워드 쌍(ess_1+ess_2)에 대해 서로 다른 브랜드/변수 키워드가 "
                f"2개 이상 있어야 표시됩니다 — 광고마다 variable_keywords를 다양하게 입력했는지 확인하세요."
            )

    all_keywords = set()
    for kws in list(perf_df["essential_keywords"]) + list(perf_df["variable_keywords"]):
        all_keywords.update(k for k in kws if k)

    context = {
        "css_path": "./templates/report.css",
        "theme": theme,
        "report": {
            "title": "키워드 분석 보고서",
            "client": config["account_name"],
            "quarter_label": config["period_label"],
            "year": "",
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "brand": config.get("brand", "De;part"),
            "period_label": config["period_label"],
            "ads_count": f"{perf_df['ad_id'].nunique()}개",
            "keywords_count": f"{len(all_keywords)}개",
        },
        "charts": charts,
        "keywords": {
            "overall_top_note": f"*{config['min_doc_freq']}개 이상의 광고에 등장한 단어만 표시",
            "overall_top_tables": filter_none(o_top),
            "overall_combo_pages": [{"note": f"*전체 평균 CTR: {overall_ctr}%", "cards": cards}],
            "overall_bottom_note": f"*{config['min_doc_freq']}개 이상의 광고에 등장한 단어만 표시",
            "overall_bottom_tables": filter_none(o_bot),
            "main_target": {"title": main_label} if has_main_target else None,
            "main_top_tables": filter_none(m_top) if m_top else None,
            "main_combo_pages": [{"note": f"*{main_label} 평균 CTR: {main_ctr}%", "cards": cards_main}] if has_main_target else None,
            "main_bottom_tables": filter_none(m_bot) if m_bot else None,
            "avoid_target": {"title": avoid_label} if has_avoid_target else None,
            "avoid_top_tables": filter_none(a_top) if a_top else None,
            "avoid_combo_pages": [{"note": f"*{avoid_label} 평균 CTR: {avoid_ctr}%", "cards": cards_avoid}] if has_avoid_target else None,
            "avoid_bottom_tables": filter_none(a_bot) if a_bot else None,
        },
    }

    html_path = _generate_keyword_html(context, config["output_html"])
    os.makedirs(config["output_pdf_dir"], exist_ok=True)
    output_pdf = os.path.join(config["output_pdf_dir"], f"{config['account_name']}_키워드_분석_리포트.pdf")
    export_to_pdf(html_path, output_pdf)
    print(f"✅ {config['account_name']} 키워드 분석 리포트 생성 완료! ({output_pdf})")


# ============================================================
# CSV 로딩
# ============================================================

_REQUIRED_COLUMNS = {"ad_id", "gender", "age_range", "essential_keywords", "variable_keywords", "impressions", "clicks"}


_PG_ARRAY_ITEM_RE = re.compile(r'"([^"]*)"|([^,]+)')


def _split_keywords(raw):
    """essential_keywords / variable_keywords 셀 하나를 키워드 리스트로 분리한다.
    아래 세 가지 표기를 모두 인식한다.
      - "가을|니트" 처럼 파이프(|)로 구분 (수기 입력 권장 표기)
      - "{가을,니트}" 처럼 Postgres 배열 리터럴 (DB에서 그대로 export한 CSV 대응,
        따옴표로 감싸 콤마를 포함한 항목도 처리)
      - "가을,니트" 처럼 콤마로만 구분
    """
    if pd.isna(raw):
        return []
    text = str(raw).strip()
    if not text:
        return []

    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        items = []
        for quoted, plain in _PG_ARRAY_ITEM_RE.findall(inner):
            val = (quoted if quoted else plain).strip()
            if val:
                items.append(val)
        return items

    if "|" in text:
        return [t.strip() for t in text.split("|") if t.strip()]

    return [t.strip() for t in text.split(",") if t.strip()]


def _load_perf_csv(path):
    df = pd.read_csv(path, dtype={"ad_id": str})
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {sorted(missing)}")

    df["gender"] = df["gender"].astype(str).str.strip()
    df["age_range"] = df["age_range"].astype(str).str.strip()
    df["essential_keywords"] = df["essential_keywords"].apply(_split_keywords)
    df["variable_keywords"] = df["variable_keywords"].apply(_split_keywords)
    df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0)
    df["clicks"] = pd.to_numeric(df["clicks"], errors="coerce").fillna(0)
    return df


# ============================================================
# 타겟(성별/연령) 필터링 — processor.py의 _build_target_filter와 동일한 규칙
# ============================================================

def _to_str_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _map_gender(value):
    low = str(value).strip().lower()
    if low in ("f", "여성"):
        return "female"
    if low in ("m", "남성"):
        return "male"
    return value


def _base_filtered(df, age=None, gender=None):
    base = df[df["gender"].str.lower() != "unknown"]
    ages = _to_str_list(age)
    genders = [_map_gender(g) for g in _to_str_list(gender)]
    if ages:
        base = base[base["age_range"].isin(ages)]
    if genders:
        base = base[base["gender"].isin(genders)]
    return base


def _weighted_ctr(df):
    if df is None or df.empty:
        return None
    impressions = df["impressions"].sum()
    if impressions <= 0:
        return None
    return float(df["clicks"].sum()) / float(impressions) * 100.0


def _union_keywords(series):
    """같은 ad_id의 여러 행(성별/연령대別)에 essential/variable_keywords를
    일부 행에만 입력했거나 표기가 살짝 다른 경우까지 포용하기 위해,
    첫 행 값만 쓰지 않고 해당 ad_id의 모든 행에 등장한 키워드를 합집합으로 모은다."""
    merged = []
    seen = set()
    for lst in series:
        for kw in lst:
            if kw and kw not in seen:
                seen.add(kw)
                merged.append(kw)
    return merged


def _per_ad_totals(df):
    if df.empty:
        return df.iloc[0:0]
    totals = df.groupby("ad_id", as_index=False).agg(ad_imp=("impressions", "sum"), ad_clk=("clicks", "sum"))
    keywords = df.groupby("ad_id", as_index=False).agg(
        essential_keywords=("essential_keywords", _union_keywords),
        variable_keywords=("variable_keywords", _union_keywords),
    )
    return totals.merge(keywords, on="ad_id", how="left")


# ============================================================
# 키워드 성과 (processor.get_raw_keyword_performance_by_ids와 동일한 로직, CSV 기반)
# ============================================================

_RAW_KW_COLUMNS = ["keyword", "doc_freq", "total_impressions", "total_clicks", "avg_ctr"]


def raw_keyword_performance_from_df(df, age=None, gender=None, min_doc_freq=2):
    base = _base_filtered(df, age, gender)
    if base.empty:
        return pd.DataFrame(columns=_RAW_KW_COLUMNS)

    per_ad = _per_ad_totals(base)
    rows = []
    for r in per_ad.itertuples(index=False):
        keywords = set(r.essential_keywords) | set(r.variable_keywords)
        for kw in keywords:
            rows.append((r.ad_id, kw, r.ad_imp, r.ad_clk))
    if not rows:
        return pd.DataFrame(columns=_RAW_KW_COLUMNS)

    kdf = pd.DataFrame(rows, columns=["ad_id", "keyword", "ad_imp", "ad_clk"])
    agg = kdf.groupby("keyword", as_index=False).agg(
        doc_freq=("ad_id", "nunique"),
        total_impressions=("ad_imp", "sum"),
        total_clicks=("ad_clk", "sum"),
    )
    agg = agg[agg["doc_freq"] >= min_doc_freq].copy()
    if agg.empty:
        return pd.DataFrame(columns=_RAW_KW_COLUMNS)
    agg["avg_ctr"] = np.where(
        agg["total_impressions"] > 0,
        np.round(agg["total_clicks"] / agg["total_impressions"] * 100, 2),
        np.nan,
    )
    return agg[_RAW_KW_COLUMNS]


# ============================================================
# 키워드 조합 성과 (processor.get_strategic_performance_by_ids와 동일한 로직, CSV 기반)
# ============================================================

_STRAT_COLUMNS = ["ess_1", "ess_2", "combo_doc_freq", "combo_overall_ctr", "var_keyword", "with_var_ctr", "var_imps"]


def strategic_performance_from_df(df, age=None, gender=None):
    base = _base_filtered(df, age, gender)
    if base.empty:
        return pd.DataFrame(columns=_STRAT_COLUMNS)

    per_ad = _per_ad_totals(base)
    per_ad = per_ad[per_ad["essential_keywords"].apply(lambda ks: len(set(k for k in ks if k)) >= 2)]
    if per_ad.empty:
        return pd.DataFrame(columns=_STRAT_COLUMNS)

    pair_rows = []
    for r in per_ad.itertuples(index=False):
        ess_unique = sorted(set(k for k in r.essential_keywords if k))
        var_unique = sorted(set(k for k in r.variable_keywords if k))
        for e1, e2 in itertools.combinations(ess_unique, 2):
            pair_rows.append((r.ad_id, e1, e2, r.ad_imp, r.ad_clk, var_unique))
    if not pair_rows:
        return pd.DataFrame(columns=_STRAT_COLUMNS)
    pairs_df = pd.DataFrame(pair_rows, columns=["ad_id", "ess_1", "ess_2", "ad_imp", "ad_clk", "variable_keywords"])

    essential_agg = pairs_df.groupby(["ess_1", "ess_2"], as_index=False).agg(
        combo_doc_freq=("ad_id", "nunique"),
        total_imps=("ad_imp", "sum"),
        total_clicks=("ad_clk", "sum"),
    )
    essential_agg["combo_overall_ctr"] = np.where(
        essential_agg["total_imps"] > 0,
        np.round(essential_agg["total_clicks"] / essential_agg["total_imps"] * 100, 2),
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
            var_rows.append((row.ess_1, row.ess_2, vk, row.ad_imp, row.ad_clk))
    if not var_rows:
        return pd.DataFrame(columns=_STRAT_COLUMNS)

    vdf = pd.DataFrame(var_rows, columns=["ess_1", "ess_2", "var_keyword", "ad_imp", "ad_clk"])
    var_agg = vdf.groupby(["ess_1", "ess_2", "var_keyword"], as_index=False).agg(
        v_imps=("ad_imp", "sum"), v_clicks=("ad_clk", "sum"),
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
        np.round(grouped["v_clicks"] / grouped["var_imps"] * 100, 2),
        np.nan,
    )
    grouped = grouped.sort_values(by=["combo_overall_ctr", "with_var_ctr"], ascending=[False, False])
    return grouped[_STRAT_COLUMNS]


# ============================================================
# datasets 조립 (to_json_selected.py 11번 섹션과 동일한 규칙)
# ============================================================

def _bar_h_dataset(title, df):
    if df is None or df.empty:
        return None
    return {
        "kind": "bar_h",
        "title": title,
        "unit": "%",
        "labels": df["keyword"].tolist(),
        "series": [{"name": "avg_ctr", "data": df["avg_ctr"].fillna(0).tolist()}],
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


def _build_keyword_datasets(perf_df, target_configs, min_doc_freq):
    datasets = {}
    for prefix, age, gender, label in target_configs:
        raw_df = raw_keyword_performance_from_df(perf_df, age, gender, min_doc_freq)
        for is_top in (True, False):
            suffix = "top" if is_top else "bottom"
            exclude_zero_ctr = not is_top
            sorted_df = raw_df.sort_values(by=["avg_ctr", "total_impressions"], ascending=[not is_top, False])

            nouns = filter_keywords_by_pos(sorted_df, "noun", exclude_zero_ctr=exclude_zero_ctr)
            ds = _bar_h_dataset(f"{label} {suffix.upper()} 10 (명사)", nouns)
            if ds:
                datasets[f"{prefix}_{suffix}_noun"] = ds

            vas = filter_keywords_by_pos(sorted_df, "verb_adj", exclude_zero_ctr=exclude_zero_ctr)
            ds = _bar_h_dataset(f"{label} {suffix.upper()} 10 (형용사)", vas)
            if ds:
                datasets[f"{prefix}_{suffix}_va"] = ds

        if not raw_df.empty:
            # top-10/bottom-10로 잘리기 전, 원본 키워드 전체를 기준으로
            # 명사/형용사·동사 어디에도 분류되지 않는 키워드만 골라낸다
            # (head(10) 절단으로 인한 오탐을 피하기 위해 필터 함수가 아닌
            # 분류 함수(_normalize_keyword_by_pos)를 키워드 단위로 직접 호출한다).
            dropped = sorted(
                kw for kw in set(raw_df["keyword"])
                if _normalize_keyword_by_pos(kw, "noun") is None and _normalize_keyword_by_pos(kw, "verb_adj") is None
            )
            if dropped:
                print(
                    f"⚠️  [{label}] 명사/형용사·동사 어디로도 분류되지 않아 표에서 제외된 키워드 "
                    f"({len(dropped)}개): {', '.join(dropped)}"
                )
                print(
                    "    → kiwi 형태소 분석기가 '단독 어간' 형태를 인식하지 못하면 발생합니다. "
                    "예) 형용사는 '따뜻' 대신 '따뜻하'처럼 '하'를 붙여 입력하면 인식률이 올라갑니다."
                )

        strat_df = strategic_performance_from_df(perf_df, age, gender)
        combo_ds = _build_combo_dataset(strat_df, label)
        if combo_ds:
            datasets[f"{prefix}_keyword_combo_detail"] = combo_ds
        elif strat_df is None or strat_df.empty:
            print(
                f"ℹ️  [{label}] 키워드 조합 데이터 없음 — essential_keywords가 2개 이상인 광고가 없거나 "
                "variable_keywords가 비어 있는지 확인하세요."
            )

    return datasets


# ============================================================
# 렌더링 (scripts/reporter.py의 generate_html과 동일한 방식이나
# template_keywords.html을 사용하고, 임의 출력 경로를 지정할 수 있음)
# ============================================================

def _translate_gender(data):
    if isinstance(data, dict):
        return {k: _translate_gender(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_translate_gender(i) for i in data]
    if isinstance(data, str):
        res = data.replace("female", "여성").replace("Female", "여성")
        res = res.replace("male", "남성").replace("Male", "남성")
        return res
    return data


def _generate_keyword_html(context, output_path):
    translated_context = _translate_gender(context)
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("template_keywords.html")
    output = template.render(translated_context)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    run()
