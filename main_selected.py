import json
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
import pandas as pd
from scripts.processor import _normalize_keyword_by_pos, _best_adverb_score, kiwi, VERB_ADJ_TAGS
from scripts.visualizer import build_color_map, complementary_hex, render_dataset, is_dark_color, render_bubble_chart, render_purchase_pie_chart, render_follower_gender_doughnut_chart, render_follower_age_gender_stacked_barh_chart
from scripts.reporter import generate_html
from to_json_selected import run as generate_json
import time

# main.py의 유틸 함수들 그대로 재사용
from main import (
    _load_env_file, _parse_s3_location, _safe_name,
    _materialize_content_thumbnails,
    _load_report, _top_targets, _normalize_selector, _has_selector,
    _append_da_if_predicate, _is_predicate_for_display,
    _transform_rows_labels, _apply_display_predicate_suffix,
    _walk_display_blocks, _target_ctr, _target_label,
    _average_series, _combo_cards, export_to_pdf,
)

_KOREAN_RE = re.compile(r"[가-힣]")


def run():
    start_time = time.time()

    # ============================================================
    # ▼▼▼ 여기에 보고서를 생성할 ad_id 목록을 입력하세요 ▼▼▼
    config = {
        "ad_ids": [
            # 예시: 123456, 789012
        ],
        # fb_ad_account_id: Instagram 팔로워/오가닉 데이터를 포함하려면 입력
        # 불필요하면 빈 문자열("")로 두세요
        "fb_ad_account_id": "",
        # 기간 강제 지정 (빈 문자열이면 ad_ids의 성과 데이터 기간 자동 사용)
        "start": "",
        "end": "",
        # 타겟 필터 (필요 없으면 빈 문자열)
        "main_age": [],
        "main_gender": "",
        "avoid_age": "",
        "avoid_gender": "",
        "currency": "",   # ""=원화, "dollar"=달러
    }
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # ============================================================

    ad_ids = config["ad_ids"]
    if not ad_ids:
        raise ValueError("config['ad_ids'] 에 광고 ID를 1개 이상 입력해주세요.")

    fb_ad_account_id = config.get("fb_ad_account_id", "")
    date_start = config.get("start", "")
    date_end = config.get("end", "")
    main_age = config["main_age"]
    main_gender = config["main_gender"]
    avoid_age = config["avoid_age"]
    avoid_gender = config["avoid_gender"]
    currency = config["currency"]

    has_main_target = _has_selector(main_age) or _has_selector(main_gender)
    has_avoid_target = _has_selector(avoid_age) or _has_selector(avoid_gender)
    main_label = _target_label(main_age, main_gender)
    avoid_label = _target_label(avoid_age, avoid_gender)

    # JSON 생성
    generate_json(
        ad_ids=ad_ids,
        fb_ad_account_id=fb_ad_account_id,
        start=date_start,
        end=date_end,
        main_age=main_age,
        main_gender=main_gender,
        avoid_age=avoid_age,
        avoid_gender=avoid_gender,
        currency=currency,
    )

    report_path = "json_reports/integrated_report.json"
    theme_color = "#8C8C89"

    report_json = _load_report(report_path)
    _apply_display_predicate_suffix(report_json)
    meta = report_json.get("meta", {})
    summary = report_json.get("summary", {})
    datasets = report_json.get("datasets", {})

    acc_name = meta.get("account_name", "")
    period = meta.get("period", "")
    period_ads = meta.get("period_ads", "")
    period_contents = meta.get("period_contents", "")
    year = period.split("-")[0] if period else ""
    generated_at = meta.get("generated_at") or datetime.now().strftime("%Y-%m-%d %H:%M")

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

    def add_chart(key, dataset_key, **kwargs):
        ds = datasets.get(dataset_key)
        if dataset_key in keyword_b_palette_keys and (ds or {}).get("kind") == "bar_h":
            kwargs.setdefault("palette", COMP_CMAP)
        svg = render_dataset(ds, color_map, **kwargs)
        if isinstance(svg, str) and svg:
            charts[key] = svg

    def _count_text(value):
        if value is None:
            return "-"
        txt = str(value).strip()
        if not txt or txt == "-":
            return "-"
        return f"{txt}개"

    add_chart("followers", "insta_followers")
    add_chart("ctr_weekly", "ctr_trend_weekly")
    add_chart("ctr_monthly", "ctr_trend_monthly")
    add_chart("organic_views_1", "organic_trend")
    add_chart("organic_views_2", "organic_trend_monthly")
    add_chart("profile_visits_1", "insta_profile_visits")
    add_chart("profile_visits_2", "insta_profile_visits_monthly")

    heatmap_ds = datasets.get("target_heatmap")
    heatmap_imp = render_dataset(heatmap_ds, color_map, metric="impressions")
    if heatmap_imp:
        charts["heatmap_impressions"] = heatmap_imp
    heatmap_ctr = render_dataset(heatmap_ds, color_map, metric="ctr")
    if heatmap_ctr:
        charts["heatmap_ctr"] = heatmap_ctr

    purchase_heatmap_ds = datasets.get("purchase_heatmap")
    purchase_heatmap = render_dataset(purchase_heatmap_ds, color_map, metric="purchases")
    if purchase_heatmap:
        charts["purchase_heatmap"] = purchase_heatmap

    add_chart("keyword_overall_top_noun", "overall_top_noun")
    add_chart("keyword_overall_top_verb_adj", "overall_top_va")
    add_chart("keyword_overall_bottom_noun", "overall_bottom_noun")
    add_chart("keyword_overall_bottom_verb_adj", "overall_bottom_va")
    add_chart("keyword_main_top_noun", "main_top_noun")
    add_chart("keyword_main_top_verb_adj", "main_top_va")
    add_chart("keyword_main_bottom_noun", "main_bottom_noun")
    add_chart("keyword_main_bottom_verb_adj", "main_bottom_va")
    add_chart("keyword_avoid_top_noun", "avoid_top_noun")
    add_chart("keyword_avoid_top_verb_adj", "avoid_top_va")
    add_chart("keyword_avoid_bottom_noun", "avoid_bottom_noun")
    add_chart("keyword_avoid_bottom_verb_adj", "avoid_bottom_va")
    add_chart("purchase_roas_weekly", "purchase_roas_weekly")
    add_chart("purchase_roas_monthly", "purchase_roas_monthly")
    add_chart("purchase_count_weekly", "purchase_count_weekly")
    add_chart("purchase_count_monthly", "purchase_count_monthly")
    add_chart("spend_revenue_weekly", "spend_revenue_weekly")
    add_chart("spend_revenue_monthly", "spend_revenue_monthly")

    gender_clean_ds = datasets.get("gender_clean")
    if gender_clean_ds and gender_clean_ds.get("labels") and gender_clean_ds.get("series"):
        charts["gender_clean"] = render_follower_gender_doughnut_chart(gender_clean_ds, color_map)
    age_gender_clean_ds = datasets.get("age_gender_clean")
    if age_gender_clean_ds and age_gender_clean_ds.get("labels") and age_gender_clean_ds.get("series"):
        charts["age_clean"] = render_follower_age_gender_stacked_barh_chart(age_gender_clean_ds, color_map)
    gender_unknown_ds = datasets.get("gender_unknown")
    if gender_unknown_ds and gender_unknown_ds.get("labels") and gender_unknown_ds.get("series"):
        charts["gender_unknown"] = render_follower_gender_doughnut_chart(gender_unknown_ds, color_map)
    age_known_unknown_ds = datasets.get("age_known_unknown")
    if age_known_unknown_ds and age_known_unknown_ds.get("labels") and age_known_unknown_ds.get("series"):
        charts["age_unknown"] = render_follower_age_gender_stacked_barh_chart(age_known_unknown_ds, color_map)

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

        return {"title": title, "headers": [_header_with_break(rank_head), _header_with_break(kw_head), "평균 CTR"], "rows": rows, "footnote": ""}

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

    top_items = render_dataset(datasets.get("content_top_analysis"), color_map)
    if not isinstance(top_items, list):
        top_items = []
    bottom_items = render_dataset(datasets.get("content_bottom_analysis"), color_map)
    if not isinstance(bottom_items, list):
        bottom_items = []
    _materialize_content_thumbnails(top_items + bottom_items)

    target_rows = (datasets.get("target_heatmap") or {}).get("rows") or []
    impressions_rank, impressions_footnote = _top_targets(target_rows, "impressions")
    ctr_rank, ctr_footnote = _top_targets(target_rows, "ctr", filter_low_imps=True)

    purchase_rows = (datasets.get("purchase_heatmap") or {}).get("rows") or []
    purchase_rank, purchase_footnote = _top_targets(purchase_rows, "purchases")

    overall_ctr_val = _average_series(datasets.get("ctr_trend_weekly"))
    overall_ctr = f"{overall_ctr_val:.2f}" if isinstance(overall_ctr_val, (int, float)) else "-"

    main_ctr_val = _target_ctr(target_rows, main_age, main_gender) if has_main_target else None
    main_ctr = f"{main_ctr_val:.2f}" if isinstance(main_ctr_val, (int, float)) else "-"

    avoid_ctr_val = _target_ctr(target_rows, avoid_age, avoid_gender) if has_avoid_target else None
    avoid_ctr = f"{avoid_ctr_val:.2f}" if isinstance(avoid_ctr_val, (int, float)) else "-"

    cards = _combo_cards(datasets.get("overall_keyword_combo_detail"), palette=THEME_CMAP)
    cards_main = _combo_cards(datasets.get("main_keyword_combo_detail"), palette=THEME_CMAP) if has_main_target else []
    cards_avoid = _combo_cards(datasets.get("avoid_keyword_combo_detail"), palette=COMP_CMAP) if has_avoid_target else []

    purchase_contents_pages = report_json.get("purchase_contents_pages", {"is_visible": False})
    if purchase_contents_pages.get("is_visible"):
        for page_items in purchase_contents_pages.get("pages", []):
            _materialize_content_thumbnails(page_items)
            for item in page_items:
                target_details = item.get("target_details") or []
                item["chart"] = render_purchase_pie_chart(target_details, color_map) if target_details else ""

    context = {
        "css_path": "./templates/report.css",
        "theme": theme,
        "report": {
            "title": "보고서",
            "client": acc_name,
            "quarter_label": period,
            "year": year,
            "generated_at": generated_at,
            "brand": "De;part",
            "period_ads": period_ads or "-",
            "period_contents": period_contents or "-",
            "keyword_count": f"{summary.get('total_keywords', '-')}개",
            "ads_count": _count_text(summary.get("total_ads")),
            "contents_count": _count_text(summary.get("total_contents")),
            "keywords_count": _count_text(summary.get("total_keywords")),
            "overview_notes": [
                f"광고 {summary.get('total_ads', '-')}개",
                f"콘텐츠 {summary.get('total_contents', '-')}개",
            ],
        },
        "content": {
            "top_note": "",
            "top": top_items,
            "bottom_note": "",
            "bottom": bottom_items,
            "overall_ctr": overall_ctr,
        },
        "charts": charts,
        "annotations": {"ctr": [], "organic": []},
        "target": {
            "impressions_rank": impressions_rank,
            "impressions_footnote": impressions_footnote,
            "ctr_note": "",
            "ctr_rank": ctr_rank,
            "ctr_footnote": ctr_footnote,
            "purchase_rank": purchase_rank,
            "purchase_footnote": purchase_footnote,
        },
        "keywords": {
            "overall_top_note": "*1개 이상의 콘텐츠에 등장한 단어만 표시",
            "overall_top_tables": filter_none(o_top),
            "overall_combo_pages": [{"note": f"*계정 전체 평균 CTR: {overall_ctr}%", "cards": cards}],
            "overall_bottom_note": "*1개 이상의 콘텐츠에 등장한 단어만 표시",
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
        "appendix_groups": report_json.get("appendix_groups", []),
        "appendix": [],
        "purchase_analysis_pages": report_json.get("purchase_analysis_pages", {"is_visible": False}),
        "purchase_contents_pages": report_json.get("purchase_contents_pages", {"is_visible": False}),
        "purchase_age_gender_page": report_json.get("purchase_age_gender_page", {"is_visible": False}),
        "spend_revenue_pages": report_json.get("spend_revenue_pages", {"is_visible": False}),
        "follower_demographics_pages": report_json.get("follower_demographics_pages", {"is_visible": False}),
    }

    generate_html(context)
    export_to_pdf("report.html", f"outputs/{acc_name}_선택광고_리포트.pdf")
    print(f"✅ {acc_name} 선택 광고 리포트 생성 완료!")

    end_time = time.time()
    print("-" * 50)
    print(f"⏳ 총 소요 시간: {end_time - start_time:.2f}초")
    print("-" * 50)


if __name__ == "__main__":
    run()
