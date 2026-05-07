import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from scripts.processor import (
    # 계정 레벨 (account_id 기반)
    get_instagram_followers, get_profile_visits_monthly,
    get_organic_data, get_organic_monthly_data,
    has_follower_demographics_data, get_follower_demographics_latest_date,
    get_demographics_ratio, get_follower_age_gender_known_only,
    get_age_known_unknown_by_age, get_follower_age_gender_distribution,
    # ad_ids 기반
    get_ad_meta_by_ids,
    get_active_ad_count_by_ids, get_total_content_count_by_ids,
    get_ad_period_by_ids, get_content_period_by_ids,
    get_total_keyword_count_by_ids,
    get_ctr_data_by_ids, get_ctr_monthly_data_by_ids,
    get_imp_threshold_by_ids, get_target_heatmap_by_ids,
    get_purchase_heatmap_by_ids, get_purchase_age_gender_heatmap_page_data_by_ids,
    get_raw_keyword_performance_by_ids, get_strategic_performance_by_ids,
    get_content_cards_by_ids, get_a_content_target_ctr_data,
    has_purchase_data_by_ids,
    get_purchase_roas_weekly_by_ids, get_purchase_roas_monthly_by_ids,
    get_purchase_count_weekly_by_ids, get_purchase_count_monthly_by_ids,
    has_revenue_data_by_ids,
    get_spend_and_revenue_weekly_by_ids, get_spend_and_revenue_monthly_by_ids,
    get_purchase_contents_pages_data_by_ids, get_a_content_target_purchase_data,
    get_essence_target_performance_by_ids, get_variable_target_performance_by_ids,
    filter_keywords_by_pos,
)


def run(ad_ids, fb_ad_account_id="", main_age="", main_gender="",
        avoid_age="", avoid_gender="", currency="",
        start="", end=""):
    """
    ad_ids: 보고서를 생성할 ad_id 리스트 (필수)
    fb_ad_account_id: Instagram 팔로워/오가닉 데이터 조회용 (선택)
    start/end: 기간 강제 지정 (빈 문자열이면 DB에서 자동 도출)
    """
    if not ad_ids:
        raise ValueError("ad_ids가 비어 있습니다.")

    # 1. ad_ids에서 계정 메타 자동 도출
    meta = get_ad_meta_by_ids(ad_ids)
    if meta is None:
        raise RuntimeError(f"ad_ids {ad_ids} 에 해당하는 데이터를 찾을 수 없습니다.")

    target_id = meta["account_id"]
    acc_name = meta["brand_name"]
    db_start = meta["start"]
    db_end = meta["end"]

    date_start = start if start else db_start
    date_end = end if end else db_end

    end_dt = datetime.strptime(date_end, "%Y-%m-%d")
    actual_end = (end_dt - timedelta(days=end_dt.weekday())).strftime("%Y-%m-%d")

    ad_start, ad_end = get_ad_period_by_ids(ad_ids)
    content_start, content_end = get_content_period_by_ids(ad_ids, date_start, date_end)

    currency_symbol = "$" if currency == "dollar" else "원"

    # 2. 최종 리포트 구조
    final_report = {
        "meta": {
            "account_name": acc_name,
            "period": f"{date_start} ~ {actual_end}",
            "period_ads": f"{ad_start} ~ {ad_end}",
            "period_contents": f"{content_start} ~ {content_end}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "currency": currency,
        "currency_symbol": currency_symbol,
        "summary": {
            "total_ads": get_active_ad_count_by_ids(ad_ids, date_start, date_end),
            "total_contents": get_total_content_count_by_ids(ad_ids, date_start, date_end),
            "total_keywords": get_total_keyword_count_by_ids(ad_ids, date_start, date_end),
        },
        "datasets": {},
    }
    datasets = final_report["datasets"]

    def add_ds(key, kind, title, df, unit="", x=None, ys=None, extra_meta=None):
        if df is None or df.empty:
            return
        df_c = df.copy().reset_index() if df.index.name else df.copy()
        columns = list(df_c.columns)

        if kind != "table" and x not in columns:
            possible_x = [c for c in columns if any(w in c.lower() for w in ["date", "at", "start", "week", "time"])]
            x = possible_x[0] if possible_x else columns[0]

        if x in columns and any(w in x.lower() for w in ["date", "time", "at"]):
            try:
                df_c[x] = pd.to_datetime(df_c[x], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        data_obj = {"kind": kind, "title": title, "unit": unit}
        if extra_meta:
            data_obj.update(extra_meta)

        if kind == "table":
            data_obj["rows"] = df_c.replace({pd.NA: None, pd.NaT: None, np.nan: None}).to_dict(orient="records")
        else:
            data_obj["labels"] = df_c[x].fillna("Unknown").tolist()
            target_ys = ys if ys else df_c.select_dtypes(include=["number"]).columns.tolist()
            series_data = []
            for y_req in target_ys:
                matched = next((c for c in columns if y_req.lower() == c.lower() or y_req.lower() in c.lower()), None)
                if matched and matched != x:
                    series_data.append({"name": matched, "data": df_c[matched].fillna(0).tolist()})
            data_obj["series"] = series_data

        datasets[key] = data_obj

    # 3. 인스타그램 / 오가닉 (계정 레벨 – fb_ad_account_id 있을 때만)
    if fb_ad_account_id:
        print("인스타그램 및 오가닉 추이 생성 중...")
        insta_df = get_instagram_followers(fb_ad_account_id, date_start, date_end)
        add_ds("insta_followers", "line", "팔로워 추이", insta_df, "명", "updated_at", ["follower_count"])
        add_ds("insta_profile_visits", "line", "프로필 방문 수(주별)", insta_df, "회", "updated_at", ["profile_views"])
        profile_monthly_df = get_profile_visits_monthly(fb_ad_account_id, date_start, date_end)
        add_ds("insta_profile_visits_monthly", "line", "인스타그램 프로필 방문수 (월별)", profile_monthly_df, "회", "updated_at", ["profile_views"])
        organic_df = get_organic_data(target_id, date_start, date_end)
        add_ds("organic_trend", "line", "오가닉 조회수 추이 (주별)", organic_df, "회", "date_start", ["organic_impressions"])
        organic_monthly_df = get_organic_monthly_data(target_id, date_start, date_end)
        add_ds("organic_trend_monthly", "line", "오가닉 조회수 추이 (월별)", organic_monthly_df, "회", "date_start", ["organic_impressions"])
    else:
        print("fb_ad_account_id 미입력 – 인스타그램/오가닉 섹션 생략")

    # 4. 팔로워 인구통계 (계정 레벨)
    if fb_ad_account_id and has_follower_demographics_data(target_id, date_start, date_end):
        print("팔로워 인구통계학 데이터 생성 중...")
        followers_df = get_instagram_followers(fb_ad_account_id, date_start, date_end)
        current_followers = None
        if followers_df is not None and not followers_df.empty:
            s = followers_df["follower_count"].dropna()
            if not s.empty:
                current_followers = int(s.iloc[-1])

        gender_clean_df = get_demographics_ratio(target_id, date_start, date_end, "gender", "exclude_unknown")
        age_gender_clean_df = get_follower_age_gender_known_only(target_id, date_start, date_end)
        gender_unknown_df = get_demographics_ratio(target_id, date_start, date_end, "gender", "unknown_vs_known")
        age_known_unknown_df = get_age_known_unknown_by_age(target_id, date_start, date_end)
        follower_demo_latest_date = get_follower_demographics_latest_date(target_id, date_start, date_end)
        age_gender_distribution_df = get_follower_age_gender_distribution(target_id, date_start, date_end)

        if gender_clean_df is not None:
            datasets["gender_clean"] = {
                "chart_type": "doughnut",
                "title": "팔로워 성별·연령 구성",
                "labels": gender_clean_df["category"].astype(str).tolist(),
                "series": [{"name": "비율", "data": gender_clean_df["ratio"].astype(float).tolist()}],
                "unit": "%",
                "center_text": f"{current_followers:,}" if current_followers is not None else None,
                "center_subtext": "팔로워",
            }
        if age_gender_clean_df is not None:
            datasets["age_gender_clean"] = {
                "chart_type": "stacked_barh",
                "title": "연령대별 성별 분포 (알 수 없음 제외)",
                "labels": age_gender_clean_df["age_range"].astype(str).tolist(),
                "series": [
                    {"name": "male", "data": age_gender_clean_df["male"].astype(float).tolist()},
                    {"name": "female", "data": age_gender_clean_df["female"].astype(float).tolist()},
                ],
                "unit": "명",
            }
        if age_gender_distribution_df is not None:
            datasets["age_gender_distribution"] = {
                "chart_type": "stacked_barh",
                "title": "연령대별 팔로워 분포(성별 구성)",
                "labels": age_gender_distribution_df["age_range"].astype(str).tolist(),
                "series": [
                    {"name": "male", "data": age_gender_distribution_df["male"].astype(float).tolist()},
                    {"name": "female", "data": age_gender_distribution_df["female"].astype(float).tolist()},
                ],
                "unit": "명",
            }
        if gender_unknown_df is not None:
            unknown_ratio = None
            unknown_row = gender_unknown_df[gender_unknown_df["category"].isin(["알 수 없음", "Unknown"])]
            if not unknown_row.empty:
                unknown_ratio = float(unknown_row["ratio"].iloc[0])
            datasets["gender_unknown"] = {
                "chart_type": "doughnut",
                "title": "성별 데이터 식별 여부",
                "labels": gender_unknown_df["category"].astype(str).tolist(),
                "series": [{"name": "비율", "data": gender_unknown_df["ratio"].astype(float).tolist()}],
                "unit": "%",
                "center_text": f"{unknown_ratio:.1f}%" if unknown_ratio is not None else None,
                "center_subtext": "알 수 없음 비율",
            }
        if age_known_unknown_df is not None:
            datasets["age_known_unknown"] = {
                "chart_type": "stacked_barh",
                "title": "연령대별 성별 데이터 식별 여부 분포",
                "labels": age_known_unknown_df["age_range"].astype(str).tolist(),
                "series": [
                    {"name": "known", "data": age_known_unknown_df["known"].astype(float).tolist()},
                    {"name": "unknown", "data": age_known_unknown_df["unknown"].astype(float).tolist()},
                ],
                "unit": "명",
            }
        final_report["follower_demographics_pages"] = {
            "is_visible": True,
            "latest_date": follower_demo_latest_date,
            "titles": {
                "section_title": "팔로워 인구통계학 분석",
                "page_1_title": "성별 및 연령대별 팔로워 분포",
            },
        }
    else:
        final_report["follower_demographics_pages"] = {"is_visible": False}

    # 5. CTR 추이
    print("CTR 추이 생성 중...")
    add_ds("ctr_trend_weekly", "line", "주별 CTR 추이", get_ctr_data_by_ids(ad_ids, date_start, date_end), "%", "week_start", ["ctr"])
    add_ds("ctr_trend_monthly", "line", "월별 CTR 추이", get_ctr_monthly_data_by_ids(ad_ids, date_start, date_end), "%", "month_start", ["ctr"])

    # 6. ROAS / 구매건수
    if has_purchase_data_by_ids(ad_ids, date_start, date_end):
        print("ROAS, 구매건수 생성 중...")
        add_ds("purchase_roas_weekly", "line", "평균 ROAS (주별)", get_purchase_roas_weekly_by_ids(ad_ids, date_start, date_end), "%", "week_start", ["avg_roas"])
        add_ds("purchase_roas_monthly", "line", "평균 ROAS (월별)", get_purchase_roas_monthly_by_ids(ad_ids, date_start, date_end), "%", "month_start", ["avg_roas"])
        add_ds("purchase_count_weekly", "line", "구매전환 (주별)", get_purchase_count_weekly_by_ids(ad_ids, date_start, date_end), "건", "week_start", ["purchases"])
        add_ds("purchase_count_monthly", "line", "구매전환 (월별)", get_purchase_count_monthly_by_ids(ad_ids, date_start, date_end), "건", "month_start", ["purchases"])
        final_report["purchase_analysis_pages"] = {
            "is_visible": True,
            "titles": {"section_title": "전체 매출 데이터 분석", "page_1_title": "평균 ROAS", "page_2_title": "구매전환 건수"},
        }
    else:
        print("ROAS, 구매건수 없음...")
        final_report["purchase_analysis_pages"] = {"is_visible": False, "titles": {"section_title": "전체 매출 데이터 분석", "page_1_title": "평균 ROAS", "page_2_title": "구매전환 건수"}}

    # 7. 광고비 & 매출발생
    if has_revenue_data_by_ids(ad_ids, date_start, date_end):
        print("광고비/매출발생 데이터 생성 중...")
        add_ds("spend_revenue_weekly", "line", "광고비 & 매출발생 (주별)", get_spend_and_revenue_weekly_by_ids(ad_ids, date_start, date_end), currency_symbol, "week_start", ["spend", "revenue"], extra_meta={"show_legend": True})
        add_ds("spend_revenue_monthly", "line", "광고비 & 매출발생 (월별)", get_spend_and_revenue_monthly_by_ids(ad_ids, date_start, date_end), currency_symbol, "month_start", ["spend", "revenue"], extra_meta={"show_legend": True})
        final_report["spend_revenue_pages"] = {"is_visible": True, "titles": {"section_title": "전체 매출 데이터 분석", "page_1_title": "광고비 & 매출 발생"}}
    else:
        print("광고비/매출발생 데이터 없음...")
        final_report["spend_revenue_pages"] = {"is_visible": False}

    # 8. 구매 발생 콘텐츠
    purchase_contents_data = get_purchase_contents_pages_data_by_ids(ad_ids, date_start, date_end)
    if purchase_contents_data and purchase_contents_data.get("total_count", 0) > 0:
        print("구매 발생 콘텐츠 생성 중...")
        enriched_pages = []
        for page_items in purchase_contents_data["pages"]:
            enriched_items = []
            for item in page_items:
                detail_df = get_a_content_target_purchase_data(item["ad_ids"], date_start, date_end)
                item["target_details"] = detail_df.to_dict(orient="records") if detail_df is not None else []
                enriched_items.append(item)
            enriched_pages.append(enriched_items)
        final_report["purchase_contents_pages"] = {
            "is_visible": True,
            "title": purchase_contents_data["title"],
            "pages": enriched_pages,
            "total_count": purchase_contents_data["total_count"],
        }
    else:
        print("구매 발생 콘텐츠 없음...")
        final_report["purchase_contents_pages"] = {"is_visible": False}

    # 9. 구매 전환 히트맵
    purchase_age_gender_data = get_purchase_age_gender_heatmap_page_data_by_ids(ad_ids, date_start, date_end)
    if purchase_age_gender_data.get("is_visible"):
        heatmap_rows = purchase_age_gender_data["heatmap"]
        heatmap_df = heatmap_rows.copy()
        heatmap_df["purchases"] = pd.to_numeric(heatmap_df["purchases"], errors="coerce").fillna(0)
        valid_df = heatmap_df[heatmap_df["purchases"] > 0]
        if len(valid_df) >= 1:
            print("구매 전환 히트맵 생성 중...")
            final_report["purchase_age_gender_page"] = {
                "is_visible": True,
                "title": purchase_age_gender_data.get("title", "타겟별 구매전환"),
                "heatmap": heatmap_df.to_dict(orient="records"),
            }
        else:
            final_report["purchase_age_gender_page"] = {"is_visible": False}
    else:
        final_report["purchase_age_gender_page"] = {"is_visible": False}

    # 10. 타겟 히트맵
    print("타겟 히트맵 데이터 생성 중...")
    _, threshold = get_imp_threshold_by_ids(ad_ids, date_start, date_end)
    target_df = get_target_heatmap_by_ids(ad_ids, date_start, date_end, threshold)
    add_ds("target_heatmap", "table", "타겟별 노출 및 CTR 성과", target_df)

    purchase_heatmap_df = get_purchase_heatmap_by_ids(ad_ids, date_start, date_end)
    if purchase_heatmap_df is not None and not purchase_heatmap_df.empty:
        add_ds("purchase_heatmap", "table", "타겟별 구매전환 성과", purchase_heatmap_df)

    # 11. 키워드 분석
    print("키워드 분석 생성 중...")

    def _norm_age(v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        if isinstance(v, (list, tuple, set, np.ndarray, pd.Series)):
            items = [str(x).strip() for x in v if str(x).strip()]
            return items if items else None
        v = str(v).strip()
        return v if v else None

    def _norm_gender(v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            return v if v else None
        if isinstance(v, (list, tuple, set, np.ndarray, pd.Series)):
            items = [str(x).strip() for x in v if str(x).strip()]
            return items if items else None
        v = str(v).strip()
        return v if v else None

    target_configs = [("overall", None, None, "전체")]
    main_age_n = _norm_age(main_age)
    main_gender_n = _norm_gender(main_gender)
    avoid_age_n = _norm_age(avoid_age)
    avoid_gender_n = _norm_gender(avoid_gender)

    if main_age_n or main_gender_n:
        target_configs.append(("main", main_age_n, main_gender_n, "메인 타겟"))
    if avoid_age_n or avoid_gender_n:
        target_configs.append(("avoid", avoid_age_n, avoid_gender_n, "기피 타겟"))

    kw_futures = {}
    strat_futures = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for prefix, age, gen, label in target_configs:
            kw_futures[prefix] = executor.submit(get_raw_keyword_performance_by_ids, ad_ids, date_start, date_end, age, gen)
            strat_futures[prefix] = executor.submit(get_strategic_performance_by_ids, ad_ids, date_start, date_end, age, gen)
        kw_results = {k: f.result() for k, f in kw_futures.items()}
        strat_results = {k: f.result() for k, f in strat_futures.items()}

    for prefix, age, gen, label in target_configs:
        raw_kw_df = kw_results[prefix]
        for is_top in [True, False]:
            suffix = "top" if is_top else "bottom"
            exclude_zero_ctr = not is_top
            sorted_df = raw_kw_df.sort_values(by=["avg_ctr", "total_impressions"], ascending=[not is_top, False])
            nouns = filter_keywords_by_pos(sorted_df, "noun", exclude_zero_ctr=exclude_zero_ctr)
            add_ds(f"{prefix}_{suffix}_noun", "bar_h", f"{label} {suffix.upper()} 10 (명사)", nouns, "%", "keyword", ["ctr"])
            vas = filter_keywords_by_pos(sorted_df, "verb_adj", exclude_zero_ctr=exclude_zero_ctr)
            add_ds(f"{prefix}_{suffix}_va", "bar_h", f"{label} {suffix.upper()} 10 (형용사)", vas, "%", "keyword", ["ctr"])

        strat_df = strat_results[prefix]
        if strat_df is not None and not strat_df.empty:
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
            if not top_combos.empty:
                final_strat_df = strat_df.merge(top_combos[combo_keys], on=combo_keys, how="inner")
                final_strat_df = final_strat_df.sort_values(by=["combo_overall_ctr", "ess_1", "ess_2", "with_var_ctr"], ascending=[False, True, True, False])
                final_strat_df = final_strat_df.groupby(combo_keys, sort=False).head(8)
                add_ds(f"{prefix}_keyword_combo_detail", "table", f"{label} 상세 분석", final_strat_df)

    # 12. 콘텐츠 분석 (선택된 ad_ids를 CTR 기준으로 상위/하위 3개)
    print("콘텐츠별 성과 생성 중...")
    all_cards = get_content_cards_by_ids(ad_ids, date_start, date_end)

    top_cards = all_cards[:3]
    bottom_cards = (all_cards[-3:] if len(all_cards) > 3 else [])

    def enrich_cards(cards):
        enriched = []
        for item in cards:
            detail_df = get_a_content_target_ctr_data(item["ad_id"], date_start, date_end)
            if detail_df is not None:
                item["target_details"] = detail_df.to_dict(orient="records")
            enriched.append(item)
        return enriched

    final_report["datasets"]["content_top_analysis"] = {
        "kind": "content_card",
        "title": "성과 top 콘텐츠 분석",
        "items": enrich_cards(top_cards),
    }
    final_report["datasets"]["content_bottom_analysis"] = {
        "kind": "content_card",
        "title": "성과 bottom 콘텐츠 분석",
        "items": enrich_cards(bottom_cards),
    }

    # 13. 별첨 – 키워드 상세
    print("별첨 자료용 키워드 상세 분석 생성 중...")
    df_ess = get_essence_target_performance_by_ids(ad_ids, date_start, date_end)
    df_var = get_variable_target_performance_by_ids(ad_ids, date_start, date_end)

    def format_rows(df, col_indices):
        if df is None or df.empty:
            return []
        temp_df = df.replace({pd.NA: None, pd.NaT: None, np.nan: None})
        return temp_df.iloc[:, col_indices].values.tolist()

    def build_ranked_rows(df, col_indices, start_rank, end_rank):
        rows = format_rows(df, col_indices)
        sliced = rows[max(start_rank - 1, 0): max(end_rank, 0)]
        return [[rank] + row for rank, row in enumerate(sliced, start=start_rank)]

    def build_appendix_split_items(base_title, subtitle, headers, df, col_indices):
        items = []
        for s, e in [(1, 25), (26, 50)]:
            ranked = build_ranked_rows(df, col_indices, s, e)
            if ranked:
                items.append({"title": f"{base_title} ({s}~{e}위)", "subtitle": subtitle, "headers": headers, "rows": ranked, "footnote": "*등장 광고 수 상위 50개 기준"})
        return items

    def build_appendix_full_item(base_title, subtitle, headers, df, col_indices):
        rows = format_rows(df, col_indices)
        if not rows:
            return []
        ranked = [[i + 1] + row for i, row in enumerate(rows)]
        return [{"title": f"{base_title} (전체)", "subtitle": subtitle, "headers": headers, "rows": ranked, "footnote": "*등장한 전체 키워드 기준"}]

    appendix_items = []
    appendix_items.extend(build_appendix_full_item("많이 사용한 업종 필수 키워드 - 노출", "키워드가 가장 많이 노출된 타겟", ["랭킹", "키워드", "등장 광고 수", "최다 노출 타겟", "타겟 노출량", "노출 비중", "총 노출량"], df_ess, [0, 1, 2, 3, 4, 5]))
    appendix_items.extend(build_appendix_full_item("많이 사용한 업종 필수 키워드 - 클릭", "키워드가 가장 많이 노출된 타겟", ["랭킹", "키워드", "등장 광고 수", "최다 클릭 타겟", "타겟 클릭량", "클릭 비중", "총 클릭량"], df_ess, [0, 1, 6, 7, 8, 9]))
    appendix_items.extend(build_appendix_split_items("많이 사용한 브랜드 변수 키워드 - 노출", "키워드가 가장 많이 노출된 타겟", ["랭킹", "키워드", "등장 광고 수", "최다 노출 타겟", "타겟 노출량", "노출 비중", "총 노출량"], df_var, [0, 1, 2, 3, 4, 5]))
    appendix_items.extend(build_appendix_split_items("많이 사용한 브랜드 변수 키워드 - 클릭", "키워드가 가장 많이 노출된 타겟", ["랭킹", "키워드", "등장 광고 수", "최다 클릭 타겟", "타겟 클릭량", "클릭 비중", "총 클릭량"], df_var, [0, 1, 6, 7, 8, 9]))

    final_report["appendix_groups"] = [{"title": "", "items": appendix_items}]

    # 14. JSON 저장
    output_path = "json_reports/integrated_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4, default=str)

    print(f"✅ 선택 광고 리포트 JSON 생성 완료: {output_path}")


if __name__ == "__main__":
    run(ad_ids=[])
