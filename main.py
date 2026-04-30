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
from to_json import run as generate_json
import time


_KOREAN_RE = re.compile(r"[가-힣]")


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists() or not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)



def _parse_s3_location(url: str) -> tuple[str, str] | None:
    text = str(url or "").strip()
    if not text:
        return None
    if text.startswith("s3://"):
        rest = text[5:]
        if "/" not in rest:
            return None
        bucket, key = rest.split("/", 1)
        bucket, key = bucket.strip(), key.strip()
        return (bucket, key) if bucket and key else None

    parsed = urlparse(text)
    host = parsed.netloc.lower()
    path = unquote(parsed.path.lstrip("/"))
    if not host or not path:
        return None

    if not host.endswith("amazonaws.com"):
        return None

    host_parts = host.split(".")
    if "s3" in host_parts and host_parts[0] != "s3":
        s3_idx = host_parts.index("s3")
        bucket = ".".join(host_parts[:s3_idx]).strip()
        key = path.strip()
        return (bucket, key) if bucket and key else None

    if host_parts[0] == "s3":
        if "/" not in path:
            return None
        bucket, key = path.split("/", 1)
        bucket, key = bucket.strip(), key.strip()
        return (bucket, key) if bucket and key else None

    return None


def _safe_name(token: Any) -> str:
    text = str(token or "").strip()
    if not text:
        return ""
    return re.sub(r"[^0-9A-Za-z_.-]", "_", text)


def _materialize_content_thumbnails(items: list[dict[str, Any]], output_dir: str = "static/thumbnail") -> None:
    if not items:
        return

    _load_env_file(Path("db_update/.env"))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import boto3
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region_name=os.environ.get("AWS_REGION", "ap-northeast-2"),
        )
        has_boto3 = True
    except Exception:
        print("boto3 not installed: will use local cache only")
        s3 = None
        has_boto3 = False

    cache: dict[str, str] = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

    for item in items:
        src = str(item.get("thumbnail") or "").strip()
        if not src:
            continue

        if src in cache:
            item["thumbnail"] = cache[src]
            continue

        # (추가)이미 웹 URL이면 그대로 사용 : 구매 콘텐츠 썸네일 수집 오류
        # if src.startswith("http://") or src.startswith("https://"):
        #     item["thumbnail"] = src
        #     cache[src] = src
        #     continue

        s3_loc = _parse_s3_location(src)
        if not s3_loc:
            continue
        bucket, key = s3_loc

        key_ext = Path(key).suffix.lower()
        ext = ".jpg" if key_ext == ".jpeg" else (key_ext if key_ext in valid_exts else ".jpg")
        name_seed = _safe_name(item.get("fb_ad_id")) or hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]
        filename = f"{name_seed}{ext}"
        local_file = out_dir / filename

        # 로컬에 파일이 이미 있으면 S3 다운로드 없이 바로 사용
        if local_file.exists() and local_file.stat().st_size > 0:
            local_src = f"./{local_file.as_posix()}"
            item["thumbnail"] = local_src
            cache[src] = local_src
            continue

        if not has_boto3:
            print(f"thumbnail skipped (no boto3, no local cache): {filename}")
            continue

        try:
            s3.download_file(bucket, key, str(local_file))
        except Exception as exc:
            if not local_file.exists() or local_file.stat().st_size == 0:
                print(f"thumbnail download failed: bucket={bucket} key={key} err={exc}")
                continue

        local_src = f"./{local_file.as_posix()}"
        item["thumbnail"] = local_src
        cache[src] = local_src


def _load_report(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _top_targets(rows, metric: str, limit: int = 2, filter_low_imps: bool = False):
    """상위 타겟 순위 산출.
    filter_low_imps=True 일 때: unknown 제외 후 전체 노출수의 5% 이하 타겟을 순위에서 제외하고 footnote 반환.
    Returns: (rank_lines, footnote_text)
    """
    if not rows:
        return [], ""

    df = pd.DataFrame(rows)
    if df.empty or metric not in df.columns:
        return [], ""

    df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0)

    # CTR 필터링에서만 impressions 필요
    if filter_low_imps:
        if "impressions" not in df.columns:
            return [], ""

        df["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0)

        # unknown 제외
        df_filtered = df[df["gender"].astype(str).str.lower() != "unknown"].copy()
        if df_filtered.empty:
            return [], ""

        # 전체 노출수(unknown 제외) 기준 5% 임계값 산출
        total_imps = int(df_filtered["impressions"].sum())
        threshold = total_imps * 0.05

        # 5% 이하 필터링
        df_filtered = df_filtered[df_filtered["impressions"] > threshold].copy()
        footnote = f"노출수가 전체 노출수({total_imps:,})의 5%({int(threshold):,}) 이하인 타겟은 제외"
    else:
        # purchases/impressions 랭킹도 unknown 제외하고 싶으면 이 줄 유지
        df_filtered = df[df["gender"].astype(str).str.lower() != "unknown"].copy()
        footnote = ""

    if df_filtered.empty:
        return [], footnote

    # purchases는 합계 기준, 나머지는 기존처럼 행 기준 정렬
    if metric == "purchases":
        df_sorted = (
            df_filtered.groupby(["age", "gender"], as_index=False)[metric]
            .sum()
            .sort_values(by=metric, ascending=False)
        )
        footnote = "구매전환 수 합계 기준"
    else:
        df_sorted = df_filtered.sort_values(by=metric, ascending=False)

    results = []
    for idx, row in enumerate(df_sorted.head(limit).itertuples(), 1):
        age = str(getattr(row, "age", "") or "")
        gender = str(getattr(row, "gender", "") or "")

        if gender.lower() == "female":
            gender_label = "여성"
        elif gender.lower() == "male":
            gender_label = "남성"
        else:
            gender_label = gender

        label = f"{idx}위 : {age} {gender_label}".strip()
        results.append(label)

    return results, footnote


def _normalize_selector(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _has_selector(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return any(str(v).strip() for v in value if v is not None)
    return bool(str(value).strip())


def _append_da_if_predicate(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    token = value.strip()
    if not token or token.endswith("다"):
        return value
    if len(token) < 2:
        return value
    if " " in token or not _KOREAN_RE.search(token):
        return value

    if not _is_predicate_for_display(token):
        return value
    return f"{token}다"


def _is_predicate_for_display(token: str) -> bool:
    if _normalize_keyword_by_pos(token, "verb_adj") is not None:
        return True

    adverb_score = _best_adverb_score(token)
    best_pred_score = None
    for tokens, score in kiwi.analyze(f"{token}다", top_n=3):
        if not tokens:
            continue
        cand_score = float(score)
        first = next((tok for tok in tokens if tok.tag in VERB_ADJ_TAGS), None)
        if first and first.form == token:
            if best_pred_score is None or cand_score > best_pred_score:
                best_pred_score = cand_score
        if len(tokens) >= 2 and tokens[0].form + tokens[1].form == token:
            if tokens[1].tag in {"XSA", "XSV"}:
                if best_pred_score is None or cand_score > best_pred_score:
                    best_pred_score = cand_score
    if best_pred_score is None:
        return False
    if adverb_score is not None and adverb_score >= best_pred_score:
        return False
    return True


def _transform_rows_labels(value: Any) -> Any:
    if isinstance(value, list):
        return [_transform_rows_labels(item) for item in value]
    if isinstance(value, dict):
        return {k: _transform_rows_labels(v) for k, v in value.items()}
    return _append_da_if_predicate(value)


def _apply_display_predicate_suffix(report_json: dict) -> None:
    for key in ("datasets", "appendix_groups", "appendix"):
        block = report_json.get(key)
        if not isinstance(block, (dict, list)):
            continue
        report_json[key] = _walk_display_blocks(block)


def _walk_display_blocks(value: Any) -> Any:
    if isinstance(value, list):
        return [_walk_display_blocks(item) for item in value]
    if isinstance(value, dict):
        transformed = {}
        for key, item in value.items():
            if key in {"labels", "rows"}:
                transformed[key] = _transform_rows_labels(item)
            else:
                transformed[key] = _walk_display_blocks(item)
        return transformed
    return value


def _target_ctr(rows, age: Any = None, gender: Any = None):
    if not rows:
        return None

    df = pd.DataFrame(rows)
    if df.empty or "impressions" not in df.columns:
        return None

    def _selector_values(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            vals = [str(v).strip() for v in value if str(v).strip()]
            return vals
        text = str(value).strip()
        return [text] if text else []

    age_values = _selector_values(age)
    gender_values = _selector_values(gender)
    if age_values:
        df = df[df["age"].astype(str).isin(age_values)]
    if gender_values:
        df = df[df["gender"].astype(str).isin(gender_values)]
    if df.empty:
        return None

    impressions = pd.to_numeric(df["impressions"], errors="coerce").fillna(0.0)
    total_impressions = float(impressions.sum())
    if total_impressions <= 0:
        return None

    if "clicks" in df.columns:
        clicks = pd.to_numeric(df["clicks"], errors="coerce").fillna(0.0)
        total_clicks = float(clicks.sum())
        return (total_clicks / total_impressions) * 100.0

    if "ctr" in df.columns:
        ctr = pd.to_numeric(df["ctr"], errors="coerce").fillna(0.0)
        estimated_clicks = impressions * ctr / 100.0
        return (float(estimated_clicks.sum()) / total_impressions) * 100.0

    return None


def _target_label(age: str | None, gender: str | None) -> str:
    def _format_part(value: Any, default_text: str) -> str:
        if value is None:
            return default_text
        if isinstance(value, (list, tuple, set)):
            items = [str(v).strip() for v in value if str(v).strip()]
            return ", ".join(items) if items else default_text
        text = str(value).strip()
        return text if text else default_text

    age_text = _format_part(age, "전체 연령")
    gender_text = _format_part(gender, "전체 성별")
    return f"{age_text} {gender_text}".strip()


def _average_series(dataset: dict):
    if not dataset:
        return None
    series = dataset.get("series") or []
    if not series:
        return None
    data = series[0].get("data") or []
    if not data:
        return None
    return sum(data) / len(data)


def _combo_cards(dataset: dict, palette: list[str] | None = None):
    rows = (dataset or {}).get("rows") or []
    if not rows:
        return []

    df = pd.DataFrame(rows)
    base_color = palette[0] if palette else "#4e73df"
    color_map = build_color_map(base_color)

    if "with_var_ctr" not in df.columns:
        return []
    df["with_var_ctr"] = pd.to_numeric(df["with_var_ctr"], errors="coerce")
    df = df.dropna(subset=["with_var_ctr"])
    if df.empty:
        return []
    
    if "combo_overall_ctr" in df.columns:
        df["combo_overall_ctr"] = pd.to_numeric(df["combo_overall_ctr"], errors="coerce")
        df = df.dropna(subset=["combo_overall_ctr"])
        df = df.sort_values(by="combo_overall_ctr", ascending=False)
    else:
        return []

    combo_keys = ["ess_1", "ess_2", "combo_overall_ctr"]
    combo_sizes = df.groupby(combo_keys, sort=False, dropna=False).size().reset_index(name="item_count")
    combo_rank = (
        combo_sizes[combo_sizes["item_count"] >= 2]
        .sort_values(by="combo_overall_ctr", ascending=False)
        .head(6)
        .reset_index(drop=True)
    )

    cards = []
    for i, row in enumerate(combo_rank.itertuples(index=False), 1):
        e1 = getattr(row, "ess_1")
        e2 = getattr(row, "ess_2")
        ctr = getattr(row, "combo_overall_ctr")
        group_df = df[
            (df["ess_1"] == e1)
            & (df["ess_2"] == e2)
            & (df["combo_overall_ctr"] == ctr)
        ].copy()
        if len(group_df) < 2:
            continue
        
        group_df["with_var_ctr"] = pd.to_numeric(group_df["with_var_ctr"], errors="coerce")
        group_df = group_df.dropna(subset=["with_var_ctr"])
        if group_df.empty:
            continue

        imps_series = pd.to_numeric(group_df.get("var_imps"), errors="coerce").fillna(1.0)

        mini_ds = {
            "kind": "bubble",
            "labels": group_df["var_keyword"].astype(str).tolist(),
            "series": [
                {"name": "CTR", "data": group_df["with_var_ctr"].tolist()},
                {"name": "Imps", "data": imps_series.tolist()},
            ],
            "unit": "%",
        }

        chart_svg = render_bubble_chart(mini_ds, color_map, compact=True, palette=palette)
        if chart_svg and "<svg" in chart_svg:
            chart_svg = chart_svg[chart_svg.find("<svg"):]

        if isinstance(ctr, (int, float)):
            ctr_text = f"{ctr:.2f}"
        else:
            ctr_text = str(ctr) if ctr is not None else "-"

        cards.append({
            "rank": i,
            "title": f"조합 {i}위 : {e1} + {e2} ({ctr_text}%)",
            "sub": "함께 쓰인 브랜드 변수 키워드별 성과",
            "image": chart_svg,
            "ctr_text": f"{ctr_text}%",
        })
        
    return cards


from playwright.sync_api import sync_playwright
# pdf 변환 함수
def export_to_pdf(html_path, output_pdf_path):
    with sync_playwright() as p:
        # 브라우저 실행 (백그라운드)
        browser = p.chromium.launch(args=["--allow-file-access-from-files", "--disable-web-security"])
        page = browser.new_page()
        
        # 1. HTML 파일 로드 (절대 경로 권장)
        import os
        file_url = f"file://{os.path.abspath(html_path)}"
        page.goto(file_url, wait_until="networkidle") # 네트워크 활동이 멈출 때까지 대기
        
        # 2. PDF 저장 설정
        page.pdf(
            path=output_pdf_path,
            print_background=True, # 배경 색상/이미지 포함 (중요!)
            prefer_css_page_size=True,
            margin={"top": "0px", "bottom": "0px", "left": "0px", "right": "0px"}
        )
        
        browser.close()
    print(f" PDF 저장 완료: {output_pdf_path}")

# 변수 지정 함수
# target_id 에 account_id
def run():
    start_time = time.time()

    config = {
        "target_id": 29,
        "fb_ad_account_id":"act_2112797419492230",
        "start":"2025-11-17",
        "end": "2026-03-29",
        "main_age": ["25-34", "35-44"],
        "main_gender": "",
        "avoid_age": "",
        "avoid_gender": "",
        "currency": ""  # ""=원화, "dollar"=달러
    }
    target_id, fb_ad_account_id = config["target_id"], config["fb_ad_account_id"]
    start, end = config["start"], config["end"]
    main_age, main_gender = config["main_age"], config["main_gender"]
    avoid_age, avoid_gender = config["avoid_age"], config["avoid_gender"]
    has_main_target = _has_selector(main_age) or _has_selector(main_gender)
    has_avoid_target = _has_selector(avoid_age) or _has_selector(avoid_gender)
    main_label = _target_label(main_age, main_gender)
    avoid_label = _target_label(avoid_age, avoid_gender)  
    currency=config["currency"]

    # 3. to_json 실행코드 (수정된 파라미터 방식)
    generate_json(target_id=target_id, fb_ad_account_id=fb_ad_account_id,\
                  start=start, end=end,\
                   main_age=main_age, main_gender=main_gender,\
                    avoid_age=avoid_age, avoid_gender=avoid_gender, currency=currency)
    
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
    # 상위 키워드·버블차트: theme_color 기반 (진한 → 밝은 순)
    THEME_CMAP = [color_map["darker"], color_map["base"], color_map["light"]]
    # 하위 키워드·avoid: 보색 기반 (진한 → 밝은 순)
    COMP_CMAP  = [comp_color_map["darker"], comp_color_map["base"], comp_color_map["light"]]
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
        "overall_bottom_noun",
        "overall_bottom_va",
        "main_bottom_noun",
        "main_bottom_va",
        "avoid_top_noun",
        "avoid_top_va",
    }

    def add_chart(key: str, dataset_key: str, **kwargs):
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
    # 구매 히트맵 추가
    purchase_heatmap_ds = datasets.get("purchase_heatmap")

    purchase_heatmap = render_dataset(
        purchase_heatmap_ds,
        color_map,
        metric="purchases"
    )

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

    # 구매 데이터 추가
    add_chart("purchase_roas_weekly", "purchase_roas_weekly")
    add_chart("purchase_roas_monthly", "purchase_roas_monthly")
    add_chart("purchase_count_weekly", "purchase_count_weekly")
    add_chart("purchase_count_monthly", "purchase_count_monthly")

    # 광고비 & 매출발생 추가
    add_chart("spend_revenue_weekly", "spend_revenue_weekly")
    add_chart("spend_revenue_monthly", "spend_revenue_monthly")

    # 팔로워 인구통계학 추가
    gender_clean_ds = datasets.get("gender_clean")

    if gender_clean_ds and gender_clean_ds.get("labels") and gender_clean_ds.get("series"):
        charts["gender_clean"] = render_follower_gender_doughnut_chart(
            gender_clean_ds, color_map
        )

    age_gender_clean_ds = datasets.get("age_gender_clean")
    if age_gender_clean_ds and age_gender_clean_ds.get("labels") and age_gender_clean_ds.get("series"):
        charts["age_clean"] = render_follower_age_gender_stacked_barh_chart(
            age_gender_clean_ds, color_map
        )

    gender_unknown_ds = datasets.get("gender_unknown")
    if gender_unknown_ds and gender_unknown_ds.get("labels") and gender_unknown_ds.get("series"):
        charts["gender_unknown"] = render_follower_gender_doughnut_chart(
            gender_unknown_ds, color_map
        )

    age_known_unknown_ds = datasets.get("age_known_unknown")
    if age_known_unknown_ds and age_known_unknown_ds.get("labels") and age_known_unknown_ds.get("series"):
        charts["age_unknown"] = render_follower_age_gender_stacked_barh_chart(
            age_known_unknown_ds, color_map
        )


    def add_table(dataset_key: str, title: str, rank_head: str, kw_head: str):
        ds = datasets.get(dataset_key)
        
        # [수정] 데이터프레임 형식이 아니라 labels/series 형식을 체크합니다.
        if not ds or "labels" not in ds or "series" not in ds:
            return None
        
        labels = ds.get("labels", [])
        # series 안의 첫 번째 요소에서 data 리스트를 가져옵니다.
        series_data = ds.get("series", [{}])[0].get("data", [])
        
        rows = []
        # labels(키워드)와 series_data(CTR 값)를 매칭 — 동률은 같은 순위, 다음 순위는 건너뜀
        rank = 1
        for i, (label, value) in enumerate(zip(labels, series_data)):
            if i > 0:
                prev_value = series_data[i - 1]
                if value != prev_value:
                    rank = i + 1   # 동률이 있으면 그 개수만큼 건너뜀
            rows.append([
                f"{rank}위",
                label,
                f"{value:.2f}%"
            ])
        
        if not rows:
            return None

        def _header_with_break(text: str) -> str:
            head = str(text)
            if "<br>" in head:
                return head
            return head.replace("(", "<br>(") if "(" in head else head

        return {
            "title": title,
            "headers": [_header_with_break(rank_head), _header_with_break(kw_head), "평균 CTR"],
            "rows": rows,
            "footnote": ""
        }
        

    # 2. 각 계층별(Overall, Main, Avoid) 테이블 묶음 생성
    # [Overall]
    o_top = [
        add_table("overall_top_noun", "전체 TOP 10 (명사)", "순위(상위)", "키워드(명사)"),
        add_table("overall_top_va", "전체 TOP 10 (형용사/동사)", "순위(상위)", "키워드(형용사/동사)")
    ]
    o_bot = [
        add_table("overall_bottom_noun", "전체 BOTTOM 10 (명사)", "순위(하위)", "키워드(명사)"),
        add_table("overall_bottom_va", "전체 BOTTOM 10 (형용사/동사)", "순위(하위)", "키워드(형용사/동사)")
    ]

    # [Main Target] - 조건부 생성
    m_top, m_bot = [], []
    if has_main_target:
        m_top = [
            add_table("main_top_noun", f"{main_label} TOP 10 (명사)", "순위(상위)", "키워드(명사)"),
            add_table("main_top_va", f"{main_label} TOP 10 (형용사/동사)", "순위(상위)", "키워드(형용사/동사)")
        ]
        m_bot = [
            add_table("main_bottom_noun", f"{main_label} BOTTOM 10 (명사)", "순위(하위)", "키워드(명사)"),
            add_table("main_bottom_va", f"{main_label} BOTTOM 10 (형용사/동사)", "순위(하위)", "키워드(형용사/동사)")
        ]

    # [Avoid Target] - 조건부 생성
    a_top, a_bot = [], []
    if has_avoid_target:
        a_top = [
            add_table("avoid_top_noun", f"{avoid_label} TOP 10 (명사)", "순위(상위)", "키워드(명사)"),
            add_table("avoid_top_va", f"{avoid_label} TOP 10 (형용사/동사)", "순위(상위)", "키워드(형용사/동사)")
        ]
        a_bot = [
            add_table("avoid_bottom_noun", f"{avoid_label} BOTTOM 10 (명사)", "순위(하위)", "키워드(명사)"),
            add_table("avoid_bottom_va", f"{avoid_label} BOTTOM 10 (형용사/동사)", "순위(하위)", "키워드(형용사/동사)")
        ]

    # 3. None 값(데이터 없음) 필터링 함수
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

    # 구매 페이지 - 조건부 생성
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
            "keyword_count": f"{summary.get('total_keywords', '-') }개",
            "ads_count": _count_text(summary.get("total_ads")),
            "contents_count": _count_text(summary.get("total_contents")),
            "keywords_count": _count_text(summary.get("total_keywords")),
            "overview_notes": [
                f"광고 {summary.get('total_ads', '-') }개",
                f"콘텐츠 {summary.get('total_contents', '-') }개",
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
        "annotations": {
            "ctr": [],
            "organic": [],
        },
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
            "overall_top_note": "*3개 이상의 콘텐츠에 등장한 단어만 표시",
            "overall_top_tables": filter_none(o_top),
            "overall_combo_pages": [
                {
                    "note": f"*3개 이상의 콘텐츠에 등장한 조합만 표시<br>*업종 필수 키워드: 동일 업종의 상위 브랜드 10개의 웹사이트에서 자주 사용된 단어"
                    f"<br>*브랜드 변수 키워드: 필수 키워드 외 콘텐츠에 활용된 단어<br><br>*계정 전체 평균 CTR: {overall_ctr}%",
                    "cards": cards,
                }
            ],
            "overall_bottom_note": "*3개 이상의 콘텐츠에 등장한 단어만 표시",
            "overall_bottom_tables": filter_none(o_bot),
            "main_target": {"title": main_label} if has_main_target else None,
            "main_top_tables": filter_none(m_top) if m_top else None,
            "main_combo_pages": [
                {
                    "note": f"*3개 이상의 콘텐츠에 등장한 조합만 표시<br>*업종 필수 키워드: 동일 업종의 상위 브랜드 10개의 웹사이트에서 자주 사용된 단어"
                    f"<br>*브랜드 변수 키워드: 필수 키워드 외 콘텐츠에 활용된 단어<br><br>*{main_label} 평균 CTR: {main_ctr}%",
                    "cards": cards_main,
                }
            ] if has_main_target else None,
            "main_bottom_tables": filter_none(m_bot) if m_bot else None,
            "avoid_target": {"title": avoid_label} if has_avoid_target else None,
            "avoid_top_tables":filter_none(a_top) if a_top else None,
            "avoid_combo_pages": [
                {
                    "note": f"*3개 이상의 콘텐츠에 등장한 조합만 표시<br>*업종 필수 키워드: 동일 업종의 상위 브랜드 10개의 웹사이트에서 자주 사용된 단어"
                    f"<br>*브랜드 변수 키워드: 필수 키워드 외 콘텐츠에 활용된 단어<br><br>*{avoid_label} 평균 CTR: {avoid_ctr}%",
                    "cards": cards_avoid,
                }
            ] if has_avoid_target else None,
            "avoid_bottom_tables":filter_none(a_bot) if a_bot else None
        },
        "appendix_groups": report_json.get("appendix_groups", []),
        # [
        #     {
        #         "title": "",
        #         "items": [
        #             {"title": "", "subtitle": "", "image": "", "headers": [], "rows": [[]], "footnote": ""}
        #         ],
        #     }
        # ]
        "appendix": [],
        "purchase_analysis_pages": report_json.get(    # ROAS, 구매건수 추가
            "purchase_analysis_pages",
            {"is_visible": False}
        ),
        "purchase_contents_pages": report_json.get(  # 구매 컨텐츠 추가
            "purchase_contents_pages",
            {"is_visible": False}
        ),
        "purchase_age_gender_page": report_json.get(  # 구매 전환 히트맵 추가
            "purchase_age_gender_page",
            {"is_visible": False}
        ),
                "spend_revenue_pages": report_json.get(  # 광고/매출금액 추가
            "spend_revenue_pages",
            {"is_visible": False}
        ),
                "follower_demographics_pages": report_json.get(  # 팔로워 인구통계 추가
            "follower_demographics_pages",
            {"is_visible": False}
        ),
    }

    generate_html(context)
    
    # PDF 변환 추가
    export_to_pdf("report.html", f"outputs/{acc_name}_리포트.pdf")
    
    print(f"✅ {acc_name} 리포트 생성 완료!")

    end_time = time.time()
    elapsed_time = end_time - start_time # 소요 시간(초)

    print("-" * 50)
    print(f"⏳ 총 소요 시간: {elapsed_time:.2f}초") # 소수점 2자리까지 표시
    print("-" * 50)

if __name__ == "__main__":
    run()
