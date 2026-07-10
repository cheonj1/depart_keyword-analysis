# scripts/processor.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.db_connector import get_engine
from datetime import datetime, timedelta

# .env에서 NLTK_DATA 경로 로드 후 nltk 초기화
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except Exception:
    pass

try:
    import nltk as _nltk
    _nltk_data = os.environ.get("NLTK_DATA")
    if _nltk_data and str(Path(_nltk_data).expanduser()) not in _nltk.data.path:
        _nltk.data.path.insert(0, str(Path(_nltk_data).expanduser()))
    from nltk import pos_tag as _en_pos_tag
except Exception:
    _en_pos_tag = None

# 제목 부분 : (기업명) 광고 계정

def get_account_meta(account_id):
    """account_name과 currency를 함께 반환. returns (name: str, currency: str)"""
    engine = get_engine()

    query = """
        SELECT
            brand_name[1]  AS brand_name,
            aa.name        AS account_name,
            aa.currency    AS currency
        FROM ad_accounts aa
        JOIN business_portfolios bp ON aa.business_portfolio_id = bp.id
        JOIN clients cl ON bp.client_id = cl.id
        JOIN client_info ci ON cl.id = ci.client_id
        WHERE aa.id = %(account_id)s
        LIMIT 1
    """

    df = pd.read_sql(query, engine, params={"account_id": account_id})

    if df.empty:
        return str(account_id), "KRW"

    brand_name   = df.iloc[0]["brand_name"]
    account_name = df.iloc[0]["account_name"]
    currency     = str(df.iloc[0]["currency"] or "").upper()

    name = brand_name or account_name or str(account_id)
    return name, currency


def get_account_name(account_id):
    name, _ = get_account_meta(account_id)
    return name

# ----------------------------------

# ad_performance_daily 실제 마지막 적재일
def get_last_collected_date(account_id, date_start, date_end):
    """ad_performance_daily에 실제 노출/지출이 발생한 마지막 as_of_date를 반환한다 (계정/캠페인 화이트리스트, date_start~date_end 범위 내).

    노출/지출이 전혀 없는(모두 0인) row는 실제 데이터가 아니라 ETL 상의 빈 스텁 row일 수 있으므로 제외한다.
    """
    engine = get_engine()
    query = f"""
        SELECT MAX(apd.as_of_date) AS last_date
        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        JOIN ad_sets ads ON a.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        WHERE a.account_id = {account_id}
          AND apd.as_of_date >= '{date_start}'
          AND apd.as_of_date <= '{date_end}'
          AND (apd.impressions > 0 OR apd.spend > 0)
          AND ({account_id} IN (3, 2, 26)
              OR c.name ILIKE '%%depart%%'
              OR c.name LIKE '%%디파트%%'
              OR c.name ILIKE '%%de;part%%')
    """
    df = pd.read_sql(query, engine)
    if df.empty or pd.isna(df.loc[0, "last_date"]):
        return None
    return str(df.loc[0, "last_date"])

# 총 광고 개수
def get_active_ad_count(account_id, date_start, date_end):
    """해당 기간 동안 노출이 1회라도 발생한 광고(ad_id)의 총 개수를 반환"""
    engine = get_engine()

    # COUNT(DISTINCT ad_id)를 사용하여 중복 없이 광고 개수를 셉니다.
    # 성과(ad_performance_daily) 기준으로 실제 해당 기간에 노출이 발생한 광고만 집계합니다.
    query = f"""
        SELECT COUNT(DISTINCT ad.id) as ad_count
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        WHERE ad.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26)
                OR c.name ILIKE '%%depart%%' 
                OR c.name LIKE '%%디파트%%' 
                OR c.name ILIKE '%%de;part%%')
    """

    df = pd.read_sql(query, engine)

    if not df.empty:
        return int(df.iloc[0]['ad_count'])
    return 0

# 총 콘텐츠 개수
def get_total_content_count(account_id, date_start, date_end):
    """해당 기간 동안 성과가 발생한 광고에 연결된 콘텐츠(source_ig_media_id)의 총 개수를 반환"""
    engine = get_engine()

    # get_content_period()와 동일하게 ad_performance_daily 기준으로 대상 광고를 특정합니다.
    query = f"""
        SELECT COUNT(DISTINCT ad.source_ig_media_id) as content_count
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        JOIN ig_contents ig
            ON ad.source_ig_media_id = ig.fb_ig_media_id
        WHERE ad.account_id = {account_id}
            AND ig.ig_permalink IS NOT NULL
            AND ig.ig_timestamp IS NOT NULL
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26)
                OR c.name ILIKE '%%depart%%'
                OR c.name LIKE '%%디파트%%'
                OR c.name ILIKE '%%de;part%%')
    """

    df = pd.read_sql(query, engine)

    if not df.empty:
        return int(df.iloc[0]['content_count'])
    return 0

# 광고 진행 기간
def get_ad_period(account_id, date_start, date_end):
    engine = get_engine()
    query = f"""
        SELECT
            MIN(ad.fb_created_time) AS start_date,
            MAX(ad.fb_created_time)::date AS end_date
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        WHERE ad.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26) OR
                c.name ILIKE '%%depart%%'
                OR c.name LIKE '%%디파트%%'
                OR c.name ILIKE '%%de;part%%')
    """
    df = pd.read_sql(query, engine)
    if not df.empty:
        start = df.iloc[0]['start_date']
        end = df.iloc[0]['end_date']
        return start, end
    return None, None

# 콘텐츠 진행 기간
def get_content_period(account_id, date_start, date_end):
    engine = get_engine()
    # 해당 기간에 실제 노출된 광고들의 ig_timestamp(업로드일) 범위를 반환합니다.
    # ig_timestamp 날짜로 필터링하지 않고, ad_performance_daily 기준으로 대상 광고를 특정합니다.
    query = f"""
        SELECT
            MIN(ig.ig_timestamp) AS start_date,
            MAX(ig.ig_timestamp) AS end_date
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        JOIN ig_contents ig
            ON ad.source_ig_media_id = ig.fb_ig_media_id
        WHERE ad.account_id = {account_id}
            AND ig.ig_timestamp IS NOT NULL
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26) 
                OR c.name ILIKE '%%depart%%' 
                OR c.name LIKE '%%디파트%%' 
                OR c.name ILIKE '%%de;part%%')
    """
    df = pd.read_sql(query, engine)
    if not df.empty and pd.notna(df.iloc[0]['start_date']) and pd.notna(df.iloc[0]['end_date']):
        start = df.iloc[0]['start_date'].date()
        end = df.iloc[0]['end_date'].date() # timestampz to date
        return start, end
    return None, None

# 총 키워드 개수
def get_total_keyword_count(account_id, date_start, date_end):
    engine = get_engine()
    # 성과(apd.date) 기준으로 해당 기간에 노출이 발생한 광고의 키워드만 집계합니다.
    query = f"""
        SELECT DISTINCT ak.essential_keywords, ak.variable_keywords
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        LEFT JOIN ad_keywords ak ON ad.id = ak.ad_id
        WHERE ad.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26) 
                OR c.name ILIKE '%%depart%%' 
                OR c.name LIKE '%%디파트%%' 
                OR c.name ILIKE '%%de;part%%')
    """

    df = pd.read_sql(query, engine)

    all_keywords = set()
    
    for _, row in df.iterrows():
        for col in ['essential_keywords', 'variable_keywords']:
            val = row[col]
            
            # [수정 포인트] 리스트/배열 형태여도 에러 나지 않게 검사
            if val is None:
                continue
                
            # 만약 이미 리스트(배열) 형태라면 (예: ['A', 'B'])
            if isinstance(val, (list, np.ndarray)):
                for k in val:
                    if k: all_keywords.add(str(k).strip())
                    
            # 만약 문자열 형태라면 (예: '{A,B}')
            elif isinstance(val, str):
                cleaned = val.replace('{', '').replace('}', '').strip()
                if cleaned:
                    kws = cleaned.split(',')
                    for k in kws:
                        k_strip = k.strip()
                        if k_strip:
                            all_keywords.add(k_strip)
                            
    return len(all_keywords)

# ----------------------------------

# 인스타그램 팔로워 데이터 가져오기
def get_instagram_followers(fb_ad_account_id, date_start, date_end):
    # engine = get_engine() 
    # query = f"""
    # SELECT DISTINCT ON (iid.updated_at::date)
    #     aa.account_name, 
    #     iid.updated_at, 
    #     ii.follower_count, 
    #     iid.profile_views
    # FROM ig_insights_daily iid
    # JOIN ig_account ia ON iid.ig_id = ia.ig_id
    # JOIN business_portfolio bp ON ia.business_id = bp.business_id
    # JOIN ad_account aa ON bp.business_id = aa.business_id 
    # JOIN campaign c ON aa.account_id = c.account_id
    # WHERE aa.account_id = '{account_id}'
    #     AND iid.updated_at >= '{date_start}'
    #     AND iid.updated_at <= '{date_end}'
    #     AND (c.campaign_name ILIKE '%%depart%%' OR c.campaign_name LIKE '%%디파트%%' OR c.campaign_name ILIKE '%%de;part%%')
    # ORDER BY iid.updated_at::date, iid.updated_at ASC
    # """
    # # ORDER BY의 첫 번째 기준은 DISTINCT ON과 일치해야 하며, 
    # # 그 뒤에 ASC를 붙여 가장 빠른 시점을 선택합니다.

    # df = pd.read_sql(query, engine)

    engine = get_engine() # engine_db -> engine 통합 완료
    query = f"""
    SELECT DISTINCT ON (it.as_of_date)
        aa.name AS account_name, 
        it.as_of_date AS updated_at, 
        it.followers_count AS follower_count, 
        it.profile_views
    FROM ig_insights_total it
    JOIN ig_accounts ia
        ON it.ig_id = ia.id
    JOIN ad_accounts aa 
        ON ia.id = aa.ig_account_id
    WHERE aa.fb_ad_account_id = '{fb_ad_account_id}'
        AND it.as_of_date >= '{date_start}'
        AND it.as_of_date 
            <= (DATE_TRUNC('week', '{date_end}'::date) - INTERVAL '1 day')::date
    ORDER BY it.as_of_date ASC
    """
    # 통합시 campaigns테이블명, ad_account_id컬럼명 주의 !!
    # ORDER BY의 첫 번째 기준은 DISTINCT ON과 일치해야 하며, 
    # 그 뒤에 ASC를 붙여 가장 빠른 시점을 선택합니다.

    df = pd.read_sql(query, engine)

    if df.empty:
        return None
        
    return df

def get_profile_visits_monthly(fb_ad_account_id, date_start, date_end):
    # 1. 원본 데이터 가져오기
    df = get_instagram_followers(fb_ad_account_id, date_start, date_end)

    if df is None or df.empty:
        return None
    
    if 'profile_views' not in df.columns:
        return None

    # 2. 실제 달력 연월 기준으로 그룹화 (4개 단위 아님)
    df['updated_at'] = pd.to_datetime(df['updated_at'])
    df['year_month'] = df['updated_at'].dt.to_period('M')

    # 3. 연월별 합산 (방문수는 sum, updated_at은 해당 월의 첫 날짜 사용)
    monthly_df = df.groupby('year_month', sort=True).agg(
        updated_at=('updated_at', 'first'),
        profile_views=('profile_views', 'sum')
    ).reset_index(drop=True)

    return monthly_df

# 주차별 CTR(%) 데이터 가져오기
def get_ctr_data(account_id, date_start, date_end):
    engine = get_engine()

    # 1. 쿼리: apd와 ad를 JOIN하여 account_id 기준으로 데이터 추출

    query = f"""
        SELECT
            DATE_TRUNC('week', apd.as_of_date)::date as week_start, -- 해당 주의 월요일 날짜
            SUM(clicks) as total_clicks,
            SUM(impressions) as total_impressions,
            ROUND((SUM(clicks)::numeric / NULLIF(SUM(impressions), 0)::numeric) * 100, 2) as ctr
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        LEFT JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        WHERE ad.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26) 
                OR c.name ILIKE '%%depart%%' 
                OR c.name LIKE '%%디파트%%' 
                OR c.name ILIKE '%%de;part%%')
        GROUP BY week_start
        ORDER BY week_start;
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        return None

    return df


def get_ctr_monthly_data(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        SELECT
            DATE_TRUNC('month', apd.as_of_date)::date as month_start,
            SUM(clicks) as total_clicks,
            SUM(impressions) as total_impressions,
            ROUND((SUM(clicks)::numeric / NULLIF(SUM(impressions), 0)::numeric) * 100, 2) as ctr
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        LEFT JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        WHERE ad.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26) 
                OR c.name ILIKE '%%depart%%' 
                OR c.name LIKE '%%디파트%%' 
                OR c.name ILIKE '%%de;part%%')
        GROUP BY month_start
        ORDER BY month_start;
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        return None

    return df

# 주차별 organic_impressions 데이터 가져오기
def get_organic_data(account_id, date_start, date_end):
    engine = get_engine()
    # 파라미터로 받은 기간 범위 내에 있는 주차 데이터만 가져옴
    query = f"""
        SELECT
            org.date_start, 
            org.date_end, 
            org.organic_views AS organic_impressions
        FROM ig_organic_insights org
        JOIN ig_accounts ia
            ON org.ig_id = ia.id
        JOIN business_portfolios bp
            ON ia.business_portfolio_id = bp.id
        JOIN ad_accounts aa
            ON bp.id = aa.business_portfolio_id
        WHERE aa.id = {account_id}
            AND org.date_start >= '{date_start}'
            AND org.date_start <= (
                DATE_TRUNC('week', '{date_end}'::date) - INTERVAL '1 day'
            )::date
        ORDER BY org.date_start ASC
    """
    df = pd.read_sql(query, engine)
   
    if df.empty:
        return None

    return df

def get_organic_monthly_data(account_id, date_start, date_end):
    df = get_organic_data(account_id, date_start, date_end)

    if df is None or df.empty:
        return None
    
    if 'organic_impressions' not in df.columns:
        return None

    # 실제 달력 연월 기준으로 그룹화 (4개 단위 아님)
    df['date_start'] = pd.to_datetime(df['date_start'])
    df['year_month'] = df['date_start'].dt.to_period('M')

    # 연월별 집계: 시작일은 해당 월의 첫 번째 date_start, 종료일은 마지막 date_end, 수치는 합산
    monthly_df = df.groupby('year_month', sort=True).agg(
        date_start=('date_start', 'first'),
        date_end=('date_end', 'last'),
        organic_impressions=('organic_impressions', 'sum')
    ).reset_index(drop=True)

    return monthly_df


# ----------------------------------
# 이전 분기 평균선 데이터
# ----------------------------------

def get_quarter_info(date_value):
    """date_value가 속한 (연도, 분기번호)를 반환. 보고서 '현재 기간'의 분기 라벨에 사용."""
    if isinstance(date_value, str):
        dt = datetime.strptime(date_value, "%Y-%m-%d")
    else:
        dt = date_value

    quarter = (dt.month - 1) // 3 + 1
    return dt.year, quarter

def get_prev_quarter_range(date_end):
    """
    date_end가 속한 분기의 '직전' 분기 (연도, 분기번호, 시작일, 종료일)을 반환.
    Q1(1~3월) -> 작년 Q4(10~12월)
    Q2(4~6월) -> Q1(1~3월)
    Q3(7~9월) -> Q2(4~6월)
    Q4(10~12월) -> Q3(7~9월)
    """
    if isinstance(date_end, str):
        end_dt = datetime.strptime(date_end, "%Y-%m-%d")
    else:
        end_dt = date_end

    quarter = (end_dt.month - 1) // 3 + 1

    if quarter == 1:
        prev_year, prev_quarter = end_dt.year - 1, 4
    else:
        prev_year, prev_quarter = end_dt.year, quarter - 1

    start_month = (prev_quarter - 1) * 3 + 1
    start_date = datetime(prev_year, start_month, 1).date()

    if prev_quarter == 4:
        end_date = datetime(prev_year, 12, 31).date()
    else:
        end_date = (datetime(prev_year, start_month + 3, 1) - timedelta(days=1)).date()

    return prev_year, prev_quarter, start_date, end_date


def get_prev_quarter_organic_avg(account_id, date_end):
    """이전 분기 오가닉 조회수(주별) 평균"""
    year, quarter, start_date, end_date = get_prev_quarter_range(date_end)
    df = get_organic_data(account_id, str(start_date), str(end_date))
    if df is None or df.empty or 'organic_impressions' not in df.columns:
        return None
    vals = df['organic_impressions'].dropna()
    if vals.empty:
        return None
    return {"year": year, "quarter": quarter, "avg": float(vals.mean())}


def get_prev_quarter_profile_visits_avg(fb_ad_account_id, date_end):
    """이전 분기 프로필 방문수(주별) 평균"""
    year, quarter, start_date, end_date = get_prev_quarter_range(date_end)
    df = get_instagram_followers(fb_ad_account_id, str(start_date), str(end_date))
    if df is None or df.empty or 'profile_views' not in df.columns:
        return None
    vals = df['profile_views'].dropna()
    if vals.empty:
        return None
    return {"year": year, "quarter": quarter, "avg": float(vals.mean())}


def get_prev_quarter_ctr_avg(account_id, date_end):
    """이전 분기 CTR(주별) 평균"""
    year, quarter, start_date, end_date = get_prev_quarter_range(date_end)
    df = get_ctr_data(account_id, str(start_date), str(end_date))
    if df is None or df.empty or 'ctr' not in df.columns:
        return None
    vals = df['ctr'].dropna()
    if vals.empty:
        return None
    return {"year": year, "quarter": quarter, "avg": float(vals.mean())}



# 인스타그램 프로필 방문수 데이터 가져오기

# 전체 노출 수 및 threshold 가져오기
def get_imp_threshold(account_id, date_start, date_end):
    engine = get_engine()

    # 1. 전체 노출수 및 기준값 계산 (Note용)
    total_stats_query = f"""
        SELECT SUM(impressions) as total_site_imp
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        LEFT JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        WHERE ad.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26) 
                OR c.name ILIKE '%%depart%%' 
                OR c.name LIKE '%%디파트%%' 
                OR c.name ILIKE '%%de;part%%')
    """

    total_site_imp = pd.read_sql(total_stats_query, engine).iloc[0]['total_site_imp'] or 0
    threshold = total_site_imp * 0.0005  # 0.05% 기준

    return total_site_imp, threshold

# CTR 상위 광고 3개 정보 데이터 가져오기 (임계점 이상 노출 광고)  
def get_content_ctr_data(account_id, date_start, date_end, threshold, is_top=True):
    engine = get_engine()
    
    order_direction = "DESC" if is_top else "ASC"

    # 2. 개별 광고 데이터 가져오기 (uploaded_at, ig_permalink 포함)
    
    ads_query = f"""
    SELECT
        COALESCE(ig.fb_ig_media_id, 'AD_' || ad.id::text)   AS content_key,  -- 묶음 키
        MIN(ig.fb_ig_media_id)    AS source_ig_media_id,  -- 게시물 식별자 (드릴다운 합산용)
        MIN(ad.id)                AS id,         -- 대표 광고 id (썸네일/표시용 아무거나 1개)
        MIN(ad.ad_name)           AS ad_name,    -- 대표 광고 이름
        MIN(ad.fb_ad_id)          AS fb_ad_id,
        COALESCE(MIN(ig.ig_timestamp), MIN(ad.fb_created_time)) AS uploaded_at,
        COALESCE(
            MAX(NULLIF(ig.thumbnail_url, '')),
            MAX(NULLIF(ig.media_url, '')),
            MAX(NULLIF(ad.thumb_link, ''))
        ) AS thumbnail,
        ROUND(
            (SUM(apd.clicks)::numeric / NULLIF(SUM(apd.impressions), 0)::numeric) * 100, 2
        ) AS ctr                                 -- 합산 클릭/합산 노출 → 콘텐츠당 CTR 1개
    FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
    LEFT JOIN ad_performance_daily apd ON ad.id = apd.ad_id
    LEFT JOIN ig_contents ig
        ON ad.source_ig_media_id = ig.fb_ig_media_id
    WHERE ad.account_id = {account_id}
        AND apd.as_of_date >= '{date_start}'
        AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
        AND ({account_id} IN (3, 2, 26)
            OR c.name ILIKE '%%depart%%'
            OR c.name LIKE '%%디파트%%'
            OR c.name ILIKE '%%de;part%%')
        AND ig.fb_ig_media_id is NOT NULL
    GROUP BY
        COALESCE(ig.fb_ig_media_id, 'AD_' || ad.id::text)   -- ★ 이 키 하나로 "중복만 합치기"
    HAVING SUM(apd.impressions) >= {threshold}
    ORDER BY ctr {order_direction}
    LIMIT 3;
    """
    ads_df = pd.read_sql(ads_query, engine)

    if ads_df.empty:
        return []

    # 2. 결과 가공 (딕셔너리 리스트 형태로 3개 모두 저장)
    results = []
    for _, row in ads_df.iterrows():
        thumb_val = row.get('thumbnail')
        if pd.isna(thumb_val):
            thumb_val = None
        else:
            thumb_val = str(thumb_val).strip() or None
        results.append({
            'ad_id': row['id'],
            'fb_ad_id': row.get('fb_ad_id'),
            'uploaded_at': row['uploaded_at'].date() if pd.notna(row['uploaded_at']) else None,
            'thumbnail': thumb_val,
            'ctr': row['ctr'],
            'source_ig_media_id': row.get('source_ig_media_id'),
        })

    return results # 이제 3개의 데이터가 담긴 리스트를 반환합니다.


# 반응 기반 콘텐츠 성과 (좋아요+저장+공유 합산) 상위/하위 3개
def get_content_reaction_data(account_id, date_start, date_end, is_top=True, metric='total_reaction'):
    engine = get_engine()
    order_direction = "DESC" if is_top else "ASC"

    metric_col_map = {
        'total_reaction': "COALESCE(ici.likes, 0) + COALESCE(ici.shares, 0) + COALESCE(ici.saved, 0)",
        'likes':  "ici.likes",
        'saves':  "ici.saved",
        'shares': "ici.shares",
    }
    order_expr = metric_col_map.get(metric, metric_col_map['total_reaction'])

    outer_order_map = {
        "ici.likes":  "total_likes",
        "ici.saved":  "total_saves",
        "ici.shares": "total_shares",
        # total_reaction은 복합 표현식이므로 SELECT에서 부여한 별칭 그대로 사용한다.
        "COALESCE(ici.likes, 0) + COALESCE(ici.shares, 0) + COALESCE(ici.saved, 0)": "total_reaction",
    }
    # order_expr와 일치하는 별칭이 없으면 total_reaction을 기본값으로 사용한다.
    outer_order_expr = outer_order_map.get(order_expr, "total_reaction")



    query = f"""
    SELECT
        ad_id,
        fb_ad_id,
        uploaded_at,
        caption,
        thumbnail,
        ig_media_type,
        total_likes,
        total_shares,
        total_saves,
        total_comments,
        total_reaction,
        ctr,
        content_id
    FROM (
        -- 내부 쿼리: fb_ig_media_id 기준으로 중복을 제거한다.
        -- DISTINCT ON (ig.fb_ig_media_id) 는 각 Instagram 콘텐츠에 대해
        -- ORDER BY 절에서 첫 번째로 정렬된 행 1개만 남긴다.
        -- 같은 콘텐츠를 참조하는 광고가 여러 개여도 지표 기준 최우선 광고 1개만 선택된다.
        SELECT DISTINCT ON (ig.fb_ig_media_id)
            ad.id                                           AS ad_id,
            ad.fb_ad_id,
            ig.ig_timestamp                                 AS uploaded_at,
            ig.caption                                      AS caption,
            COALESCE(NULLIF(ig.thumbnail_url, ''), NULLIF(ig.media_url, '')) AS thumbnail,
            ig.ig_media_type,
            ig.id                                           AS content_id,
            ici.likes                                       AS total_likes,
            ici.shares                                      AS total_shares,
            ici.saved                                       AS total_saves,
            ici.comments                                    AS total_comments,
            COALESCE(ici.likes, 0)
                + COALESCE(ici.shares, 0)
                + COALESCE(ici.saved, 0)                    AS total_reaction,
            (
            -- 동일 게시물(source_ig_media_id)을 참조하는 모든 광고의 클릭/노출을 합산하여 CTR 계산
            -- (단일 ad.id만 보면 같은 게시물이 여러 광고에서 돌았을 때 값이 누락됨)
            SELECT ROUND(
                (SUM(apd2.clicks)::numeric / NULLIF(SUM(apd2.impressions), 0)) * 100, 2
            )
            FROM ad_performance_daily apd2
            JOIN ads ad2      ON apd2.ad_id = ad2.id
            JOIN ad_sets ads2 ON ad2.ad_set_id = ads2.id
            JOIN campaigns c2 ON ads2.campaign_id = c2.id
            WHERE ad2.source_ig_media_id = ad.source_ig_media_id
              AND ad2.account_id = {account_id}
              AND apd2.as_of_date >= '{date_start}'
              AND apd2.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
              AND ({account_id} IN (3, 2, 26)
                  OR c2.name ILIKE '%%depart%%'
                  OR c2.name LIKE '%%디파트%%'
                  OR c2.name ILIKE '%%de;part%%')
            )                                               AS ctr
        FROM ads ad
            JOIN ad_sets ads ON ad.ad_set_id = ads.id
            JOIN campaigns c ON ads.campaign_id = c.id
            JOIN ig_contents ig
                ON ad.source_ig_media_id = ig.fb_ig_media_id
            JOIN LATERAL (
                SELECT likes, shares, saved, comments
                FROM ig_content_insights
                WHERE content_id = ig.id
                ORDER BY as_of_date DESC
                LIMIT 1
            ) ici ON true
        WHERE ad.account_id = {account_id}
            AND ig.ig_timestamp IS NOT NULL
            AND (ig.ig_timestamp AT TIME ZONE 'Asia/Seoul')::date >= '{date_start}'::date
            AND (ig.ig_timestamp AT TIME ZONE 'Asia/Seoul')::date
                <= (DATE_TRUNC('week', '{date_end}'::date) - INTERVAL '1 day')::date
            AND ({account_id} IN (3, 2, 26)
                OR c.name ILIKE '%%depart%%'
                OR c.name LIKE '%%디파트%%'
                OR c.name ILIKE '%%de;part%%')
        -- DISTINCT ON 기준 컬럼이 ORDER BY 첫 번째에 와야 한다.
        -- 그 뒤에 지표 정렬을 추가하면 동일 콘텐츠 중 지표 최우선 행이 선택된다.
        ORDER BY ig.fb_ig_media_id, {order_expr} {order_direction}
    ) deduped
    -- 중복 제거 후 원하는 지표 기준으로 재정렬하여 상위/하위 5개를 선택한다.
    ORDER BY 
        {outer_order_expr} {order_direction}, 
        uploaded_at DESC, ctr DESC, content_id ASC
    LIMIT 10;
    """

    ads_df = pd.read_sql(query, engine)
    if ads_df.empty:
        return []

    results = []
    for _, row in ads_df.iterrows():
        thumb_val = row.get('thumbnail')
        if pd.isna(thumb_val):
            thumb_val = None
        else:
            thumb_val = str(thumb_val).strip() or None

        results.append({
            'ad_id':          row['ad_id'],
            'fb_ad_id':       row.get('fb_ad_id'),
            'uploaded_at':    row['uploaded_at'].date() if pd.notna(row['uploaded_at']) else None,
            'caption':        str(row.get('caption') or '').strip(),
            'thumbnail':      thumb_val,
            'ig_media_type':  row.get('ig_media_type'),
            'total_likes':    int(row['total_likes'] or 0),
            'total_shares':   int(row['total_shares'] or 0),
            'total_saves':    int(row['total_saves'] or 0),
            'total_comments': int(row['total_comments'] or 0),
            'total_reaction': int(row['total_reaction'] or 0),
            'ctr': (None if pd.isna(row['ctr']) else float(row['ctr'])),
            'content_id': int(row['content_id']),
        })

    return results


def get_reaction_metric_avg(account_id, date_start, date_end, metric='likes'):
    """
    전체 기간(date_start ~ date_end), 전체 콘텐츠를 대상으로
    지정 반응 지표(likes / saves / shares)의 평균값을 반환한다.
    LIMIT 없이 집계하므로 get_content_reaction_data의 5개 샘플 평균과 다르다.
    """
    engine = get_engine()

    # 지표별 집계 컬럼 표현식
    metric_col_map = {
        'likes':  'ici.likes',
        'saves':  'ici.saved',
        'shares': 'ici.shares',
    }
    agg_expr = metric_col_map.get(metric, 'ici.likes')

    query = f"""
    SELECT ROUND(AVG(metric_val)::numeric, 2) AS metric_avg
    FROM (
        SELECT DISTINCT ON (ig.fb_ig_media_id)
            COALESCE({agg_expr}, 0) AS metric_val      -- agg_expr: 'ici.saved' 등 그대로
        FROM ads ad
        JOIN ad_sets t  ON ad.ad_set_id = t.id
        JOIN campaigns c ON t.campaign_id = c.id
        JOIN ig_contents ig ON ad.source_ig_media_id = ig.fb_ig_media_id
        JOIN LATERAL (
            SELECT likes, saved, shares
            FROM ig_content_insights
            WHERE content_id = ig.id
            ORDER BY as_of_date DESC
            LIMIT 1
        ) ici ON true
        WHERE ad.account_id = {account_id}
            AND ig.ig_timestamp IS NOT NULL
            AND (ig.ig_timestamp AT TIME ZONE 'Asia/Seoul')::date >= '{date_start}'::date
            AND (ig.ig_timestamp AT TIME ZONE 'Asia/Seoul')::date
                <= (DATE_TRUNC('week', '{date_end}'::date) - INTERVAL '1 day')::date
            AND ({account_id} IN (3,2,26)
                OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')
        ORDER BY ig.fb_ig_media_id       -- DISTINCT ON dedup (콘텐츠당 1행)
    ) deduped;
    """

    result = pd.read_sql(query, engine)
    if result.empty or pd.isna(result['metric_avg'].iloc[0]):
        return 0.0
    return float(result['metric_avg'].iloc[0])


def log_missing_reaction_insights(account_id, date_start, date_end):
    """
    리포트 기간 내 분석 대상 게시물(source_ig_media_id 기준) 중
    ig_content_insights에 데이터가 전혀 없는 게시물을 찾아 콘솔에 로그로 표시한다.
    """
    engine = get_engine()

    query = f"""
        SELECT DISTINCT ON (ig.fb_ig_media_id)
            ad.id AS ad_id,
            ad.source_ig_media_id,
            ig.id AS content_id,
            ig.ig_timestamp,
            ig.caption
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        JOIN ig_contents ig ON ad.source_ig_media_id = ig.fb_ig_media_id
        LEFT JOIN ig_content_insights ici
            ON ici.content_id = ig.id
        WHERE ad.account_id = {account_id}
            AND ig.ig_timestamp IS NOT NULL
            AND (ig.ig_timestamp AT TIME ZONE 'Asia/Seoul')::date >= '{date_start}'::date
            AND (ig.ig_timestamp AT TIME ZONE 'Asia/Seoul')::date
                <= (DATE_TRUNC('week', '{date_end}'::date) - INTERVAL '1 day')::date
            AND ({account_id} IN (3, 2, 26)
                OR c.name ILIKE '%%depart%%'
                OR c.name LIKE '%%디파트%%'
                OR c.name ILIKE '%%de;part%%')
            AND ici.content_id IS NULL
    """

    df = pd.read_sql(query, engine)

    missing = df.to_dict(orient='records')

    if missing:
        for row in missing:
            print(
                f"[반응 지표 데이터 누락] source_ig_media_id={row['source_ig_media_id']} "
                f"ad_id={row['ad_id']} content_id={row['content_id']} "
                f"업로드일={row['ig_timestamp']} — ig_content_insights 데이터 없음"
            )
    else:
        print("반응 지표 데이터 커버리지 확인: 누락 없음")

    return missing



# 특정 광고(또는 동일 게시물의 모든 광고)의 타겟별 CTR 데이터
def get_a_content_target_ctr_data(ad_id, date_start, date_end, account_id=None, source_ig_media_id=None):
    engine = get_engine()

    # source_ig_media_id가 주어지면 동일 게시물을 참조하는 모든 광고를 합산한다.
    # 없으면(예: 게시물 연결이 없는 콘텐츠) 기존처럼 단일 ad_id 기준으로 조회한다.
    if source_ig_media_id:
        content_filter = f"ad.source_ig_media_id = '{source_ig_media_id}'"
    else:
        content_filter = f"ad.id = {ad_id}"

    account_filter = f"AND ad.account_id = {account_id}" if account_id else ""

    query = f"""
        SELECT
            apd.age_range AS age, apd.gender,
            ROUND((SUM(apd.clicks)::numeric / NULLIF(SUM(apd.impressions), 0)::numeric) * 100, 2) as ctr
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        LEFT JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        WHERE {content_filter}
            {account_filter}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND apd.gender != 'unknown'
        GROUP BY apd.age_range, apd.gender
        ORDER BY ctr DESC;
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        return None

    return df


# 타겟별 노출/클릭 합계, ctr
def get_target_avg_imp_ctr(account_id, date_start, date_end):
    engine = get_engine()
    
    query = f"""
        SELECT 
        apd.age_range AS age, 
        apd.gender, 
        SUM(apd.impressions) AS impressions, 
        SUM(apd.clicks) AS clicks,
        -- NULLIF를 사용하여 분모(impressions)가 0이면 NULL로 처리
        -- CTR 공식은 (클릭 / 노출) * 100입니다.
        ROUND(
            (SUM(apd.clicks)::numeric / NULLIF(SUM(apd.impressions), 0)::numeric) * 100, 
            2
        ) AS ctr
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        LEFT JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        WHERE ad.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26) 
                OR c.name ILIKE '%%depart%%' 
                OR c.name LIKE '%%디파트%%' 
                OR c.name ILIKE '%%de;part%%')
        GROUP BY apd.age_range, apd.gender
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        return None

    return df

def get_target_avg_imp_ctr_threshold(account_id, date_start, date_end, threshold):
    engine = get_engine()
    
    query = f"""
        SELECT 
        apd.age_range AS age, 
        apd.gender, 
        SUM(apd.impressions) AS impressions, 
        SUM(apd.clicks) AS clicks,
        -- NULLIF를 사용하여 분모(impressions)가 0이면 NULL로 처리
        -- CTR 공식은 (클릭 / 노출) * 100입니다.
        ROUND(
            (SUM(apd.clicks)::numeric / NULLIF(SUM(apd.impressions), 0)::numeric) * 100, 
            2
        ) AS ctr
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        LEFT JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        WHERE ad.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26) 
                OR c.name ILIKE '%%depart%%' 
                OR c.name LIKE '%%디파트%%' 
                OR c.name ILIKE '%%de;part%%')
            AND apd.gender != 'unknown'
        GROUP BY apd.age_range, apd.gender
        HAVING SUM(apd.impressions) >= {threshold}
    """

    df = pd.read_sql(query, engine)
    
    if df.empty:
        return None

    return df



# 키워드 마다의 imp, click, ctr
def _to_str_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        out = []
        for v in value:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def _sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _build_target_filter(target_age=None, target_gender=None):
    clauses = []

    ages = _to_str_list(target_age)
    if ages:
        if len(ages) == 1:
            clauses.append(f"apd.age_range = {_sql_quote(ages[0])}")
        else:
            age_list = ", ".join(_sql_quote(a) for a in ages)
            clauses.append(f"apd.age_range IN ({age_list})")

    genders = _to_str_list(target_gender)
    if genders:
        mapped_genders = []
        for g in genders:
            g_low = str(g).strip().lower()
            if g_low in ['f', '여성']:
                mapped_genders.append('female')
            elif g_low in ['m', '남성']:
                mapped_genders.append('male')
            else:
                mapped_genders.append(g)
        
        if len(mapped_genders) == 1:
            clauses.append(f"apd.gender = {_sql_quote(mapped_genders[0])}")
        else:
            gender_list = ", ".join(_sql_quote(g) for g in mapped_genders)
            clauses.append(f"apd.gender IN ({gender_list})")

    return "".join(f" AND {c}" for c in clauses)


def get_raw_keyword_performance(account_id, date_start, date_end, target_age=None, target_gender=None, is_top=True):
    engine = get_engine()

    # is_top은 하위호환을 위해 파라미터는 유지하되, SQL 정렬 제거 (to_json에서 Python 정렬)
    target_filter = _build_target_filter(target_age, target_gender)

    query = f"""
        SELECT
            ek.keyword,
            COUNT(DISTINCT perf.ad_body) as doc_freq,
            SUM(perf.ad_imp) as total_impressions,
            SUM(perf.ad_clk) as total_clicks,
            ROUND((SUM(perf.ad_clk)::numeric / NULLIF(SUM(perf.ad_imp), 0)) * 100, 2) as avg_ctr
        FROM (
            -- [1단계] 조건에 맞는 광고의 성과 데이터를 ad_id별로 먼저 합산
            SELECT
                apd.ad_id,
                MAX(a.body) as ad_body,
                SUM(apd.impressions) as ad_imp,
                SUM(apd.clicks) as ad_clk
            FROM ad_performance_daily apd
            INNER JOIN ads a ON apd.ad_id = a.id
            INNER JOIN ad_sets ads ON a.ad_set_id = ads.id
            INNER JOIN campaigns c ON ads.campaign_id = c.id
            WHERE a.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'::date
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            {target_filter}
            AND ({account_id} IN (3, 2, 26) OR c.name ~* 'depart|디파트|de;part')
            AND apd.gender != 'unknown'
            AND apd.ad_id IS NOT NULL
            GROUP BY apd.ad_id
        ) perf
        INNER JOIN (
            -- [2단계] 키워드만 따로 추출 (필요한 광고 ID에 대해서만)
            SELECT DISTINCT
                ak.ad_id,
                CASE
                    WHEN REPLACE(REPLACE(TRIM(k.keyword), ' ', ''), ',', '') IN ('브라키오', '브라', '키오') THEN '브라키오'
                    WHEN REPLACE(REPLACE(TRIM(k.keyword), ' ', ''), ',', '') = '숭하' THEN '뒤숭숭하'
                    ELSE TRIM(k.keyword)
                END AS keyword
            FROM ad_keywords ak
            CROSS JOIN LATERAL UNNEST(
                COALESCE(ak.essential_keywords, ARRAY[]::text[]) ||
                COALESCE(ak.variable_keywords, ARRAY[]::text[])
            ) AS k(keyword)
        ) ek ON perf.ad_id = ek.ad_id
        GROUP BY ek.keyword
        HAVING COUNT(DISTINCT perf.ad_body) >= 3
    """
    return pd.read_sql(query, engine)


# get_raw_keyword_performance로부터 얻은 df를 원하는 type(명,형동)으로 구분하여 나누는 함수 (10개 단어)
from kiwipiepy import Kiwi
from functools import lru_cache
kiwi = Kiwi()

NOUN_TAGS = {"NNG", "NNP"}
VERB_ADJ_TAGS = {"VA", "VV"}
ADVERB_TAGS = {"MAG", "MAJ"}
VALID_KEYWORD_TAGS = NOUN_TAGS | VERB_ADJ_TAGS
BLOCKED_KEYWORD_FORMS = {"포로"}
_VERB_MORPHEME_PREFIXES = ("었", "았", "겠", "셨", "였", "었었", "았았")
# 형용사 판별 접미사 목록
_EN_JJ_SUFFIXES = ("ful", "ous", "ive", "ible", "able", "ish", "less", "ent", "ant", "ern", "al", "y")
# 한국어 패션 광고에서 명사로 고정 사용되는 영문 어휘
_EN_FASHION_NOUN_SET = frozenset({
    "look", "wear", "outfit", "fit", "style", "mood", "basic",
    "outer", "inner", "item", "piece", "set", "top", "bottom",
    "cut", "line", "detail", "color", "texture", "fabric",
    "intelligence", "signature",
})
# NLTK 맥락 태그가 패션 명사와 동일해 자동 구분 불가한 동사/형용사 명시 목록
# (이 목록에 있는 단어는 명사 버킷에 들어가지 않도록 강제)
_EN_FORCE_VERB_ADJ = frozenset({
    "avoid", "refresh", "repaint", "longer", "shorter", "faster", "slower",
    "online", "offline", "trendy", "thank", "upgrade", "update",
})
# 콘텐츠 표기 관용어일 뿐 실제 키워드가 아니므로 분석에서 완전히 제외
_EN_BLOCKED_KEYWORDS = frozenset({"feat", "ex"})
# kiwi가 XR 어근을 NNG로 오분류해 명사 버킷에 들어가는 형용사/동사 어간 직접 지정
_KO_FORCE_VERB_ADJ = frozenset({
    "거창하", "거창",   # 거창하다(VA) — kiwi가 '거창'을 XR 어근으로 분리해 NNG 오분류
    "이르",             # 이르다(VA, '러' 불규칙) — kiwi가 VA-R/VV-R/VV-I 태그를 정확 매칭하지 못해 NNP(명사)로 오분류
    "뒤숭숭하",         # 뒤숭숭하다(VA) — kiwi가 '뒤숭숭'을 MAG/XR 어근+'하'(XSA)로 분석해 VA 태그를 못 잡아 명사/형용사 둘 다 None 처리되는 것을 방지
})

# kiwi가 '다리다'/'다리고' 활용형을 동사(옷을 다리다)로 오분석해
# _looks_like_predicate_stem이 True를 반환, 명사인 어간이 동사/형용사 버킷으로 분류되는 것을 막기 위한 예외 목록
_KO_FORCE_NOUN = frozenset({
    "다리",  # 다리(NNG, 다리/교량) — kiwi가 '다리고'를 다리다(VV)로 오분석해 동사로 오분류
})

# main.py 표시 단계에서 '다'를 붙여 재분석할 때(예: '다리'+'다'='다리다')
# 전혀 다른 동사(옷을 다리다)로 오분석되어 명사가 동사처럼 표시되는 것을 막기 위한 예외 목록
_KO_BLOCK_DA_SUFFIX = frozenset({
    "다리",  # 다리다(VV, 옷을 다리다)로 오분석 — '다리'는 항상 명사(다리)로만 표시
})

_nltk_path_ready = False

def _ensure_nltk_path():
    global _nltk_path_ready
    if _nltk_path_ready:
        return
    nltk_data = os.environ.get("NLTK_DATA", "").strip().strip("\"'")
    if nltk_data:
        import nltk as _n
        path = str(Path(nltk_data).expanduser())
        if path not in _n.data.path:
            _n.data.path.insert(0, path)
    _nltk_path_ready = True


@lru_cache(maxsize=50000)
def _keyword_pos_candidates(raw_text):
    """
    키워드 문자열에 대해 형태소 분석 후보(형태, 태그, 점수)를 반환.
    """
    text = str(raw_text).strip()
    if not text:
        return tuple()

    candidates = []
    for tokens, score in kiwi.analyze(text, top_n=5):
        first = next((tok for tok in tokens if tok.tag in VALID_KEYWORD_TAGS), None)
        if first is None:
            continue
        form = first.form.strip()
        if len(form) < 2 or form.isdigit():
            continue
        candidates.append((form, first.tag, float(score)))
    return tuple(candidates)


def _pick_best_candidate(candidates, allowed_tags):
    """
    허용 태그 집합에서 점수가 가장 높은 후보 1개를 고른다.
    """
    best = None
    for form, tag, score in candidates:
        if tag not in allowed_tags:
            continue
        if best is None or score > best[2]:
            best = (form, tag, score)
    return best


@lru_cache(maxsize=50000)
def _best_adverb_score(raw_text):
    """
    원문이 부사(MAG/MAJ)로 해석되는 후보 중 최고 점수를 반환.
    형용사/동사 점수보다 높으면 부사 오분류 가능성이 높다.
    """
    text = str(raw_text).strip()
    if not text:
        return None

    best_score = None
    for tokens, score in kiwi.analyze(text, top_n=5):
        if not tokens:
            continue
        first = tokens[0]
        if first.tag not in ADVERB_TAGS:
            continue
        form = first.form.strip()
        if len(form) < 2 or form.isdigit():
            continue
        cur = float(score)
        if best_score is None or cur > best_score:
            best_score = cur
    return best_score


@lru_cache(maxsize=50000)
def _best_verb_adj_score(raw_text):
    """VV/VA 및 불규칙 활용(VV-R, VV-I, VA-R, VA-I 등) 포함한 최선 동사/형용사 점수 반환."""
    text = str(raw_text).strip()
    if not text:
        return None
    best_score = None
    for tokens, score in kiwi.analyze(text, top_n=5):
        if not tokens:
            continue
        first = tokens[0]
        ftag = str(first.tag)
        if ftag.startswith("VV") or ftag.startswith("VA"):
            cur = float(score)
            if best_score is None or cur > best_score:
                best_score = cur
    return best_score


@lru_cache(maxsize=50000)
def _best_overall_score(raw_text):
    """kiwi.analyze top_n=5 후보 중 가장 높은 점수 반환.
    NNG 명사 점수가 이보다 5점 이상 낮으면 명사로 보지 않는다."""
    text = str(raw_text).strip()
    if not text:
        return None
    best = None
    for tokens, score in kiwi.analyze(text, top_n=5):
        if not tokens:
            continue
        cur = float(score)
        if best is None or cur > best:
            best = cur
    return best


def _is_blocked_keyword_form(form):
    token = str(form).strip()
    if not token:
        return True
    return token in BLOCKED_KEYWORD_FORMS


@lru_cache(maxsize=50000)
def _looks_like_predicate_stem(form):
    """'강하'처럼 명사/용언 어간이 겹치는 경우를 분리하기 위한 보조 판별.
    form+'다'를 재분석해 동일 어형이 VA/VV로 해석되면 용언 어간으로 본다.
    '조이다'처럼 '다' 앞에서 VV 대신 NNG+VCP로 분석되는 경우는 '고'/'면' 어미로 보완 검출."""
    stem = str(form).strip()
    if not stem:
        return False

    # top_n=1: 최선 해석만 신뢰. 하위 순위 VA 해석으로 인한 오분류 방지
    for tokens, _ in kiwi.analyze(f"{stem}다", top_n=1):
        if not tokens:
            continue
        first = next((tok for tok in tokens if tok.tag in VALID_KEYWORD_TAGS), None)
        if first and first.form == stem and first.tag in VERB_ADJ_TAGS:
            return True
        # 예: 강/XR+하/XSA+다/EF, 코디/NNG+하/XSV+다/EF, 여유/NNG+롭/XSA-I+다/EF
        if len(tokens) >= 2 and tokens[0].form + tokens[1].form == stem:
            if (tokens[0].tag in {"XR", "NNG", "NNP"}
                    and (tokens[1].tag in {"XSA", "XSV"}
                         or str(tokens[1].tag).startswith("XSA"))):
                return True

    for ending in ("고", "면"):
        for tokens, _ in kiwi.analyze(f"{stem}{ending}", top_n=1):
            if not tokens:
                continue
            first = next((tok for tok in tokens if tok.tag in VALID_KEYWORD_TAGS), None)
            if first and first.form == stem and first.tag in VERB_ADJ_TAGS:
                return True

    return False

def _normalize_keyword_by_pos(text, pos_type='noun'):
    """
    텍스트를 형태소 분석해 지정한 품사(noun / verb_adj)에 맞는 원형 토큰만 반환.
    영어 키워드(한글 없음)는 Kiwi 분석 불가 → noun으로만 통과, verb_adj는 제외.
    """
    import re as _re
    if not _re.search(r"[가-힣]", str(text)):
        cleaned = _re.sub(r"[^a-zA-Z0-9]", "", str(text)).strip()
        if len(cleaned) < 2:
            return None
        # 4-맥락 NLTK 태깅으로 영어 품사 판별
        # 단독 태깅의 NN 편향을 보정: solo/the(명사맥락)/is(형용사맥락)/they(동사맥락)
        lower = cleaned.lower()
        if lower in _EN_BLOCKED_KEYWORDS:
            return None
        try:
            _ensure_nltk_path()
            if not _en_pos_tag:
                raise RuntimeError("pos_tag unavailable")

            # 명시적 목록 우선 처리 (NLTK 맥락보다 우선)
            if lower in _EN_FASHION_NOUN_SET:
                en_tag = "NN"
            elif lower in _EN_FORCE_VERB_ADJ:
                en_tag = "JJ"
            else:
                t_solo = _en_pos_tag([lower])[0][1]
                t_the  = _en_pos_tag(["the",  lower])[1][1]
                t_is   = _en_pos_tag(["is",   lower])[1][1]
                t_they = _en_pos_tag(["they", lower])[1][1]

                s4 = len(lower) >= 4 and any(lower.endswith(s) for s in _EN_JJ_SUFFIXES)
                s5 = len(lower) >= 5 and any(lower.endswith(s) for s in _EN_JJ_SUFFIXES)

                if t_solo.startswith("VBG") and t_the.startswith("NN"):
                    # 게런드-명사 (lightning, evening): 단독=VBG지만 the맥락=NN
                    en_tag = "NN"
                elif t_solo.startswith("JJ") or t_the.startswith("JJ"):
                    # 확실한 형용사: 단독 또는 the맥락에서 JJ
                    en_tag = "JJ"
                elif t_is.startswith("JJ") and s4:
                    # is맥락 JJ + 형용사 접미사 (stylish, elegant, cozy, timeless 등)
                    en_tag = "JJ"
                elif s5 and not t_they.startswith(("VBP", "VBD")):
                    # 접미사만 있고 동사 맥락 없음 (effortless 등 드문 케이스)
                    en_tag = "JJ"
                elif t_they.startswith(("VBP", "VBD")) and not t_is.startswith("JJ"):
                    # 순수 동사: they맥락=VBP/VBD이고 is맥락≠JJ
                    # (is=JJ이면 명사-동사 겸용어 가능성 → 명사 유지)
                    en_tag = "VBP"
                else:
                    en_tag = t_solo if not t_solo.startswith("VBG") else "NN"
        except Exception:
            en_tag = "NN"
        if pos_type == 'noun':
            return cleaned if en_tag.startswith("NN") else None
        if pos_type == 'verb_adj':
            return cleaned if (en_tag.startswith("VB") or en_tag.startswith("JJ")) else None
        return None

    # 한국어 형용사/동사 강제 목록: kiwi가 후보를 아예 못 내는 경우도 있으므로
    # candidates 유무와 무관하게 최우선으로 확인한다.
    cleaned_text = str(text).strip()
    if cleaned_text in _KO_FORCE_VERB_ADJ:
        if pos_type == "noun":
            return None
        if pos_type == "verb_adj":
            return cleaned_text

    if cleaned_text in _KO_FORCE_NOUN:
        if pos_type == "noun":
            return cleaned_text
        if pos_type == "verb_adj":
            return None

    candidates = _keyword_pos_candidates(text)
    if not candidates:
        return None

    noun_best = _pick_best_candidate(candidates, NOUN_TAGS)
    verb_adj_best = _pick_best_candidate(candidates, VERB_ADJ_TAGS)

    if pos_type == "noun":
        if noun_best is None:
            return None
        # 용언 후보가 더 우세하면 명사로 강제 변환하지 않는다.
        if verb_adj_best is not None and verb_adj_best[2] >= noun_best[2]:
            return None
        # 부사 해석이 더 우세하면 명사로 분류하지 않는다 (예: 제대로→제대 오분류 방지)
        adverb_score = _best_adverb_score(text)
        if adverb_score is not None and adverb_score >= noun_best[2]:
            return None
        # 불규칙 동사 포함 동사/형용사 점수가 명사와 1.5 이내면 동사로 판단 (예: 입지)
        va_score = _best_verb_adj_score(text)
        if va_score is not None and va_score >= noun_best[2] - 1.5:
            return None
        # 명사 후보 점수가 전체 최선 점수보다 5점 이상 낮으면 명사로 보지 않는다 (예: 스럽지)
        best_overall = _best_overall_score(text)
        if best_overall is not None and noun_best[2] < best_overall - 5.0:
            return None
        noun_form = noun_best[0]
        # 원문 전체가 명사 후보로 존재하면 부분 추출보다 우선 (예: 아우터 → 아우 아님)
        if noun_form != text:
            for form, tag, _ in candidates:
                if form == text and tag in NOUN_TAGS:
                    noun_form = text
                    break
        if _is_blocked_keyword_form(noun_form):
            return None
        # 동사 시제 형태소로 시작하는 형태는 명사가 아님 (예: 었기, 았는데)
        if noun_form.startswith(_VERB_MORPHEME_PREFIXES):
            return None
        # 어간형(예: 강하)이 명사로 오인되는 케이스를 차단
        if _looks_like_predicate_stem(noun_form):
            return None
        # 조사가 붙은 형태(예: 핏과)가 NNG로 오인된 경우를 차단
        for toks, score in kiwi.analyze(noun_form, top_n=8):
            if (len(toks) >= 2
                    and str(toks[-1].tag).startswith("J")
                    and noun_form.endswith(toks[-1].form)
                    and len(noun_form) - len(toks[-1].form) >= 1
                    and score >= noun_best[2] - 4.0):
                return None
        return noun_form

    if pos_type == "verb_adj":
        adverb_score = _best_adverb_score(text)
        if verb_adj_best is not None:
            # 명사가 동급/우세한 경우엔 용언으로 쉽게 분류하지 않는다.
            # 단, 같은 어형(예: 강하)이고 실제 용언 어간으로 판별되면 허용.
            if noun_best is not None and noun_best[2] >= verb_adj_best[2]:
                if noun_best[0] != verb_adj_best[0]:
                    return None
                if not _looks_like_predicate_stem(verb_adj_best[0]):
                    return None
            # 부사 해석이 더 우세하면 형용사/동사로 분류하지 않는다.
            if adverb_score is not None and adverb_score >= verb_adj_best[2]:
                return None
            if _is_blocked_keyword_form(verb_adj_best[0]):
                return None
            return verb_adj_best[0]
        # 분석 후보에 VA/VV가 없어도, +다 재분석에서 용언 어간으로 판별되면 허용
        if noun_best is not None and _looks_like_predicate_stem(noun_best[0]):
            if adverb_score is not None and adverb_score >= noun_best[2]:
                return None
            if _is_blocked_keyword_form(noun_best[0]):
                return None
            return noun_best[0]
        return None

    return None

def filter_keywords_by_pos(df, pos_type='noun', exclude_zero_ctr=False):
    """
    pos_type: 'noun' (NNG, NNP), 'verb_adj' (VV, VA)
    exclude_zero_ctr: True면 avg_ctr <= 0 항목 제외
    """
    if df is None or df.empty:
        return None

    # 새로운 컬럼에 정제된 키워드 할당
    df['cleaned_kw'] = df['keyword'].apply(lambda x: _normalize_keyword_by_pos(x, pos_type))
    
    # 필터링 후 원형 키워드로 치환
    filtered_df = df.dropna(subset=['cleaned_kw']).copy()
    if filtered_df.empty:
        return None
    filtered_df['keyword'] = filtered_df['cleaned_kw']
    filtered_df = filtered_df.drop(columns=['cleaned_kw'])

    # 같은 원형 키워드는 합산 집계
    agg_map = {}
    if 'doc_freq' in filtered_df.columns:
        agg_map['doc_freq'] = 'sum'
    if 'total_impressions' in filtered_df.columns:
        agg_map['total_impressions'] = 'sum'
    if 'total_clicks' in filtered_df.columns:
        agg_map['total_clicks'] = 'sum'
    if 'avg_ctr' in filtered_df.columns and 'total_impressions' not in filtered_df.columns:
        agg_map['avg_ctr'] = 'mean'

    if agg_map:
        filtered_df = filtered_df.groupby('keyword', as_index=False).agg(agg_map)

    # 노출/클릭 합계가 있으면 CTR 재계산
    if {'total_clicks', 'total_impressions'}.issubset(filtered_df.columns):
        filtered_df['avg_ctr'] = np.where(
            filtered_df['total_impressions'] > 0,
            np.round((filtered_df['total_clicks'] / filtered_df['total_impressions']) * 100, 2),
            np.nan
        )

    # 입력 정렬 방향(top/bottom)을 최대한 유지
    if 'avg_ctr' in filtered_df.columns and len(filtered_df) > 1:
        orig_avg = pd.to_numeric(df.get('avg_ctr'), errors='coerce')
        orig_avg = orig_avg.dropna()
        is_ascending = False
        if len(orig_avg) >= 2:
            is_ascending = bool(orig_avg.iloc[0] <= orig_avg.iloc[-1])
        filtered_df = filtered_df.sort_values(
            by=['avg_ctr', 'total_impressions'] if 'total_impressions' in filtered_df.columns else ['avg_ctr'],
            ascending=[is_ascending, False] if 'total_impressions' in filtered_df.columns else [is_ascending]
        )

    if exclude_zero_ctr and 'avg_ctr' in filtered_df.columns:
        ctr_vals = pd.to_numeric(filtered_df['avg_ctr'], errors='coerce')
        filtered_df = filtered_df[ctr_vals > 0]

    return filtered_df.head(10)

# 전체 기간 CTR
def get_overall_ctr(account_id, date_start, date_end):
    engine = get_engine()
    
    # 1. 쿼리: apd와 ad를 JOIN하여 account_id 기준으로 데이터 추출

    query = f"""
        SELECT ROUND((SUM(apd.clicks)::numeric / NULLIF(SUM(apd.impressions), 0)::numeric) * 100, 2) as ctr
        FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        LEFT JOIN ad_performance_daily apd ON ad.id = apd.ad_id
        WHERE ad.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26) 
                OR c.name ILIKE '%%depart%%' 
                OR c.name LIKE '%%디파트%%' 
                OR c.name ILIKE '%%de;part%%')
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        return None

    return df.iloc[0]['ctr']



# 필수 키워드 조합(A+B)마다의 전체ctr성과 + 필수키워드 조합마다의 변수 키워드별마다의 ctr 성과
# essential keywords 조합

def get_strategic_performance(account_id, date_start, date_end, target_age=None, target_gender=None):
    engine = get_engine()
    
    target_filter = _build_target_filter(target_age, target_gender)

    query = f"""
        WITH ad_raw AS (
            -- 1. 광고별 필수/변수 키워드와 기초 성과를 가져옴
            SELECT 
                ad.id,
                MAX(ad.body) as ad_body,
                ak.essential_keywords,
                ak.variable_keywords,
                SUM(apd.impressions) as ad_imps,
                SUM(apd.clicks) as ad_clicks
            FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
            JOIN ad_keywords ak ON ad.id = ak.ad_id
            LEFT JOIN ad_performance_daily apd ON ad.id = apd.ad_id
            WHERE ad.account_id = {account_id}
                AND apd.as_of_date >= '{date_start}'
                AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
                {target_filter}
                AND ({account_id} IN (3, 2, 26) 
                    OR c.name ILIKE '%%depart%%' 
                    OR c.name LIKE '%%디파트%%' 
                    OR c.name ILIKE '%%de;part%%')
            GROUP BY ad.id, ak.essential_keywords, ak.variable_keywords
            HAVING array_length(ak.essential_keywords, 1) >= 2 -- 필수 키워드가 2개 이상인 것만
        ),
        raw_pairs AS (
            -- 2. 필수 키워드 리스트 내에서 가능한 모든 2개 조합(Pair) 생성
            -- SNS, 브랜드, 채널 -> (SNS, 브랜드), (SNS, 채널), (브랜드, 채널)로 확장
            SELECT 
                id,
                ad_body,
                ad_imps,
                ad_clicks,
                variable_keywords,
                NULLIF(TRIM(essential_keywords[i]), '') as ess_a,
                NULLIF(TRIM(essential_keywords[j]), '') as ess_b
            FROM ad_raw,
            LATERAL generate_series(1, array_length(essential_keywords, 1)) i,
            LATERAL generate_series(i + 1, array_length(essential_keywords, 1)) j
        ),
        combo_pairs AS (
            -- 3. 조합 순서를 정규화해 (A,B)와 (B,A)를 동일 조합으로 통합
            SELECT DISTINCT
                id,
                ad_body,
                ad_imps,
                ad_clicks,
                variable_keywords,
                LEAST(ess_a, ess_b) as ess_1,
                GREATEST(ess_a, ess_b) as ess_2
            FROM raw_pairs
            WHERE ess_a IS NOT NULL AND ess_b IS NOT NULL
        ),
        essential_agg AS (
            -- 4. 생성된 [ess_1, ess_2] 쌍을 기준으로 전체 성과 합산
            SELECT 
                ess_1, ess_2,
                COUNT(DISTINCT id) as combo_doc_freq,
                SUM(ad_imps) as total_imps,
                ROUND((SUM(ad_clicks)::numeric / NULLIF(SUM(ad_imps), 0)::numeric) * 100, 2) as combo_overall_ctr
            FROM combo_pairs
            GROUP BY ess_1, ess_2
            HAVING COUNT(DISTINCT id) >= 3 -- 3개 이상의 광고에서 발견된 조합만
        ),
        variable_agg AS (
            -- 5. 해당 조합이 포함된 광고들 내에서 변수 키워드별 성과 계산
            SELECT 
                cp.ess_1, cp.ess_2,
                vk.var_keyword,
                COUNT(DISTINCT cp.ad_body) as var_body_doc_freq,
                SUM(cp.ad_imps) as v_imps,
                SUM(cp.ad_clicks) as v_clicks
            FROM combo_pairs cp
            INNER JOIN essential_agg ea ON cp.ess_1 = ea.ess_1 AND cp.ess_2 = ea.ess_2
            CROSS JOIN LATERAL (
                SELECT DISTINCT
                    CASE
                        WHEN REPLACE(REPLACE(TRIM(v.keyword), ' ', ''), ',', '') IN ('브라키오', '브라', '키오') THEN '브라키오'
                        WHEN REPLACE(REPLACE(TRIM(v.keyword), ' ', ''), ',', '') = '숭하' THEN '뒤숭숭하'
                        ELSE TRIM(v.keyword)
                    END AS var_keyword
                FROM UNNEST(COALESCE(cp.variable_keywords, ARRAY[]::text[])) AS v(keyword)
            ) vk
            GROUP BY cp.ess_1, cp.ess_2, vk.var_keyword
            HAVING COUNT(DISTINCT cp.ad_body) >= 3
        ),
        top_essential AS (
            -- 6. Python에서 top6만 쓰므로 상위 15개로 미리 제한
            SELECT ess_1, ess_2, combo_doc_freq, combo_overall_ctr
            FROM essential_agg
            ORDER BY combo_overall_ctr DESC
            LIMIT 15
        )
        -- 7. 최종 결합 및 정렬
        SELECT
            te.ess_1, te.ess_2,
            te.combo_doc_freq,
            te.combo_overall_ctr,
            va.var_keyword,
            va.v_clicks,
            ROUND((va.v_clicks::numeric / NULLIF(va.v_imps, 0)::numeric) * 100, 2) as with_var_ctr,
            va.v_imps as var_imps
        FROM top_essential te
        JOIN variable_agg va ON te.ess_1 = va.ess_1 AND te.ess_2 = va.ess_2
        ORDER BY te.combo_overall_ctr DESC, with_var_ctr DESC
    """
    df = pd.read_sql(query, engine)
    if df is None or df.empty:
        return df

    # 버블차트용 변수 키워드는 명사(NNG/NNP)만 남긴다.
    df = df.copy()
    df["noun_keyword"] = df["var_keyword"].apply(lambda x: _normalize_keyword_by_pos(x, "noun"))
    df = df.dropna(subset=["noun_keyword"])
    if df.empty:
        return pd.DataFrame(columns=["ess_1", "ess_2", "combo_doc_freq", "combo_overall_ctr", "var_keyword", "with_var_ctr", "var_imps"])

    # 형태소 정규화로 동일 명사가 합쳐질 수 있으므로 합산 후 CTR 재계산
    df["var_keyword"] = df["noun_keyword"]
    grouped = (
        df.groupby(["ess_1", "ess_2", "combo_doc_freq", "combo_overall_ctr", "var_keyword"], as_index=False)
        .agg(v_clicks=("v_clicks", "sum"), var_imps=("var_imps", "sum"))
    )
    grouped["with_var_ctr"] = np.where(
        grouped["var_imps"] > 0,
        np.round((grouped["v_clicks"] / grouped["var_imps"]) * 100, 2),
        np.nan
    )

    grouped = grouped.sort_values(by=["combo_overall_ctr", "with_var_ctr"], ascending=[False, False])
    return grouped[["ess_1", "ess_2", "combo_doc_freq", "combo_overall_ctr", "var_keyword", "with_var_ctr", "var_imps"]]


# 별첨 필수키워드
def get_essence_target_performance(account_id, date_start, date_end):
    engine = get_engine()
    
    query = f"""
    SELECT 
    res.single_ess AS "키워드",
    res.total_ad_count AS "등장 광고 수",
    -- 노출 파트
    MAX(CASE WHEN res.imp_rank = 1 THEN res.age || ' ' || res.gender END) AS "최다 노출 타겟",
    MAX(CASE WHEN res.imp_rank = 1 THEN res.target_imp END)::bigint AS "타겟 노출량",
    ROUND(MAX(CASE WHEN res.imp_rank = 1 THEN res.target_imp END)::numeric / NULLIF(MAX(res.total_imp), 0) * 100, 1) || '%%' AS "노출 비중",
    MAX(res.total_imp)::bigint AS "총 노출량",
    -- 클릭 파트
    MAX(CASE WHEN res.clk_rank = 1 THEN res.age || ' ' || res.gender END) AS "최다 클릭 타겟",
    MAX(CASE WHEN res.clk_rank = 1 THEN res.target_clk END)::bigint AS "타겟 클릭량",
    ROUND(MAX(CASE WHEN res.clk_rank = 1 THEN res.target_clk END)::numeric / NULLIF(MAX(res.total_clk), 0) * 100, 1) || '%%' AS "클릭 비중",
    MAX(res.total_clk)::bigint AS "총 클릭량"
    FROM (
        SELECT 
            ts.single_ess, ts.age, ts.gender, ts.target_imp, ts.target_clk,
            summ.total_ad_count,
            SUM(ts.target_imp) OVER(PARTITION BY ts.single_ess) as total_imp,
            SUM(ts.target_clk) OVER(PARTITION BY ts.single_ess) as total_clk,
            RANK() OVER (PARTITION BY ts.single_ess ORDER BY ts.target_imp DESC, ts.age) as imp_rank,
            RANK() OVER (PARTITION BY ts.single_ess ORDER BY ts.target_clk DESC, ts.age) as clk_rank
        FROM (
            SELECT 
                ak_u.single_ess, p.age, p.gender,
                SUM(p.imp) as target_imp,
                SUM(p.clk) as target_clk
            FROM (
                SELECT ad_id, UNNEST(essential_keywords) as single_ess
                FROM ad_keywords
                WHERE essential_keywords IS NOT NULL AND ARRAY_LENGTH(essential_keywords, 1) > 0
            ) ak_u
            INNER JOIN (
                SELECT 
                    apd.ad_id, apd.age_range AS age, apd.gender,
                    SUM(apd.impressions) as imp, SUM(apd.clicks) as clk
                FROM ad_performance_daily apd
                INNER JOIN ads a ON apd.ad_id = a.id
            INNER JOIN ad_sets ads ON a.ad_set_id = ads.id
            INNER JOIN campaigns c ON ads.campaign_id = c.id
                WHERE a.account_id = {account_id}
                AND apd.as_of_date >= '{date_start}'::date
                AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
                AND ({account_id} IN (3, 2, 26) OR c.name ~* 'depart|디파트|de;part')
                GROUP BY 1, 2, 3
            ) p ON ak_u.ad_id = p.ad_id
            GROUP BY 1, 2, 3
        ) ts
        INNER JOIN (
            SELECT
                UNNEST(ak.essential_keywords) as single_ess,
                COUNT(DISTINCT ak.ad_id) as total_ad_count
            FROM ad_keywords ak
            INNER JOIN ads a ON ak.ad_id = a.id
            INNER JOIN ad_sets ads ON a.ad_set_id = ads.id
            INNER JOIN campaigns c ON ads.campaign_id = c.id
            INNER JOIN ad_performance_daily apd ON a.id = apd.ad_id
            WHERE a.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'::date
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND ({account_id} IN (3, 2, 26) OR c.name ~* 'depart|디파트|de;part')
            GROUP BY 1
        ) summ ON ts.single_ess = summ.single_ess
    ) res
    GROUP BY res.single_ess, res.total_ad_count
    ORDER BY "등장 광고 수" DESC, "총 노출량" DESC;
    """
    
    df = pd.read_sql(query, engine)
    return df

# 별첨 변수키워드
def get_variable_target_performance(account_id, date_start, date_end):
    engine = get_engine()
    
    query = f"""
    SELECT 
    res.single_var AS "키워드",
    res.total_ad_count AS "등장 광고 수",

    MAX(CASE WHEN res.imp_rank = 1 THEN res.age || ' ' || res.gender END) AS "최다 노출 타겟",
    MAX(CASE WHEN res.imp_rank = 1 THEN res.target_imp END)::bigint AS "타겟 노출량",
    ROUND(
        MAX(CASE WHEN res.imp_rank = 1 THEN res.target_imp END)::numeric
        / NULLIF(MAX(res.total_imp), 0) * 100,
        1
    ) || '%%' AS "노출 비중",
    MAX(res.total_imp)::bigint AS "총 노출량",

    MAX(CASE WHEN res.clk_rank = 1 THEN res.age || ' ' || res.gender END) AS "최다 클릭 타겟",
    MAX(CASE WHEN res.clk_rank = 1 THEN res.target_clk END)::bigint AS "타겟 클릭량",
    ROUND(
        MAX(CASE WHEN res.clk_rank = 1 THEN res.target_clk END)::numeric
        / NULLIF(MAX(res.total_clk), 0) * 100,
        1
    ) || '%%' AS "클릭 비중",
    MAX(res.total_clk)::bigint AS "총 클릭량"

    FROM (
        SELECT 
            ts.single_var,
            ts.age,
            ts.gender,
            ts.target_imp,
            ts.target_clk,
            summ.total_ad_count,

            SUM(ts.target_imp) OVER(PARTITION BY ts.single_var) as total_imp,
            SUM(ts.target_clk) OVER(PARTITION BY ts.single_var) as total_clk,

            RANK() OVER (
                PARTITION BY ts.single_var
                ORDER BY ts.target_imp DESC, ts.age
            ) as imp_rank,

            RANK() OVER (
                PARTITION BY ts.single_var
                ORDER BY ts.target_clk DESC, ts.age
            ) as clk_rank

        FROM (
            SELECT 
                ak_u.single_var,
                p.age,
                p.gender,
                SUM(p.imp) as target_imp,
                SUM(p.clk) as target_clk

            FROM (
                SELECT
                    ad_id,
                    UNNEST(variable_keywords) as single_var
                FROM ad_keywords
                WHERE variable_keywords IS NOT NULL
                    AND ARRAY_LENGTH(variable_keywords, 1) > 0
            ) ak_u

            INNER JOIN (
                SELECT 
                    apd.ad_id,
                    apd.age_range AS age,
                    apd.gender,

                    SUM(apd.impressions) as imp,
                    SUM(apd.clicks) as clk

                FROM ad_performance_daily apd

                INNER JOIN ads a
                    ON apd.ad_id = a.id

                INNER JOIN ad_sets ads
                    ON a.ad_set_id = ads.id

                INNER JOIN campaigns c
                    ON ads.campaign_id = c.id

                WHERE a.account_id = {account_id}
                    AND apd.as_of_date >= '{date_start}'::date
                    AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
                    AND (
                        {account_id} IN (3, 2, 26)
                        OR c.name ~* 'depart|디파트|de;part'
                    )

                GROUP BY 1, 2, 3

            ) p
                ON ak_u.ad_id = p.ad_id

            GROUP BY 1, 2, 3

        ) ts

        INNER JOIN (
            SELECT
                UNNEST(ak.variable_keywords) as single_var,
                COUNT(DISTINCT ak.ad_id) as total_ad_count

            FROM ad_keywords ak

            INNER JOIN ads a
                ON ak.ad_id = a.id

            INNER JOIN ad_sets ads
                ON a.ad_set_id = ads.id

            INNER JOIN campaigns c
                ON ads.campaign_id = c.id

            INNER JOIN ad_performance_daily apd
                ON a.id = apd.ad_id

            WHERE a.account_id = {account_id}
                AND apd.as_of_date >= '{date_start}'::date
                AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
                AND (
                    {account_id} IN (3, 2, 26)
                    OR c.name ~* 'depart|디파트|de;part'
                )

            GROUP BY 1

        ) summ
            ON ts.single_var = summ.single_var

    ) res

    GROUP BY res.single_var, res.total_ad_count

    ORDER BY "등장 광고 수" DESC, "총 노출량" DESC
    LIMIT 50;
    """
    
    df = pd.read_sql(query, engine)
    return df

# ----------------------------------
# 구매 데이터 분석 페이지용 함수들 (ROAS, 구매건수)
# ----------------------------------


def has_purchase_data(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        SELECT 1
        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        JOIN ad_sets ads ON a.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        WHERE a.account_id = {account_id}
          AND apd.as_of_date >= '{date_start}'::date
          AND apd.as_of_date <= '{date_end}'::date
          AND apd.purchase_count IS NOT NULL
          AND apd.purchase_count > 0
          AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')
        LIMIT 1
    """

    df = pd.read_sql(query, engine)
    return not df.empty


def get_purchase_roas_weekly(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        SELECT
            (DATE_TRUNC('week', apd.as_of_date) + INTERVAL '6 days')::date AS week_start,
            ROUND(
                (SUM(apd.spend * apd.purchase_roas) / NULLIF(SUM(apd.spend), 0))::numeric
            , 0) AS avg_roas
        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        JOIN ad_sets ads ON a.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        WHERE a.account_id = {account_id}
          AND apd.as_of_date >= '{date_start}'::date
          AND apd.as_of_date <= '{date_end}'::date
          AND apd.spend IS NOT NULL
          AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')
        GROUP BY (DATE_TRUNC('week', apd.as_of_date) + INTERVAL '6 days')::date
        ORDER BY week_start
    """

    df = pd.read_sql(query, engine)
    return None if df.empty else df


def get_purchase_roas_monthly(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        SELECT
            DATE_TRUNC('month', apd.as_of_date)::date AS month_start,
            ROUND(
                (SUM(apd.spend * apd.purchase_roas) / NULLIF(SUM(apd.spend), 0))::numeric
            , 0) AS avg_roas
        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        JOIN ad_sets ads ON a.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        WHERE a.account_id = {account_id}
          AND apd.as_of_date >= '{date_start}'::date
          AND apd.as_of_date <= '{date_end}'::date
          AND apd.purchase_roas IS NOT NULL
          AND apd.spend IS NOT NULL
          AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')
        GROUP BY DATE_TRUNC('month', apd.as_of_date)::date
        ORDER BY month_start
        """

    df = pd.read_sql(query, engine)
    return None if df.empty else df


def get_purchase_count_weekly(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        WITH weeks AS (
            SELECT generate_series(
                (DATE_TRUNC('week', '{date_start}'::date) + INTERVAL '6 days')::date,
                (DATE_TRUNC('week', '{date_end}'::date) + INTERVAL '6 days')::date,
                INTERVAL '7 days'
            )::date AS week_start
        ),
        agg AS (
            SELECT
                (DATE_TRUNC('week', apd.as_of_date) + INTERVAL '6 days')::date AS week_start,
                COALESCE(SUM(apd.purchase_count), 0) AS purchases

            FROM ad_performance_daily apd
            JOIN ads a ON apd.ad_id = a.id
            JOIN ad_sets ads ON a.ad_set_id = ads.id
            JOIN campaigns c ON ads.campaign_id = c.id

            WHERE a.account_id = {account_id}
                AND apd.as_of_date >= '{date_start}'::date
                AND apd.as_of_date <= '{date_end}'::date
                AND apd.purchase_count IS NOT NULL
                AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')

            GROUP BY (DATE_TRUNC('week', apd.as_of_date) + INTERVAL '6 days')::date
            ORDER BY week_start
        )
        SELECT
            w.week_start,
            COALESCE(a.purchases, 0) AS purchases
        FROM weeks w
        LEFT JOIN agg a ON a.week_start = w.week_start
        ORDER BY w.week_start
    """

    df = pd.read_sql(query, engine)
    return None if df.empty else df


def get_purchase_count_monthly(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        WITH months AS (
            SELECT generate_series(
                DATE_TRUNC('month', '{date_start}'::date)::date,
                DATE_TRUNC('month', '{date_end}'::date)::date,
                INTERVAL '1 month'
            )::date AS month_start
        ),
        agg AS (
            SELECT
                DATE_TRUNC('month', apd.as_of_date)::date AS month_start,
                COALESCE(SUM(apd.purchase_count), 0) AS purchases

            FROM ad_performance_daily apd
            JOIN ads a ON apd.ad_id = a.id
            JOIN ad_sets ads ON a.ad_set_id = ads.id
            JOIN campaigns c ON ads.campaign_id = c.id

            WHERE a.account_id = {account_id}
                AND apd.as_of_date >= '{date_start}'::date
                AND apd.as_of_date <= '{date_end}'::date
                AND apd.purchase_count IS NOT NULL
                AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')

            GROUP BY DATE_TRUNC('month', apd.as_of_date)::date
            ORDER BY month_start
        )
        SELECT
            m.month_start,
            COALESCE(a.purchases, 0) AS purchases
        FROM months m
        LEFT JOIN agg a ON a.month_start = m.month_start
        ORDER BY m.month_start
    """

    df = pd.read_sql(query, engine)
    return None if df.empty else df

# ROAS, 구매건수 페이지 구성
def get_purchase_analysis_pages_data(account_id, date_start, date_end):
    if not has_purchase_data(account_id, date_start, date_end):
        return None

    return {
        "roas_page": {
            "title": "평균 ROAS",
            "weekly": get_purchase_roas_weekly(account_id, date_start, date_end),
            "monthly": get_purchase_roas_monthly(account_id, date_start, date_end),
        },
        "purchase_page": {
            "title": "구매전환 건수",
            "weekly": get_purchase_count_weekly(account_id, date_start, date_end),
            "monthly": get_purchase_count_monthly(account_id, date_start, date_end),
        }
    }

# 구매전환 히트맵

def get_purchase_age_gender_heatmap(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        SELECT
            apd.age_range AS age,
            apd.gender,
            COALESCE(SUM(apd.purchase_count), 0) AS purchases

        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        JOIN ad_sets ads ON a.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id

        WHERE a.account_id = {account_id}
          AND apd.as_of_date >= '{date_start}'::date
          AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
          AND apd.purchase_count IS NOT NULL
          AND apd.age_range IS NOT NULL
          AND apd.gender IS NOT NULL
          AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')

        GROUP BY apd.age_range, apd.gender
        ORDER BY apd.age_range, apd.gender
    """

    df = pd.read_sql(query, engine)
    return None if df.empty else df

def get_purchase_age_gender_heatmap_page_data(account_id, date_start, date_end):
    heatmap_df = get_purchase_age_gender_heatmap(account_id, date_start, date_end)

    if heatmap_df is None or heatmap_df.empty:
        return {"is_visible": False}

    return {
        "is_visible": True,
        "title": "타겟별 구매전환",
        "heatmap": heatmap_df,
    }

# ----------------------------------
# 광고비 & 매출발생 페이지용 함수들
# ----------------------------------

def has_revenue_data(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        SELECT 1
        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        JOIN ad_sets ads ON a.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        WHERE a.account_id = {account_id}
          AND apd.as_of_date >= '{date_start}'::date
          AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
          AND apd.spend IS NOT NULL
          AND apd.purchase_roas IS NOT NULL
          AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')
        LIMIT 1
    """

    df = pd.read_sql(query, engine)
    return not df.empty


def get_spend_and_revenue_weekly(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        WITH weeks AS (
            SELECT generate_series(
                DATE_TRUNC('week', '{date_start}'::date)::date,
                DATE_TRUNC('week', '{date_end}'::date)::date,
                INTERVAL '7 days'
            )::date AS week_start
        ),
        agg AS (
            SELECT
                DATE_TRUNC('week', apd.as_of_date)::date AS week_start,
                ROUND(COALESCE(SUM(apd.spend), 0)::numeric, 0) AS spend,
                ROUND(COALESCE(SUM(apd.spend * apd.purchase_roas), 0)::numeric, 0) AS revenue

            FROM ad_performance_daily apd
            JOIN ads a ON apd.ad_id = a.id
            JOIN ad_sets ads ON a.ad_set_id = ads.id
            JOIN campaigns c ON ads.campaign_id = c.id

            WHERE a.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'::date
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND apd.spend IS NOT NULL
            AND apd.purchase_roas IS NOT NULL
            AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')

            GROUP BY DATE_TRUNC('week', apd.as_of_date)::date
            ORDER BY week_start
        )
        SELECT
            w.week_start,
            ROUND(COALESCE(a.spend, 0)::numeric, 0) AS spend,
            ROUND(COALESCE(a.revenue, 0)::numeric, 0) AS revenue
        FROM weeks w
        LEFT JOIN agg a ON a.week_start = w.week_start
        ORDER BY w.week_start
    """

    df = pd.read_sql(query, engine)
    if df.empty:
        return None

    # 광고비가 처음 발생하기 이전의 0값 구간(집행 시작 전)을 제거
    nonzero_idx = df.index[df["spend"] != 0]
    if len(nonzero_idx) > 0:
        df = df.loc[nonzero_idx[0]:].reset_index(drop=True)

    return df


def get_spend_and_revenue_monthly(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        WITH months AS (
            SELECT generate_series(
                DATE_TRUNC('month', '{date_start}'::date)::date,
                DATE_TRUNC('month', '{date_end}'::date)::date,
                INTERVAL '1 month'
            )::date AS month_start
        ),
        agg AS (
            SELECT
                DATE_TRUNC('month', apd.as_of_date)::date AS month_start,
                ROUND(COALESCE(SUM(apd.spend), 0)::numeric, 0) AS spend,
                ROUND(COALESCE(SUM(apd.spend * apd.purchase_roas), 0)::numeric, 0) AS revenue

            FROM ad_performance_daily apd
            JOIN ads a ON apd.ad_id = a.id
            JOIN ad_sets ads ON a.ad_set_id = ads.id
            JOIN campaigns c ON ads.campaign_id = c.id

            WHERE a.account_id = {account_id}
            AND apd.as_of_date >= '{date_start}'::date
            AND apd.as_of_date <= '{date_end}'::date
            AND apd.spend IS NOT NULL
            AND apd.purchase_roas IS NOT NULL
            AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')

            GROUP BY DATE_TRUNC('month', apd.as_of_date)::date
            ORDER BY month_start
        )
        SELECT
            m.month_start,
            ROUND(COALESCE(a.spend, 0)::numeric, 0) AS spend,
            ROUND(COALESCE(a.revenue, 0)::numeric, 0) AS revenue
        FROM months m
        LEFT JOIN agg a ON a.month_start = m.month_start
        ORDER BY m.month_start
    """

    df = pd.read_sql(query, engine)
    if df.empty:
        return None

    # 광고비가 처음 발생하기 이전의 0값 구간(집행 시작 전)을 제거
    nonzero_idx = df.index[df["spend"] != 0]
    if len(nonzero_idx) > 0:
        df = df.loc[nonzero_idx[0]:].reset_index(drop=True)

    return df


# ----------------------------------
# 구매 발생 콘텐츠 페이지용 함수들
# ----------------------------------

# 구매 발생 콘텐츠 여부
def has_purchase_content_data(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        SELECT 1
        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        JOIN ad_sets ads ON a.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        LEFT JOIN ig_contents ig ON a.source_ig_media_id = ig.fb_ig_media_id
        WHERE a.account_id = {account_id}
          AND a.source_ig_media_id IS NOT NULL
          AND (ig.ig_timestamp IS NULL OR (
              (ig.ig_timestamp AT TIME ZONE 'Asia/Seoul')::date >= '{date_start}'::date
              AND (ig.ig_timestamp AT TIME ZONE 'Asia/Seoul')::date <= (DATE_TRUNC('week', '{date_end}'::date) - INTERVAL '1 day')::date
          ))
          AND apd.as_of_date >= '{date_start}'::date
          AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
          AND apd.purchase_count IS NOT NULL
          AND apd.purchase_count > 0
          AND ({account_id} IN (3, 2, 26)
          OR c.name ILIKE '%%depart%%'
          OR c.name LIKE '%%디파트%%'
          OR c.name ILIKE '%%de;part%%')
        LIMIT 1
    """

    df = pd.read_sql(query, engine)
    return not df.empty


# 구매가 발생한 콘텐츠 전체 조회
def get_purchase_contents_data(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
        SELECT
            a.source_ig_media_id AS content_key,
            MIN(ig.ig_timestamp) AS uploaded_at,
            COALESCE(
                MAX(NULLIF(ig.thumbnail_url, '')),
                MAX(NULLIF(ig.media_url, '')),
                MAX(NULLIF(a.thumb_link, ''))
            ) AS thumbnail,
            STRING_AGG(DISTINCT a.ad_name, ' / ') AS ad_names,
            ARRAY_AGG(DISTINCT a.id) AS ad_ids,
            ARRAY_AGG(DISTINCT a.fb_ad_id) FILTER (WHERE a.fb_ad_id IS NOT NULL) AS fb_ad_ids,
            COALESCE(SUM(apd.purchase_count), 0) AS purchases
        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        JOIN ad_sets ads ON a.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        LEFT JOIN ig_contents ig ON a.source_ig_media_id = ig.fb_ig_media_id
        WHERE a.account_id = {account_id}
          AND a.source_ig_media_id IS NOT NULL
          AND (ig.ig_timestamp IS NULL OR (
              (ig.ig_timestamp AT TIME ZONE 'Asia/Seoul')::date >= '{date_start}'::date
              AND (ig.ig_timestamp AT TIME ZONE 'Asia/Seoul')::date <= (DATE_TRUNC('week', '{date_end}'::date) - INTERVAL '1 day')::date
          ))
          AND apd.as_of_date >= '{date_start}'::date
          AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
          AND ({account_id} IN (3, 2, 26)
            OR c.name ILIKE '%%depart%%'
            OR c.name LIKE '%%디파트%%'
            OR c.name ILIKE '%%de;part%%')
        GROUP BY a.source_ig_media_id
        HAVING COALESCE(SUM(apd.purchase_count), 0) >= 1
        ORDER BY purchases DESC, MIN(ig.ig_timestamp) DESC
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        return []

    results = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        thumb_val = row.get("thumbnail")
        thumb_val = None if pd.isna(thumb_val) else str(thumb_val).strip() or None

        uploaded_at = row.get("uploaded_at")
        uploaded_at = uploaded_at.date() if pd.notna(uploaded_at) else None

        results.append({
            "rank": rank,
            "content_key": row["content_key"],  # source_instagram_media_id
            "source_instagram_media_id": row["content_key"],
            "ad_name": row.get("ad_names"),
            "fb_ad_id": (row.get("fb_ad_ids") or [None])[0],
            "fb_ad_ids": row.get("fb_ad_ids") or [],
            "ad_ids": row.get("ad_ids") or [],
            "uploaded_at": uploaded_at,
            "thumbnail": thumb_val,
            "purchases": int(row["purchases"]) if pd.notna(row["purchases"]) else 0
        })

    return results

# 구매 발생 콘텐츠별 세부 데이터(성별/연령/건수)
def get_a_content_target_purchase_data(ad_ids, date_start, date_end):
    engine = get_engine()

    if not ad_ids:
        return None

    ad_ids_str = ",".join(str(int(ad_id)) for ad_id in ad_ids)

    query = f"""
        SELECT
            apd.age_range AS age,
            apd.gender,
            COALESCE(SUM(apd.purchase_count), 0) AS purchases
        FROM ad_performance_daily apd
        WHERE apd.ad_id IN ({ad_ids_str})
          AND apd.as_of_date >= '{date_start}'::date
          AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
        GROUP BY apd.age_range, apd.gender
        HAVING COALESCE(SUM(apd.purchase_count), 0) > 0
        ORDER BY purchases DESC, apd.age_range
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        return None

    return df
# 리스트를 4개씩 나누기
def chunk_list(data, chunk_size=4):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


# 구매 발생 콘텐츠 페이지 구성
def get_purchase_contents_pages_data(account_id, date_start, date_end):
    if not has_purchase_content_data(account_id, date_start, date_end):
        return {
            "title": "구매가 발생한 콘텐츠",
            "pages": [],
            "items": [],
            "total_count": 0
        }

    contents = get_purchase_contents_data(account_id, date_start, date_end)

    if not contents:
        return {
            "title": "구매가 발생한 콘텐츠",
            "pages": [],
            "items": [],
            "total_count": 0
        }

    return {
        "title": "구매가 발생한 콘텐츠",
        "pages": chunk_list(contents, 4),
        "items": contents,
        "total_count": len(contents)
    }

def get_purchase_total_count(account_id, date_start, date_end):
    """기간 내 총 구매전환 건수를 반환한다."""
    engine = get_engine()
    query = f"""
        SELECT COALESCE(SUM(apd.purchase_count), 0) AS total_purchases
        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        JOIN ad_sets ads ON a.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        WHERE a.account_id = {account_id}
          AND apd.as_of_date >= '{date_start}'::date
          AND apd.as_of_date <= '{date_end}'::date
          AND apd.purchase_count IS NOT NULL
          AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')
    """
    df = pd.read_sql(query, engine)
    return int(df["total_purchases"].iloc[0]) if not df.empty else 0


def get_purchase_summary_page_data(account_id, date_start, date_end):
    """총 구매전환 < 10건일 때 사용하는 요약 페이지 데이터를 반환한다."""
    engine = get_engine()

    summary_query = f"""
        SELECT
            COALESCE(SUM(apd.purchase_count), 0) AS total_purchases,
            ROUND(
                (SUM(CASE WHEN apd.spend IS NOT NULL AND apd.purchase_roas IS NOT NULL
                          THEN apd.spend * apd.purchase_roas END)
                / NULLIF(SUM(CASE WHEN apd.spend IS NOT NULL AND apd.purchase_roas IS NOT NULL
                                  THEN apd.spend END), 0))::numeric
            , 1) AS avg_roas
        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        JOIN ad_sets ads ON a.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        WHERE a.account_id = {account_id}
          AND apd.as_of_date >= '{date_start}'::date
          AND apd.as_of_date <= '{date_end}'::date
          AND apd.purchase_count IS NOT NULL
          AND apd.purchase_count > 0
          AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')
    """

    top_content_query = f"""
        SELECT
            a.source_ig_media_id,
            COALESCE(
                MAX(NULLIF(ig.thumbnail_url, '')),
                MAX(NULLIF(ig.media_url, ''))
            ) AS thumbnail,
            COALESCE(SUM(apd.purchase_count), 0) AS purchases,
            CASE
                WHEN SUM(apd.impressions) > 0
                THEN SUM(apd.clicks)::numeric * 100 / SUM(apd.impressions)
                ELSE 0
            END AS ctr
        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        JOIN ad_sets ads ON a.ad_set_id = ads.id
        JOIN campaigns c ON ads.campaign_id = c.id
        LEFT JOIN ig_contents ig ON a.source_ig_media_id = ig.fb_ig_media_id
        WHERE a.account_id = {account_id}
          AND a.source_ig_media_id IS NOT NULL
          AND apd.as_of_date >= '{date_start}'::date
          AND apd.as_of_date <= '{date_end}'::date
          AND apd.purchase_count IS NOT NULL
          AND apd.purchase_count > 0
          AND ({account_id} IN (3, 2, 26) OR c.name ILIKE '%%depart%%' OR c.name LIKE '%%디파트%%' OR c.name ILIKE '%%de;part%%')
        GROUP BY a.source_ig_media_id
        ORDER BY purchases DESC, ctr DESC
        LIMIT 1
    """

    summary_df = pd.read_sql(summary_query, engine)
    top_df = pd.read_sql(top_content_query, engine)

    total_purchases = int(summary_df["total_purchases"].iloc[0]) if not summary_df.empty else 0
    avg_roas_raw = summary_df["avg_roas"].iloc[0] if not summary_df.empty else None
    avg_roas = None if (avg_roas_raw is None or pd.isna(avg_roas_raw)) else float(avg_roas_raw)

    top_content = None
    if not top_df.empty:
        row = top_df.iloc[0]
        thumb_val = row.get("thumbnail")
        thumb_val = None if (thumb_val is None or pd.isna(thumb_val)) else str(thumb_val).strip() or None
        top_content = {
            "source_ig_media_id": row["source_ig_media_id"],
            "thumbnail": thumb_val,
            "purchases": int(row["purchases"]),
        }

    return {
        "total_purchases": total_purchases,
        "avg_roas": avg_roas,
        "top_content": top_content,
    }


# ----------------------------------
# 팔로워 인구통계학 페이지용 함수들
# 기준: fb_ad_account_id의 가장 최근 created_at 데이터
# ----------------------------------
def has_follower_demographics_data(account_id, date_start, date_end):
    engine = get_engine()

    query = """
        SELECT 1
        FROM ig_insights_demographics iid
        JOIN ig_accounts ig
          ON iid.ig_id = ig.id
        JOIN ad_accounts aa
          ON aa.ig_account_id = ig.id
        WHERE aa.id = %(account_id)s
          AND iid.as_of_date >= %(date_start)s
          AND iid.as_of_date <= %(date_end)s
        LIMIT 1
    """

    df = pd.read_sql(
        query,
        engine,
        params={
            "account_id": account_id,
            "date_start": date_start,
            "date_end": date_end,
        },
    )
    return not df.empty


def get_follower_demographics_latest_date(account_id, date_start, date_end):
    engine = get_engine()

    query = """
        SELECT MAX(iid.as_of_date) AS latest_date
        FROM ig_insights_demographics iid
        JOIN ig_accounts ig
          ON iid.ig_id = ig.id
        JOIN ad_accounts aa
          ON ig.id = aa.ig_account_id
        WHERE aa.id = %(account_id)s
          AND iid.as_of_date >= %(date_start)s
          AND iid.as_of_date <= %(date_end)s
    """

    df = pd.read_sql(
        query,
        engine,
        params={
            "account_id": account_id,
            "date_start": date_start,
            "date_end": date_end,
        },
    )

    if df.empty or pd.isna(df.loc[0, "latest_date"]):
        return None

    return str(df.loc[0, "latest_date"])


def get_demographics_ratio(account_id, date_start, date_end, dimension="gender", mode="exclude_unknown"):
    engine = get_engine()

    if dimension == "gender":
        category_expr = """
            CASE
                WHEN TRIM(UPPER(COALESCE(iid.gender, ''))) = 'F' THEN '여성'
                WHEN TRIM(UPPER(COALESCE(iid.gender, ''))) = 'M' THEN '남성'
                ELSE '알 수 없음'
            END
        """
        known_condition = "TRIM(UPPER(COALESCE(iid.gender, ''))) IN ('F', 'M')"
        unknown_condition = "TRIM(UPPER(COALESCE(iid.gender, ''))) NOT IN ('F', 'M')"

    elif dimension == "age":
        category_expr = "COALESCE(iid.age_range, 'Unknown')"
        known_condition = "COALESCE(iid.age_range, 'Unknown') <> 'Unknown'"
        unknown_condition = "COALESCE(iid.age_range, 'Unknown') = 'Unknown'"

    else:
        raise ValueError("dimension must be 'gender' or 'age'")

    if mode == "exclude_unknown":
        query = f"""
            WITH latest_dt AS (
                SELECT MAX(iid.as_of_date) AS dt
                FROM ig_insights_demographics iid
                JOIN ig_accounts ig
                  ON iid.ig_id = ig.id
                JOIN ad_accounts aa
                  ON aa.ig_account_id = ig.id
                WHERE aa.id = %(account_id)s
                  AND iid.as_of_date >= %(date_start)s
                  AND iid.as_of_date <= %(date_end)s
            ),
            base AS (
                SELECT
                    {category_expr} AS category,
                    SUM(iid.followers) AS value
                FROM ig_insights_demographics iid
                JOIN ig_accounts ig
                  ON iid.ig_id = ig.id
                JOIN ad_accounts aa
                  ON aa.ig_account_id = ig.id
                JOIN latest_dt l
                  ON iid.as_of_date = l.dt
                WHERE aa.id = %(account_id)s
                  AND {known_condition}
                GROUP BY category
            ),
            total AS (
                SELECT SUM(value) AS total FROM base
            )
            SELECT
                category,
                value,
                ROUND(value * 100.0 / NULLIF(total, 0), 1) AS ratio
            FROM base, total
        """

    elif mode == "unknown_vs_known":
        known_label = "남/여 전체" if dimension == "gender" else "연령 확인 가능"

        query = f"""
            WITH latest_dt AS (
                SELECT MAX(iid.as_of_date) AS dt
                FROM ig_insights_demographics iid
                JOIN ig_accounts ig
                  ON iid.ig_id = ig.id
                JOIN ad_accounts aa
                  ON aa.ig_account_id = ig.id
                WHERE aa.id = %(account_id)s
                  AND iid.as_of_date >= %(date_start)s
                  AND iid.as_of_date <= %(date_end)s
            ),
            base AS (
                SELECT
                    CASE
                        WHEN {unknown_condition} THEN '알 수 없음'
                        ELSE '{known_label}'
                    END AS category,
                    SUM(iid.followers) AS value
                FROM ig_insights_demographics iid
                JOIN ig_accounts ig
                  ON iid.ig_id = ig.id
                JOIN ad_accounts aa
                  ON aa.ig_account_id = ig.id
                JOIN latest_dt l
                  ON iid.as_of_date = l.dt
                WHERE aa.id = %(account_id)s
                GROUP BY category
            ),
            total AS (
                SELECT SUM(value) AS total FROM base
            )
            SELECT
                category,
                value,
                ROUND(value * 100.0 / NULLIF(total, 0), 1) AS ratio
            FROM base, total
            ORDER BY CASE category
                WHEN '{known_label}' THEN 1
                WHEN '알 수 없음' THEN 2
                ELSE 99
            END
        """

    else:
        raise ValueError("mode must be 'exclude_unknown' or 'unknown_vs_known'")

    df = pd.read_sql(
        query,
        engine,
        params={
            "account_id": account_id,
            "date_start": date_start,
            "date_end": date_end,
        },
    )

    if df.empty:
        return None

    if dimension == "age" and mode == "exclude_unknown":
        age_order = {
            "13-17": 1,
            "18-24": 2,
            "25-34": 3,
            "35-44": 4,
            "45-54": 5,
            "55-64": 6,
            "65+": 7,
        }
        df["sort_order"] = df["category"].map(age_order).fillna(99)
        df = df.sort_values("sort_order").drop(columns=["sort_order"]).reset_index(drop=True)

    return df


def get_follower_age_gender_known_only(account_id, date_start, date_end):
    engine = get_engine()

    query = """
        WITH latest_dt AS (
            SELECT MAX(iid.as_of_date) AS dt
            FROM ig_insights_demographics iid
            JOIN ig_accounts ig
              ON iid.ig_id = ig.id
            JOIN ad_accounts aa
              ON aa.ig_account_id = ig.id
            WHERE aa.id = %(account_id)s
              AND iid.as_of_date >= %(date_start)s
              AND iid.as_of_date <= %(date_end)s
        ),
        base AS (
            SELECT
                COALESCE(iid.age_range, 'Unknown') AS age_range,
                CASE
                    WHEN TRIM(UPPER(COALESCE(iid.gender, ''))) = 'M' THEN '남성'
                    WHEN TRIM(UPPER(COALESCE(iid.gender, ''))) = 'F' THEN '여성'
                END AS gender,
                SUM(iid.followers) AS value
            FROM ig_insights_demographics iid
            JOIN ig_accounts ig
              ON iid.ig_id = ig.id
            JOIN ad_accounts aa
              ON aa.ig_account_id = ig.id
            JOIN latest_dt l
              ON iid.as_of_date = l.dt
            WHERE aa.id = %(account_id)s
              AND TRIM(UPPER(COALESCE(iid.gender, ''))) IN ('M', 'F')
              AND COALESCE(iid.age_range, 'Unknown') <> 'Unknown'
            GROUP BY
                COALESCE(iid.age_range, 'Unknown'),
                gender
        )
        SELECT
            age_range,
            COALESCE(SUM(CASE WHEN gender = '남성' THEN value END), 0) AS male,
            COALESCE(SUM(CASE WHEN gender = '여성' THEN value END), 0) AS female
        FROM base
        GROUP BY age_range
        ORDER BY CASE age_range
            WHEN '13-17' THEN 1
            WHEN '18-24' THEN 2
            WHEN '25-34' THEN 3
            WHEN '35-44' THEN 4
            WHEN '45-54' THEN 5
            WHEN '55-64' THEN 6
            WHEN '65+' THEN 7
            ELSE 99
        END
    """

    df = pd.read_sql(
        query,
        engine,
        params={
            "account_id": account_id,
            "date_start": date_start,
            "date_end": date_end,
        },
    )
    return None if df.empty else df


def get_age_known_unknown_by_age(account_id, date_start, date_end):
    engine = get_engine()

    query = """
        WITH latest_dt AS (
            SELECT MAX(iid.as_of_date) AS dt
            FROM ig_insights_demographics iid
            JOIN ig_accounts ig
              ON iid.ig_id = ig.id
            JOIN ad_accounts aa
              ON aa.ig_account_id = ig.id
            WHERE aa.id = %(account_id)s
              AND iid.as_of_date >= %(date_start)s
              AND iid.as_of_date <= %(date_end)s
        ),
        base AS (
            SELECT
                COALESCE(iid.age_range, 'Unknown') AS age_range,
            SUM(CASE 
                WHEN TRIM(UPPER(COALESCE(iid.gender, ''))) IN ('M', 'F') 
                THEN iid.followers ELSE 0 END) AS known,
            SUM(CASE 
                WHEN TRIM(UPPER(COALESCE(iid.gender, ''))) NOT IN ('M', 'F') 
                THEN iid.followers ELSE 0 END) AS unknown
            FROM ig_insights_demographics iid
            JOIN ig_accounts ig
              ON iid.ig_id = ig.id
            JOIN ad_accounts aa
              ON aa.ig_account_id = ig.id
            JOIN latest_dt l
              ON iid.as_of_date = l.dt
            WHERE aa.id = %(account_id)s
              AND COALESCE(iid.age_range, 'Unknown') <> 'Unknown'
            GROUP BY COALESCE(iid.age_range, 'Unknown')
        )
        SELECT
            age_range,
            known,
            unknown
        FROM base
        ORDER BY CASE age_range
            WHEN '13-17' THEN 1
            WHEN '18-24' THEN 2
            WHEN '25-34' THEN 3
            WHEN '35-44' THEN 4
            WHEN '45-54' THEN 5
            WHEN '55-64' THEN 6
            WHEN '65+' THEN 7
            ELSE 99
        END
    """

    df = pd.read_sql(
        query,
        engine,
        params={
            "account_id": account_id,
            "date_start": date_start,
            "date_end": date_end,
        },
    )
    return None if df.empty else df

def get_follower_age_gender_distribution(account_id, date_start, date_end):
    engine = get_engine()

    query = """
        WITH latest_dt AS (
            SELECT MAX(iid.as_of_date) AS dt
            FROM ig_insights_demographics iid
            JOIN ig_accounts ia
              ON iid.ig_id = ia.id
            JOIN ad_accounts aa
              ON aa.ig_account_id = ia.id
            WHERE aa.id = %(account_id)s
              AND iid.as_of_date >= %(date_start)s
              AND iid.as_of_date <= %(date_end)s
        ),
        base AS (
            SELECT
                TRIM(COALESCE(iid.age_range, 'Unknown')) AS age_range,
                TRIM(UPPER(COALESCE(iid.gender, ''))) AS gender,
                SUM(iid.followers) AS value
            FROM ig_insights_demographics iid
            JOIN ig_accounts ia
              ON iid.ig_id = ia.id
            JOIN ad_accounts aa
              ON aa.ig_account_id = ia.id
            JOIN latest_dt l
              ON iid.as_of_date = l.dt
            WHERE aa.id = %(account_id)s
              AND TRIM(COALESCE(iid.age_range, 'Unknown')) <> 'Unknown'
            GROUP BY age_range, gender
        )
        SELECT
            age_range,

            -- 전체 (unknown 포함)
            SUM(value) AS total,

            -- 남/여만
            COALESCE(SUM(CASE WHEN gender = 'M' THEN value END), 0) AS male,
            COALESCE(SUM(CASE WHEN gender = 'F' THEN value END), 0) AS female

        FROM base
        GROUP BY age_range
        ORDER BY CASE age_range
            WHEN '13-17' THEN 1
            WHEN '18-24' THEN 2
            WHEN '25-34' THEN 3
            WHEN '35-44' THEN 4
            WHEN '45-54' THEN 5
            WHEN '55-64' THEN 6
            WHEN '65+' THEN 7
            ELSE 99
        END
    """

    df = pd.read_sql(
        query,
        engine,
        params={
            "account_id": account_id,
            "date_start": date_start,
            "date_end": date_end,
        },
    )

    return None if df.empty else df



# 타겟별(연령×성별) 광고비 분포 (unknown 제외)
def get_target_spend_distribution(account_id, date_start, date_end):
    engine = get_engine()

    query = f"""
    SELECT
        apd.age_range,
        apd.gender,
        SUM(apd.spend)                                              AS spend,
        ROUND(
            SUM(apd.spend)::numeric
            / NULLIF(SUM(SUM(apd.spend)) OVER (), 0)::numeric * 100, 1
        )                                                           AS spend_ratio,
        ROUND((SUM(apd.clicks)::numeric
            / NULLIF(SUM(apd.impressions), 0)::numeric) * 100, 2)  AS ctr,
        ROUND(SUM(apd.spend)::numeric
            / NULLIF(SUM(apd.clicks), 0)::numeric, 0)              AS cpc
    FROM ads ad
        JOIN ad_sets ads ON ad.ad_set_id = ads.id
        JOIN campaigns c  ON ads.campaign_id = c.id
        LEFT JOIN ad_performance_daily apd ON ad.id = apd.ad_id
    WHERE ad.account_id = {account_id}
        AND apd.as_of_date >= '{date_start}'
        AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
        AND apd.gender  NOT IN ('unknown', 'Unknown')
        AND apd.age_range NOT IN ('unknown', 'Unknown')
        AND ({account_id} IN (3, 2, 26)
            OR c.name ILIKE '%%depart%%'
            OR c.name LIKE '%%디파트%%'
            OR c.name ILIKE '%%de;part%%')
    GROUP BY apd.age_range, apd.gender
    ORDER BY apd.age_range, apd.gender;
    """

    df = pd.read_sql(query, engine)
    if df.empty:
        return None

    return df



# ─────────────────────────────────────────────────────────────
# CTR × 팔로우 4사분면 스캐터 데이터
# ─────────────────────────────────────────────────────────────

def get_ctr_follows_scatter_data(account_id: int, date_start: str, date_end: str) -> list[dict]:
    """
    콘텐츠별 CTR(광고 성과)과 팔로우 발생수(IG 콘텐츠 인사이트)를
    하나의 row로 결합하여 반환합니다.

    연결 경로:
        ad_accounts → campaigns → ad_sets → ads
        ads.source_ig_media_id = ig_contents.fb_ig_media_id
        ig_contents.id = ig_content_insights.content_id
    """
    engine = get_engine()

    query = """
        SELECT
            ic.id                                          AS content_id,
            ic.fb_ig_media_id,
            COALESCE(ic.thumbnail_url, ic.media_url)       AS thumbnail,
            ic.ig_permalink,
            ic.ig_media_type,
            ic.ig_timestamp,
            -- CTR: 기간 내 노출 가중 집계
            ROUND(
                (SUM(apd.clicks)::float
                 / NULLIF(SUM(apd.impressions)::float, 0) * 100
                )::numeric, 4
            )                                              AS ctr,
            -- follows: 기간 내 합계
            COALESCE(SUM(ici.follows), 0)                  AS follows
        FROM ig_contents ic
        -- 광고 성과 연결
        JOIN ads a             ON a.source_ig_media_id = ic.fb_ig_media_id
        JOIN ad_sets  ads_t    ON a.ad_set_id          = ads_t.id
        JOIN campaigns cp      ON ads_t.campaign_id    = cp.id
        JOIN ad_accounts aa    ON cp.ad_account_id     = aa.id
        JOIN ad_performance_daily apd ON apd.ad_id     = a.id
        -- IG 콘텐츠 인사이트 연결
        JOIN ig_content_insights ici  ON ici.content_id = ic.id
        WHERE aa.id = %(account_id)s
          AND apd.as_of_date BETWEEN %(date_start)s AND %(date_end)s
          AND ici.as_of_date BETWEEN %(date_start)s AND %(date_end)s
        GROUP BY
            ic.id, ic.fb_ig_media_id, ic.thumbnail_url,
            ic.media_url, ic.ig_permalink, ic.ig_media_type, ic.ig_timestamp
        HAVING SUM(apd.impressions) > 0
        ORDER BY ic.ig_timestamp DESC
    """

    df = pd.read_sql(
        query, engine,
        params={"account_id": account_id, "date_start": date_start, "date_end": date_end}
    )

    if df.empty:
        return []

    df["ctr"]     = pd.to_numeric(df["ctr"],     errors="coerce").fillna(0.0)
    df["follows"] = pd.to_numeric(df["follows"], errors="coerce").fillna(0.0)

    return df.to_dict("records")


def get_prev_quarter_ctr_follows_means(account_id: int, date_start: str) -> dict:
    """
    사분면 십자선 기준값 산출.
    date_start 기준 3개월 전 기간(이전 분기)의 CTR 중앙값과 팔로우 평균값을 반환합니다.

    Returns:
        {"ctr_mean": float, "follows_mean": float}
        데이터 부족 시 각 값은 None
    """
    from dateutil.relativedelta import relativedelta  # 표준 라이브러리 dateutil

    end_dt   = datetime.strptime(date_start, "%Y-%m-%d") - timedelta(days=1)
    start_dt = end_dt - relativedelta(months=3) + timedelta(days=1)

    prev_start = start_dt.strftime("%Y-%m-%d")
    prev_end   = end_dt.strftime("%Y-%m-%d")

    rows = get_ctr_follows_scatter_data(account_id, prev_start, prev_end)

    if not rows:
        return {"ctr_mean": None, "follows_mean": None}

    df = pd.DataFrame(rows)
    return {
        # 중앙값(median) → 평균(mean)으로 교체
        "ctr_mean":     float(df["ctr"].mean()),
        "follows_mean": float(df["follows"].mean()),
    }


# ==============================================================================
# ad_ids 기반 쿼리 함수 (_by_ids)
# account_id/campaign 필터 대신 ad_id IN (...) 로 직접 필터링
# ==============================================================================

def _ids_sql(ad_ids):
    return "(" + ", ".join(str(int(i)) for i in ad_ids) + ")"


def get_ad_meta_by_ids(ad_ids):
    """ad_ids에서 account_id, brand_name, start/end 자동 도출"""
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT
            a.account_id,
            ci.brand_name[1] AS brand_name,
            MIN(apd.as_of_date) AS start_date,
            MAX(apd.as_of_date) AS end_date
        FROM ads a
        JOIN ad_accounts aa ON a.account_id = aa.id
        JOIN business_portfolios bp ON aa.business_portfolio_id = bp.id
        JOIN clients cl ON bp.client_id = cl.id
        JOIN client_info ci ON cl.id = ci.client_id
        JOIN ad_performance_daily apd ON a.id = apd.ad_id
        WHERE a.id IN {ids}
        GROUP BY a.account_id, ci.brand_name
        LIMIT 1
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return None
    row = df.iloc[0]
    account_id = int(row["account_id"])
    brand_name = row["brand_name"]
    return {
        "account_id": account_id,
        "brand_name": str(brand_name) if brand_name else str(account_id),
        "start": str(row["start_date"]),
        "end": str(row["end_date"]),
    }


def get_active_ad_count_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT COUNT(DISTINCT apd.ad_id) AS ad_count
        FROM ad_performance_daily apd
        WHERE apd.ad_id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
    """
    df = pd.read_sql(query, engine)
    return int(df.iloc[0]["ad_count"]) if not df.empty else 0


def get_total_content_count_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT COUNT(DISTINCT a.source_ig_media_id) AS content_count
        FROM ads a
        JOIN ad_performance_daily apd ON a.id = apd.ad_id
        JOIN ig_contents ig ON a.source_ig_media_id = ig.fb_ig_media_id
        WHERE a.id IN {ids}
            AND ig.ig_permalink IS NOT NULL
            AND ig.ig_timestamp IS NOT NULL
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
    """
    df = pd.read_sql(query, engine)
    return int(df.iloc[0]["content_count"]) if not df.empty else 0


def get_ad_period_by_ids(ad_ids):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT MIN(fb_created_time) AS start_date, MAX(fb_created_time)::date AS end_date
        FROM ads
        WHERE id IN {ids}
    """
    df = pd.read_sql(query, engine)
    if not df.empty and pd.notna(df.iloc[0]["start_date"]):
        return df.iloc[0]["start_date"], df.iloc[0]["end_date"]
    return None, None


def get_content_period_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT MIN(ig.ig_timestamp) AS start_date, MAX(ig.ig_timestamp) AS end_date
        FROM ads a
        JOIN ad_performance_daily apd ON a.id = apd.ad_id
        JOIN ig_contents ig ON a.source_ig_media_id = ig.fb_ig_media_id
        WHERE a.id IN {ids}
            AND ig.ig_timestamp IS NOT NULL
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
    """
    df = pd.read_sql(query, engine)
    if not df.empty and pd.notna(df.iloc[0]["start_date"]):
        return df.iloc[0]["start_date"].date(), df.iloc[0]["end_date"].date()
    return None, None


def get_total_keyword_count_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT DISTINCT ak.essential_keywords, ak.variable_keywords
        FROM ads a
        JOIN ad_performance_daily apd ON a.id = apd.ad_id
        LEFT JOIN ad_keywords ak ON a.id = ak.ad_id
        WHERE a.id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
    """
    df = pd.read_sql(query, engine)
    all_keywords = set()
    for _, row in df.iterrows():
        for col in ["essential_keywords", "variable_keywords"]:
            val = row[col]
            if val is None:
                continue
            if isinstance(val, (list, np.ndarray)):
                for k in val:
                    if k:
                        all_keywords.add(str(k).strip())
            elif isinstance(val, str):
                cleaned = val.replace("{", "").replace("}", "").strip()
                for k in cleaned.split(","):
                    k_s = k.strip()
                    if k_s:
                        all_keywords.add(k_s)
    return len(all_keywords)


def get_ctr_data_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT
            DATE_TRUNC('week', apd.as_of_date)::date AS week_start,
            SUM(clicks) AS total_clicks,
            SUM(impressions) AS total_impressions,
            ROUND((SUM(clicks)::numeric / NULLIF(SUM(impressions), 0)::numeric) * 100, 2) AS ctr
        FROM ad_performance_daily apd
        WHERE apd.ad_id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
        GROUP BY week_start
        ORDER BY week_start
    """
    df = pd.read_sql(query, engine)
    return None if df.empty else df


def get_ctr_monthly_data_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT
            DATE_TRUNC('month', apd.as_of_date)::date AS month_start,
            SUM(clicks) AS total_clicks,
            SUM(impressions) AS total_impressions,
            ROUND((SUM(clicks)::numeric / NULLIF(SUM(impressions), 0)::numeric) * 100, 2) AS ctr
        FROM ad_performance_daily apd
        WHERE apd.ad_id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
        GROUP BY month_start
        ORDER BY month_start
    """
    df = pd.read_sql(query, engine)
    return None if df.empty else df


def get_imp_threshold_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT SUM(impressions) AS total_site_imp
        FROM ad_performance_daily apd
        WHERE apd.ad_id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
    """
    df = pd.read_sql(query, engine)
    total = df.iloc[0]["total_site_imp"] or 0
    return total, total * 0.0005


def get_target_heatmap_by_ids(ad_ids, date_start, date_end, threshold):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT
            apd.age_range AS age,
            apd.gender,
            SUM(apd.impressions) AS impressions,
            SUM(apd.clicks) AS clicks,
            ROUND((SUM(apd.clicks)::numeric / NULLIF(SUM(apd.impressions), 0)::numeric) * 100, 2) AS ctr
        FROM ad_performance_daily apd
        WHERE apd.ad_id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND apd.gender != 'unknown'
        GROUP BY apd.age_range, apd.gender
        HAVING SUM(apd.impressions) >= {threshold}
    """
    df = pd.read_sql(query, engine)
    return None if df.empty else df


def get_purchase_heatmap_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT
            apd.age_range AS age,
            apd.gender,
            COALESCE(SUM(apd.purchase_count), 0) AS purchases
        FROM ad_performance_daily apd
        WHERE apd.ad_id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND apd.purchase_count IS NOT NULL
            AND apd.age_range IS NOT NULL
            AND apd.gender IS NOT NULL
        GROUP BY apd.age_range, apd.gender
        ORDER BY apd.age_range, apd.gender
    """
    df = pd.read_sql(query, engine)
    return None if df.empty else df


def get_purchase_age_gender_heatmap_page_data_by_ids(ad_ids, date_start, date_end):
    heatmap_df = get_purchase_heatmap_by_ids(ad_ids, date_start, date_end)
    if heatmap_df is None or heatmap_df.empty:
        return {"is_visible": False}
    return {"is_visible": True, "title": "타겟별 구매전환", "heatmap": heatmap_df}


def get_raw_keyword_performance_by_ids(ad_ids, date_start, date_end, target_age=None, target_gender=None):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    target_filter = _build_target_filter(target_age, target_gender)
    query = f"""
        SELECT
            ek.keyword,
            COUNT(DISTINCT perf.ad_body) AS doc_freq,
            SUM(perf.ad_imp) AS total_impressions,
            SUM(perf.ad_clk) AS total_clicks,
            ROUND((SUM(perf.ad_clk)::numeric / NULLIF(SUM(perf.ad_imp), 0)) * 100, 2) AS avg_ctr
        FROM (
            SELECT
                apd.ad_id,
                MAX(a.body) AS ad_body,
                SUM(apd.impressions) AS ad_imp,
                SUM(apd.clicks) AS ad_clk
            FROM ad_performance_daily apd
            INNER JOIN ads a ON apd.ad_id = a.id
            WHERE apd.ad_id IN {ids}
                AND apd.as_of_date >= '{date_start}'::date
                AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
                {target_filter}
                AND apd.gender != 'unknown'
            GROUP BY apd.ad_id
        ) perf
        INNER JOIN (
            SELECT DISTINCT
                ak.ad_id,
                CASE
                    WHEN REPLACE(REPLACE(TRIM(k.keyword), ' ', ''), ',', '') IN ('브라키오', '브라', '키오') THEN '브라키오'
                    WHEN REPLACE(REPLACE(TRIM(k.keyword), ' ', ''), ',', '') = '숭하' THEN '뒤숭숭하'
                    ELSE TRIM(k.keyword)
                END AS keyword
            FROM ad_keywords ak
            CROSS JOIN LATERAL UNNEST(
                COALESCE(ak.essential_keywords, ARRAY[]::text[]) ||
                COALESCE(ak.variable_keywords, ARRAY[]::text[])
            ) AS k(keyword)
            WHERE ak.ad_id IN {ids}
        ) ek ON perf.ad_id = ek.ad_id
        GROUP BY ek.keyword
        HAVING COUNT(DISTINCT perf.ad_body) >= 1
    """
    return pd.read_sql(query, engine)


def get_strategic_performance_by_ids(ad_ids, date_start, date_end, target_age=None, target_gender=None):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    target_filter = _build_target_filter(target_age, target_gender)
    query = f"""
        WITH ad_raw AS (
            SELECT
                ad.id AS ad_id,
                MAX(ad.body) AS ad_body,
                ak.essential_keywords,
                ak.variable_keywords,
                SUM(apd.impressions) AS ad_imps,
                SUM(apd.clicks) AS ad_clicks
            FROM ads ad
            JOIN ad_keywords ak ON ad.id = ak.ad_id
            LEFT JOIN ad_performance_daily apd ON ad.id = apd.ad_id
            WHERE ad.id IN {ids}
                AND apd.as_of_date >= '{date_start}'
                AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
                {target_filter}
            GROUP BY ad.id, ak.essential_keywords, ak.variable_keywords
            HAVING array_length(ak.essential_keywords, 1) >= 2
        ),
        raw_pairs AS (
            SELECT
                ad_id, ad_body, ad_imps, ad_clicks, variable_keywords,
                NULLIF(TRIM(essential_keywords[i]), '') AS ess_a,
                NULLIF(TRIM(essential_keywords[j]), '') AS ess_b
            FROM ad_raw,
            LATERAL generate_series(1, array_length(essential_keywords, 1)) i,
            LATERAL generate_series(i + 1, array_length(essential_keywords, 1)) j
        ),
        combo_pairs AS (
            SELECT DISTINCT
                ad_id, ad_body, ad_imps, ad_clicks, variable_keywords,
                LEAST(ess_a, ess_b) AS ess_1,
                GREATEST(ess_a, ess_b) AS ess_2
            FROM raw_pairs
            WHERE ess_a IS NOT NULL AND ess_b IS NOT NULL
        ),
        essential_agg AS (
            SELECT
                ess_1, ess_2,
                COUNT(DISTINCT ad_id) AS combo_doc_freq,
                SUM(ad_imps) AS total_imps,
                ROUND((SUM(ad_clicks)::numeric / NULLIF(SUM(ad_imps), 0)::numeric) * 100, 2) AS combo_overall_ctr
            FROM combo_pairs
            GROUP BY ess_1, ess_2
            HAVING COUNT(DISTINCT ad_id) >= 1
        ),
        variable_agg AS (
            SELECT
                cp.ess_1, cp.ess_2,
                vk.var_keyword,
                COUNT(DISTINCT cp.ad_body) AS var_body_doc_freq,
                SUM(cp.ad_imps) AS v_imps,
                SUM(cp.ad_clicks) AS v_clicks
            FROM combo_pairs cp
            INNER JOIN essential_agg ea ON cp.ess_1 = ea.ess_1 AND cp.ess_2 = ea.ess_2
            CROSS JOIN LATERAL (
                SELECT DISTINCT
                    CASE
                        WHEN REPLACE(REPLACE(TRIM(v.keyword), ' ', ''), ',', '') IN ('브라키오', '브라', '키오') THEN '브라키오'
                        WHEN REPLACE(REPLACE(TRIM(v.keyword), ' ', ''), ',', '') = '숭하' THEN '뒤숭숭하'
                        ELSE TRIM(v.keyword)
                    END AS var_keyword
                FROM UNNEST(COALESCE(cp.variable_keywords, ARRAY[]::text[])) AS v(keyword)
            ) vk
            GROUP BY cp.ess_1, cp.ess_2, vk.var_keyword
            HAVING COUNT(DISTINCT cp.ad_body) >= 1
        ),
        top_essential AS (
            SELECT ess_1, ess_2, combo_doc_freq, combo_overall_ctr
            FROM essential_agg
            ORDER BY combo_overall_ctr DESC
            LIMIT 15
        )
        SELECT
            te.ess_1, te.ess_2,
            te.combo_doc_freq,
            te.combo_overall_ctr,
            va.var_keyword,
            va.v_clicks,
            ROUND((va.v_clicks::numeric / NULLIF(va.v_imps, 0)::numeric) * 100, 2) AS with_var_ctr,
            va.v_imps AS var_imps
        FROM top_essential te
        JOIN variable_agg va ON te.ess_1 = va.ess_1 AND te.ess_2 = va.ess_2
        ORDER BY te.combo_overall_ctr DESC, with_var_ctr DESC
    """
    df = pd.read_sql(query, engine)
    if df is None or df.empty:
        return df
    df = df.copy()
    df["noun_keyword"] = df["var_keyword"].apply(lambda x: _normalize_keyword_by_pos(x, "noun"))
    df = df.dropna(subset=["noun_keyword"])
    if df.empty:
        return pd.DataFrame(columns=["ess_1", "ess_2", "combo_doc_freq", "combo_overall_ctr", "var_keyword", "with_var_ctr", "var_imps"])
    df["var_keyword"] = df["noun_keyword"]
    grouped = (
        df.groupby(["ess_1", "ess_2", "combo_doc_freq", "combo_overall_ctr", "var_keyword"], as_index=False)
        .agg(v_clicks=("v_clicks", "sum"), var_imps=("var_imps", "sum"))
    )
    grouped["with_var_ctr"] = np.where(
        grouped["var_imps"] > 0,
        np.round((grouped["v_clicks"] / grouped["var_imps"]) * 100, 2),
        np.nan,
    )
    grouped = grouped.sort_values(by=["combo_overall_ctr", "with_var_ctr"], ascending=[False, False])
    return grouped[["ess_1", "ess_2", "combo_doc_freq", "combo_overall_ctr", "var_keyword", "with_var_ctr", "var_imps"]]


def get_content_cards_by_ids(ad_ids, date_start, date_end):
    """선택된 ad_ids 콘텐츠 카드 반환 (CTR 기준 내림차순)"""
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT
            a.id AS ad_id,
            a.fb_ad_id,
            COALESCE(ig.ig_timestamp, a.fb_created_time) AS uploaded_at,
            COALESCE(
                NULLIF(ig.thumbnail_url, ''),
                NULLIF(ig.media_url, ''),
                NULLIF(a.thumb_link, '')
            ) AS thumbnail,
            ROUND((SUM(apd.clicks)::numeric / NULLIF(SUM(apd.impressions), 0)::numeric) * 100, 2) AS ctr
        FROM ads a
        LEFT JOIN ad_performance_daily apd ON a.id = apd.ad_id
        LEFT JOIN ig_contents ig ON a.source_ig_media_id = ig.fb_ig_media_id
        WHERE a.id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
        GROUP BY a.id, a.fb_ad_id, ig.ig_timestamp, a.fb_created_time, ig.thumbnail_url, ig.media_url, a.thumb_link
        ORDER BY ctr DESC NULLS LAST
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return []
    results = []
    for _, row in df.iterrows():
        thumb_val = row.get("thumbnail")
        thumb_val = None if pd.isna(thumb_val) else str(thumb_val).strip() or None
        uploaded_at = row.get("uploaded_at")
        uploaded_at = uploaded_at.date() if pd.notna(uploaded_at) else None
        results.append({
            "ad_id": row["ad_id"],
            "fb_ad_id": row.get("fb_ad_id"),
            "uploaded_at": uploaded_at,
            "thumbnail": thumb_val,
            "ctr": row["ctr"],
        })
    return results


def has_purchase_data_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT 1
        FROM ad_performance_daily apd
        WHERE apd.ad_id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= '{date_end}'
            AND apd.purchase_count IS NOT NULL
            AND apd.purchase_count > 0
        LIMIT 1
    """
    df = pd.read_sql(query, engine)
    return not df.empty


def get_purchase_roas_weekly_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT
            (DATE_TRUNC('week', apd.as_of_date) + INTERVAL '6 days')::date AS week_start,
            ROUND(
                (SUM(apd.spend * apd.purchase_roas) / NULLIF(SUM(apd.spend), 0))::numeric
            , 0) AS avg_roas
        FROM ad_performance_daily apd
        WHERE apd.ad_id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= '{date_end}'
            AND apd.spend IS NOT NULL
        GROUP BY (DATE_TRUNC('week', apd.as_of_date) + INTERVAL '6 days')::date
        ORDER BY week_start
    """
    df = pd.read_sql(query, engine)
    return None if df.empty else df


def get_purchase_roas_monthly_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT
            DATE_TRUNC('month', apd.as_of_date)::date AS month_start,
            ROUND(
                (SUM(apd.spend * apd.purchase_roas) / NULLIF(SUM(apd.spend), 0))::numeric
            , 0) AS avg_roas
        FROM ad_performance_daily apd
        WHERE apd.ad_id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= '{date_end}'
            AND apd.purchase_roas IS NOT NULL
            AND apd.spend IS NOT NULL
        GROUP BY DATE_TRUNC('month', apd.as_of_date)::date
        ORDER BY month_start
    """
    df = pd.read_sql(query, engine)
    return None if df.empty else df


def get_purchase_count_weekly_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        WITH weeks AS (
            SELECT generate_series(
                (DATE_TRUNC('week', '{date_start}'::date) + INTERVAL '6 days')::date,
                (DATE_TRUNC('week', '{date_end}'::date) + INTERVAL '6 days')::date,
                INTERVAL '7 days'
            )::date AS week_start
        ),
        agg AS (
            SELECT
                (DATE_TRUNC('week', apd.as_of_date) + INTERVAL '6 days')::date AS week_start,
                COALESCE(SUM(apd.purchase_count), 0) AS purchases
            FROM ad_performance_daily apd
            WHERE apd.ad_id IN {ids}
                AND apd.as_of_date >= '{date_start}'
                AND apd.as_of_date <= '{date_end}'
                AND apd.purchase_count IS NOT NULL
            GROUP BY (DATE_TRUNC('week', apd.as_of_date) + INTERVAL '6 days')::date
        )
        SELECT
            w.week_start,
            COALESCE(a.purchases, 0) AS purchases
        FROM weeks w
        LEFT JOIN agg a ON a.week_start = w.week_start
        ORDER BY w.week_start
    """
    df = pd.read_sql(query, engine)
    return None if df.empty else df


def get_purchase_count_monthly_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        WITH months AS (
            SELECT generate_series(
                DATE_TRUNC('month', '{date_start}'::date)::date,
                DATE_TRUNC('month', '{date_end}'::date)::date,
                INTERVAL '1 month'
            )::date AS month_start
        ),
        agg AS (
            SELECT
                DATE_TRUNC('month', apd.as_of_date)::date AS month_start,
                COALESCE(SUM(apd.purchase_count), 0) AS purchases
            FROM ad_performance_daily apd
            WHERE apd.ad_id IN {ids}
                AND apd.as_of_date >= '{date_start}'
                AND apd.as_of_date <= '{date_end}'
                AND apd.purchase_count IS NOT NULL
            GROUP BY DATE_TRUNC('month', apd.as_of_date)::date
        )
        SELECT
            m.month_start,
            COALESCE(a.purchases, 0) AS purchases
        FROM months m
        LEFT JOIN agg a ON a.month_start = m.month_start
        ORDER BY m.month_start
    """
    df = pd.read_sql(query, engine)
    return None if df.empty else df


def has_revenue_data_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT 1
        FROM ad_performance_daily apd
        WHERE apd.ad_id IN {ids}
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
            AND apd.spend IS NOT NULL
            AND apd.purchase_roas IS NOT NULL
        LIMIT 1
    """
    df = pd.read_sql(query, engine)
    return not df.empty


def get_spend_and_revenue_weekly_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        WITH weeks AS (
            SELECT generate_series(
                DATE_TRUNC('week', '{date_start}'::date)::date,
                DATE_TRUNC('week', '{date_end}'::date)::date,
                INTERVAL '7 days'
            )::date AS week_start
        ),
        agg AS (
            SELECT
                DATE_TRUNC('week', apd.as_of_date)::date AS week_start,
                ROUND(COALESCE(SUM(apd.spend), 0)::numeric, 0) AS spend,
                ROUND(COALESCE(SUM(apd.spend * apd.purchase_roas), 0)::numeric, 0) AS revenue
            FROM ad_performance_daily apd
            WHERE apd.ad_id IN {ids}
                AND apd.as_of_date >= '{date_start}'
                AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
                AND apd.spend IS NOT NULL
                AND apd.purchase_roas IS NOT NULL
            GROUP BY DATE_TRUNC('week', apd.as_of_date)::date
        )
        SELECT
            w.week_start,
            ROUND(COALESCE(a.spend, 0)::numeric, 0) AS spend,
            ROUND(COALESCE(a.revenue, 0)::numeric, 0) AS revenue
        FROM weeks w
        LEFT JOIN agg a ON a.week_start = w.week_start
        ORDER BY w.week_start
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return None
    nonzero_idx = df.index[df["spend"] != 0]
    if len(nonzero_idx) > 0:
        df = df.loc[nonzero_idx[0]:].reset_index(drop=True)
    return df


def get_spend_and_revenue_monthly_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        WITH months AS (
            SELECT generate_series(
                DATE_TRUNC('month', '{date_start}'::date)::date,
                DATE_TRUNC('month', '{date_end}'::date)::date,
                INTERVAL '1 month'
            )::date AS month_start
        ),
        agg AS (
            SELECT
                DATE_TRUNC('month', apd.as_of_date)::date AS month_start,
                ROUND(COALESCE(SUM(apd.spend), 0)::numeric, 0) AS spend,
                ROUND(COALESCE(SUM(apd.spend * apd.purchase_roas), 0)::numeric, 0) AS revenue
            FROM ad_performance_daily apd
            WHERE apd.ad_id IN {ids}
                AND apd.as_of_date >= '{date_start}'
                AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
                AND apd.spend IS NOT NULL
                AND apd.purchase_roas IS NOT NULL
            GROUP BY DATE_TRUNC('month', apd.as_of_date)::date
        )
        SELECT
            m.month_start,
            ROUND(COALESCE(a.spend, 0)::numeric, 0) AS spend,
            ROUND(COALESCE(a.revenue, 0)::numeric, 0) AS revenue
        FROM months m
        LEFT JOIN agg a ON a.month_start = m.month_start
        ORDER BY m.month_start
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return None
    nonzero_idx = df.index[df["spend"] != 0]
    if len(nonzero_idx) > 0:
        df = df.loc[nonzero_idx[0]:].reset_index(drop=True)
    return df


def get_purchase_contents_data_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
        SELECT
            a.source_ig_media_id AS content_key,
            MIN(ig.ig_timestamp) AS uploaded_at,
            COALESCE(
                MAX(NULLIF(ig.thumbnail_url, '')),
                MAX(NULLIF(ig.media_url, '')),
                MAX(NULLIF(a.thumb_link, ''))
            ) AS thumbnail,
            STRING_AGG(DISTINCT a.ad_name, ' / ') AS ad_names,
            ARRAY_AGG(DISTINCT a.id) AS ad_ids,
            ARRAY_AGG(DISTINCT a.fb_ad_id) FILTER (WHERE a.fb_ad_id IS NOT NULL) AS fb_ad_ids,
            COALESCE(SUM(apd.purchase_count), 0) AS purchases
        FROM ad_performance_daily apd
        JOIN ads a ON apd.ad_id = a.id
        LEFT JOIN ig_contents ig ON a.source_ig_media_id = ig.fb_ig_media_id
        WHERE a.id IN {ids}
            AND a.source_ig_media_id IS NOT NULL
            AND apd.as_of_date >= '{date_start}'
            AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
        GROUP BY a.source_ig_media_id
        HAVING COALESCE(SUM(apd.purchase_count), 0) >= 1
        ORDER BY purchases DESC, MIN(ig.ig_timestamp) DESC
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return []
    results = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        thumb_val = row.get("thumbnail")
        thumb_val = None if pd.isna(thumb_val) else str(thumb_val).strip() or None
        uploaded_at = row.get("uploaded_at")
        uploaded_at = uploaded_at.date() if pd.notna(uploaded_at) else None
        results.append({
            "rank": rank,
            "content_key": row["content_key"],
            "source_instagram_media_id": row["content_key"],
            "ad_name": row.get("ad_names"),
            "fb_ad_id": (row.get("fb_ad_ids") or [None])[0],
            "fb_ad_ids": row.get("fb_ad_ids") or [],
            "ad_ids": row.get("ad_ids") or [],
            "uploaded_at": uploaded_at,
            "thumbnail": thumb_val,
            "purchases": int(row["purchases"]) if pd.notna(row["purchases"]) else 0,
        })
    return results


def get_purchase_contents_pages_data_by_ids(ad_ids, date_start, date_end):
    contents = get_purchase_contents_data_by_ids(ad_ids, date_start, date_end)
    if not contents:
        return {"title": "구매가 발생한 콘텐츠", "pages": [], "items": [], "total_count": 0}
    return {
        "title": "구매가 발생한 콘텐츠",
        "pages": chunk_list(contents, 4),
        "items": contents,
        "total_count": len(contents),
    }


def get_essence_target_performance_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
    SELECT
        res.single_ess AS "키워드",
        res.total_ad_count AS "등장 광고 수",
        MAX(CASE WHEN res.imp_rank = 1 THEN res.age || ' ' || res.gender END) AS "최다 노출 타겟",
        MAX(CASE WHEN res.imp_rank = 1 THEN res.target_imp END)::bigint AS "타겟 노출량",
        ROUND(MAX(CASE WHEN res.imp_rank = 1 THEN res.target_imp END)::numeric / NULLIF(MAX(res.total_imp), 0) * 100, 1) || '%%' AS "노출 비중",
        MAX(res.total_imp)::bigint AS "총 노출량",
        MAX(CASE WHEN res.clk_rank = 1 THEN res.age || ' ' || res.gender END) AS "최다 클릭 타겟",
        MAX(CASE WHEN res.clk_rank = 1 THEN res.target_clk END)::bigint AS "타겟 클릭량",
        ROUND(MAX(CASE WHEN res.clk_rank = 1 THEN res.target_clk END)::numeric / NULLIF(MAX(res.total_clk), 0) * 100, 1) || '%%' AS "클릭 비중",
        MAX(res.total_clk)::bigint AS "총 클릭량"
    FROM (
        SELECT
            ts.single_ess, ts.age, ts.gender, ts.target_imp, ts.target_clk,
            summ.total_ad_count,
            SUM(ts.target_imp) OVER(PARTITION BY ts.single_ess) AS total_imp,
            SUM(ts.target_clk) OVER(PARTITION BY ts.single_ess) AS total_clk,
            RANK() OVER (PARTITION BY ts.single_ess ORDER BY ts.target_imp DESC, ts.age) AS imp_rank,
            RANK() OVER (PARTITION BY ts.single_ess ORDER BY ts.target_clk DESC, ts.age) AS clk_rank
        FROM (
            SELECT ak_u.single_ess, p.age, p.gender,
                SUM(p.imp) AS target_imp, SUM(p.clk) AS target_clk
            FROM (
                SELECT ad_id, UNNEST(essential_keywords) AS single_ess
                FROM ad_keywords
                WHERE ad_id IN {ids}
                    AND essential_keywords IS NOT NULL
                    AND ARRAY_LENGTH(essential_keywords, 1) > 0
            ) ak_u
            INNER JOIN (
                SELECT apd.ad_id, apd.age_range AS age, apd.gender,
                    SUM(apd.impressions) AS imp, SUM(apd.clicks) AS clk
                FROM ad_performance_daily apd
                WHERE apd.ad_id IN {ids}
                    AND apd.as_of_date >= '{date_start}'::date
                    AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
                GROUP BY 1, 2, 3
            ) p ON ak_u.ad_id = p.ad_id
            GROUP BY 1, 2, 3
        ) ts
        INNER JOIN (
            SELECT UNNEST(ak.essential_keywords) AS single_ess,
                COUNT(DISTINCT ak.ad_id) AS total_ad_count
            FROM ad_keywords ak
            WHERE ak.ad_id IN {ids}
            GROUP BY 1
        ) summ ON ts.single_ess = summ.single_ess
    ) res
    GROUP BY res.single_ess, res.total_ad_count
    ORDER BY "등장 광고 수" DESC, "총 노출량" DESC
    """
    return pd.read_sql(query, engine)


def get_variable_target_performance_by_ids(ad_ids, date_start, date_end):
    engine = get_engine()
    ids = _ids_sql(ad_ids)
    query = f"""
    SELECT
        res.single_var AS "키워드",
        res.total_ad_count AS "등장 광고 수",
        MAX(CASE WHEN res.imp_rank = 1 THEN res.age || ' ' || res.gender END) AS "최다 노출 타겟",
        MAX(CASE WHEN res.imp_rank = 1 THEN res.target_imp END)::bigint AS "타겟 노출량",
        ROUND(MAX(CASE WHEN res.imp_rank = 1 THEN res.target_imp END)::numeric / NULLIF(MAX(res.total_imp), 0) * 100, 1) || '%%' AS "노출 비중",
        MAX(res.total_imp)::bigint AS "총 노출량",
        MAX(CASE WHEN res.clk_rank = 1 THEN res.age || ' ' || res.gender END) AS "최다 클릭 타겟",
        MAX(CASE WHEN res.clk_rank = 1 THEN res.target_clk END)::bigint AS "타겟 클릭량",
        ROUND(MAX(CASE WHEN res.clk_rank = 1 THEN res.target_clk END)::numeric / NULLIF(MAX(res.total_clk), 0) * 100, 1) || '%%' AS "클릭 비중",
        MAX(res.total_clk)::bigint AS "총 클릭량"
    FROM (
        SELECT
            ts.single_var, ts.age, ts.gender, ts.target_imp, ts.target_clk,
            summ.total_ad_count,
            SUM(ts.target_imp) OVER(PARTITION BY ts.single_var) AS total_imp,
            SUM(ts.target_clk) OVER(PARTITION BY ts.single_var) AS total_clk,
            RANK() OVER (PARTITION BY ts.single_var ORDER BY ts.target_imp DESC, ts.age) AS imp_rank,
            RANK() OVER (PARTITION BY ts.single_var ORDER BY ts.target_clk DESC, ts.age) AS clk_rank
        FROM (
            SELECT ak_u.single_var, p.age, p.gender,
                SUM(p.imp) AS target_imp, SUM(p.clk) AS target_clk
            FROM (
                SELECT ad_id, UNNEST(variable_keywords) AS single_var
                FROM ad_keywords
                WHERE ad_id IN {ids}
                    AND variable_keywords IS NOT NULL
                    AND ARRAY_LENGTH(variable_keywords, 1) > 0
            ) ak_u
            INNER JOIN (
                SELECT apd.ad_id, apd.age_range AS age, apd.gender,
                    SUM(apd.impressions) AS imp, SUM(apd.clicks) AS clk
                FROM ad_performance_daily apd
                WHERE apd.ad_id IN {ids}
                    AND apd.as_of_date >= '{date_start}'::date
                    AND apd.as_of_date <= DATE_TRUNC('week', '{date_end}'::date)::date
                GROUP BY 1, 2, 3
            ) p ON ak_u.ad_id = p.ad_id
            GROUP BY 1, 2, 3
        ) ts
        INNER JOIN (
            SELECT UNNEST(ak.variable_keywords) AS single_var,
                COUNT(DISTINCT ak.ad_id) AS total_ad_count
            FROM ad_keywords ak
            WHERE ak.ad_id IN {ids}
            GROUP BY 1
        ) summ ON ts.single_var = summ.single_var
    ) res
    GROUP BY res.single_var, res.total_ad_count
    ORDER BY "등장 광고 수" DESC, "총 노출량" DESC
    LIMIT 50
    """
    return pd.read_sql(query, engine)
