"""
Korean Stock Screener — 멀티 조건검색 + GitHub Pages HTML 출력
"""

import os
import time
import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import pandas as pd
import requests
from pykrx import stock as pykrx_stock

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GMAIL_USER       = os.environ.get("GMAIL_USER", "")
GMAIL_PASS       = os.environ.get("GMAIL_PASS", "")
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

MKTCAP_MIN       = 200_000_000_000
ENV_PERIOD       = 20
ENV_BAND         = 0.06
VOLUME_RATIO_MIN = 1.5
RSI_DAILY_MAX    = 20
RSI_WEEKLY_MAX   = 25
RSI_PERIOD       = 14
GROWTH_MIN_PCT   = 20.0
VOL_EXPLOSION    = 10.0
PRICE_SURGE      = 5.0
DUAL_BUYING_DAYS = 3


def today_str():
    return datetime.today().strftime("%Y%m%d")

def date_str(days_ago):
    return (datetime.today() - timedelta(days=days_ago)).strftime("%Y%m%d")


def get_filtered_tickers():
    today = today_str()
    result = []
    for market in ["KOSPI", "KOSDAQ"]:
        try:
            cap_df = pykrx_stock.get_market_cap_by_ticker(today, market=market)
            filtered = cap_df[cap_df["시가총액"] >= MKTCAP_MIN].index.tolist()
            result.extend(filtered)
        except Exception as e:
            log.warning(f"{market} 시가총액 조회 실패: {e}")
    log.info(f"시가총액 2,000억 이상 종목: {len(result)}개")
    return result


def fetch_ohlcv(ticker, days=80):
    end   = datetime.today()
    start = end - timedelta(days=days * 2)
    df = pykrx_stock.get_market_ohlcv_by_date(
        start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ticker
    )
    if df is None or len(df) < 20:
        return pd.DataFrame()
    return df.tail(days).copy()


def fetch_ohlcv_weekly(ticker, weeks=20):
    df = fetch_ohlcv(ticker, days=weeks * 7 + 30)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    weekly = df["종가"].resample("W").last().dropna()
    return weekly.to_frame()


def calc_rsi(series, period=RSI_PERIOD):
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-9)
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1) if not rsi.empty else 50.0


def check_envelope(ticker):
    df = fetch_ohlcv(ticker, days=60)
    if df.empty:
        return None
    df["env_mid"]   = df["종가"].rolling(ENV_PERIOD).mean()
    df["env_upper"] = df["env_mid"] * (1 + ENV_BAND)
    df["env_lower"] = df["env_mid"] * (1 - ENV_BAND)
    latest, prev = df.iloc[-1], df.iloc[-2]
    if pd.isna(latest["env_mid"]) or prev["거래량"] == 0:
        return None
    if latest["거래량"] / prev["거래량"] < VOLUME_RATIO_MIN:
        return None
    close, mid = latest["종가"], latest["env_mid"]
    signal = None
    if prev["종가"] <= prev["env_lower"] and close > latest["env_lower"]:
        signal = "하단밴드 터치 반등"
    elif prev["종가"] < prev["env_mid"] and close >= mid:
        signal = "중심선 상향 돌파"
    elif prev["종가"] < prev["env_upper"] and close >= latest["env_upper"]:
        signal = "상단밴드 돌파(강세)"
    if not signal:
        return None
    return {
        "condition": "엔벨로프",
        "signal": signal,
        "close": int(close),
        "detail": f"중심선대비 {round((close-mid)/mid*100,1):+.1f}%",
    }


def check_rsi(ticker):
    df = fetch_ohlcv(ticker, days=60)
    if df.empty:
        return None
    daily_rsi  = calc_rsi(df["종가"])
    wdf        = fetch_ohlcv_weekly(ticker, weeks=20)
    weekly_rsi = calc_rsi(wdf["종가"]) if not wdf.empty else 50.0
    hit_daily  = daily_rsi  <= RSI_DAILY_MAX
    hit_weekly = weekly_rsi <= RSI_WEEKLY_MAX
    if not (hit_daily or hit_weekly):
        return None
    flags = []
    if hit_daily:  flags.append(f"일봉 RSI {daily_rsi}")
    if hit_weekly: flags.append(f"주봉 RSI {weekly_rsi}")
    return {
        "condition": "RSI 과매도",
        "signal": " / ".join(flags),
        "close": int(df["종가"].iloc[-1]),
        "detail": "과매도 반등 구간",
    }


def check_growth(ticker):
    try:
        df = pykrx_stock.get_market_fundamental_by_ticker(today_str(), market="ALL")
        if df is None or ticker not in df.index:
            return None
        row = df.loc[ticker]
        eps = float(row.get("EPS", 0))
        bps = float(row.get("BPS", 1))
        if eps <= 0 or bps <= 0:
            return None
        roe = eps / bps * 100
        if roe < GROWTH_MIN_PCT:
            return None
        close_df = fetch_ohlcv(ticker, days=5)
        close = int(close_df["종가"].iloc[-1]) if not close_df.empty else 0
        return {
            "condition": "성장주",
            "signal": f"ROE {round(roe,1)}%",
            "close": close,
            "detail": f"EPS {int(eps):,}원 / BPS {int(bps):,}원",
        }
    except Exception as e:
        log.debug(f"{ticker} 성장주 오류: {e}")
        return None


def check_volume_explosion(ticker):
    df = fetch_ohlcv(ticker, days=10)
    if df is None or len(df) < 2:
        return None
    prev_vol = df["거래량"].iloc[-2]
    if prev_vol == 0:
        return None
    vol_ratio  = df["거래량"].iloc[-1] / prev_vol
    open_price = df["시가"].iloc[-1]
    close      = df["종가"].iloc[-1]
    if open_price == 0:
        return None
    price_chg = (close - open_price) / open_price * 100
    if vol_ratio < VOL_EXPLOSION or price_chg < PRICE_SURGE:
        return None
    return {
        "condition": "거래량 폭발",
        "signal": f"거래량 {round(vol_ratio,1)}배 / 주가 +{round(price_chg,1)}%",
        "close": int(close),
        "detail": f"당일 거래량 {int(df['거래량'].iloc[-1]):,}주",
    }


def check_dual_buying(ticker):
    try:
        df = pykrx_stock.get_market_trading_value_by_date(
            date_str(DUAL_BUYING_DAYS * 2 + 5), today_str(), ticker
        )
        if df is None or len(df) < DUAL_BUYING_DAYS:
            return None
        recent   = df.tail(DUAL_BUYING_DAYS)
        cols     = recent.columns.tolist()
        inst_col = next((c for c in cols if "기관" in c), None)
        fore_col = next((c for c in cols if "외국" in c), None)
        if not inst_col or not fore_col:
            return None
        if not ((recent[inst_col] > 0).all() and (recent[fore_col] > 0).all()):
            return None
        inst_total = int(recent[inst_col].sum())
        fore_total = int(recent[fore_col].sum())
        close_df = fetch_ohlcv(ticker, days=5)
        close = int(close_df["종가"].iloc[-1]) if not close_df.empty else 0
        return {
            "condition": "기관+외국인 쌍끌이",
            "signal": "3일 연속 동시 순매수",
            "close": close,
            "detail": f"기관 {inst_total/1e8:.1f}억 / 외국인 {fore_total/1e8:.1f}억",
        }
    except Exception as e:
        log.debug(f"{ticker} 쌍끌이 오류: {e}")
        return None


CHECKERS = [
    ("엔벨로프",           check_envelope),
    ("RSI 과매도",         check_rsi),
    ("성장주",             check_growth),
    ("거래량 폭발",        check_volume_explosion),
    ("기관+외국인 쌍끌이", check_dual_buying),
]

CONDITION_DESC = {
    "엔벨로프":           "20일 SMA ±6% 엔벨로프 신호 + 거래량 1.5배↑",
    "RSI 과매도":         "일봉 RSI 20 이하 or 주봉 RSI 25 이하",
    "성장주":             "ROE 20% 이상 (EPS/BPS 기반)",
    "거래량 폭발":        "전일 대비 거래량 10배↑ + 주가 5%↑",
    "기관+외국인 쌍끌이": "최근 3일 기관+외국인 동시 순매수",
}

CONDITION_COLOR = {
    "엔벨로프":           "#185FA5",
    "RSI 과매도":         "#0F6E56",
    "성장주":             "#854F0B",
    "거래량 폭발":        "#993556",
    "기관+외국인 쌍끌이": "#534AB7",
}


def run_screener():
    tickers = get_filtered_tickers()
    results = {name: [] for name, _ in CHECKERS}
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            log.info(f"진행 중: {i}/{len(tickers)}")
        name = pykrx_stock.get_market_ticker_name(ticker)
        for cond_name, checker in CHECKERS:
            try:
                r = checker(ticker)
                if r:
                    r["ticker"] = ticker
                    r["name"]   = name
                    results[cond_name].append(r)
            except Exception as e:
                log.debug(f"{ticker}/{cond_name} 오류: {e}")
        time.sleep(0.05)
    for cond_name, items in results.items():
        log.info(f"{cond_name}: {len(items)}종목")
    return results


# ────────────────────────────────────────────
# HTML 페이지 생성
# ────────────────────────────────────────────
def generate_html(results: dict) -> str:
    today    = datetime.today().strftime("%Y년 %m월 %d일")
    now      = datetime.today().strftime("%H:%M")
    total    = sum(len(v) for v in results.values())

    # 조건별 섹션 HTML
    sections = ""
    for cond_name, items in results.items():
        color = CONDITION_COLOR.get(cond_name, "#444")
        desc  = CONDITION_DESC.get(cond_name, "")
        badge = f'<span class="badge" style="background:{color}20;color:{color};border:1px solid {color}40">{len(items)}종목</span>'

        rows = ""
        if not items:
            rows = '<tr><td colspan="4" style="text-align:center;color:#888;padding:20px">해당 종목 없음</td></tr>'
        else:
            for r in items:
                rows += f"""
                <tr>
                  <td><strong>{r['name']}</strong><br><span class="code">{r['ticker']}</span></td>
                  <td>{r['signal']}</td>
                  <td style="text-align:right"><strong>{r['close']:,}원</strong></td>
                  <td style="color:#888;font-size:13px">{r['detail']}</td>
                </tr>"""

        sections += f"""
        <div class="section">
          <div class="section-header" style="border-left:4px solid {color}">
            <div>
              <span class="section-title">{cond_name}</span>
              {badge}
            </div>
            <span class="section-desc">{desc}</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>종목명</th>
                <th>신호</th>
                <th style="text-align:right">현재가</th>
                <th>상세</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>주식 조건검색 결과</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Noto Sans KR', sans-serif;
         background: #f5f5f5; color: #222; line-height: 1.6; }}
  .header {{ background: #0f1923; color: white; padding: 28px 24px; }}
  .header h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 4px; }}
  .header .meta {{ font-size: 13px; color: #aaa; }}
  .summary {{ display: flex; gap: 12px; padding: 16px 24px;
              background: white; border-bottom: 1px solid #eee; flex-wrap: wrap; }}
  .summary-card {{ background: #f8f9fa; border-radius: 8px; padding: 12px 20px; text-align: center; }}
  .summary-card .num {{ font-size: 24px; font-weight: 700; color: #185FA5; }}
  .summary-card .lbl {{ font-size: 12px; color: #888; margin-top: 2px; }}
  .container {{ max-width: 960px; margin: 20px auto; padding: 0 16px; }}
  .section {{ background: white; border-radius: 12px; margin-bottom: 16px;
              border: 1px solid #e8e8e8; overflow: hidden; }}
  .section-header {{ padding: 16px 20px; border-bottom: 1px solid #f0f0f0; }}
  .section-header > div {{ display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }}
  .section-title {{ font-size: 16px; font-weight: 600; }}
  .section-desc {{ font-size: 12px; color: #888; }}
  .badge {{ font-size: 12px; padding: 2px 10px; border-radius: 999px; font-weight: 500; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ padding: 10px 16px; font-size: 12px; color: #888; font-weight: 500;
        border-bottom: 1px solid #f0f0f0; text-align: left; background: #fafafa; }}
  td {{ padding: 12px 16px; font-size: 14px; border-bottom: 1px solid #f8f8f8; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #fafcff; }}
  .code {{ font-size: 12px; color: #aaa; }}
  .footer {{ text-align: center; padding: 24px; font-size: 12px; color: #aaa; }}
  @media(max-width:600px) {{ th:nth-child(4), td:nth-child(4) {{ display:none; }} }}
</style>
</head>
<body>
<div class="header">
  <h1>📊 주식 조건검색 결과</h1>
  <div class="meta">{today} {now} 기준 &nbsp;·&nbsp; 시가총액 2,000억 이상 대상</div>
</div>
<div class="summary">
  <div class="summary-card">
    <div class="num">{total}</div>
    <div class="lbl">총 발견 건수</div>
  </div>
  {''.join(f'<div class="summary-card"><div class="num" style="color:{CONDITION_COLOR.get(n,"#444")}">{len(v)}</div><div class="lbl">{n}</div></div>' for n,v in results.items())}
</div>
<div class="container">
  {sections}
</div>
<div class="footer">
  본 정보는 투자 참고용이며, 투자 판단의 책임은 본인에게 있습니다.<br>
  데이터 출처: 한국거래소(KRX) · 자동 업데이트: 평일 장 마감 후
</div>
</body>
</html>"""


# ────────────────────────────────────────────
# 알림
# ────────────────────────────────────────────
def format_text(results):
    today = datetime.today().strftime("%Y-%m-%d")
    total = sum(len(v) for v in results.values())
    lines = [f"📊 멀티 조건검색 [{today}] 총 {total}건\n"]
    for cond_name, items in results.items():
        lines.append(f"【{cond_name}】 {len(items)}종목")
        for r in items:
            lines.append(f"  ▶ {r['name']}({r['ticker']}) {r['close']:,}원 | {r['signal']}")
        lines.append("")
    return "\n".join(lines)


def send_email(subject, body):
    if not GMAIL_USER or not GMAIL_PASS:
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = GMAIL_USER
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.send_message(msg)
    log.info("이메일 발송 완료")


def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=10)
    log.info("텔레그램 발송 완료")


# ────────────────────────────────────────────
# 실행
# ────────────────────────────────────────────
def main():
    log.info("=== 멀티 조건검색 스크리너 시작 ===")
    results = run_screener()

    # HTML 저장 (GitHub Pages용)
    html = generate_html(results)
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    log.info("docs/index.html 생성 완료")

    # 알림
    today = datetime.today().strftime("%Y-%m-%d")
    total = sum(len(v) for v in results.values())
    text  = format_text(results)
    send_email(f"📊 [{today}] 조건검색 — {total}건 발견", text)
    send_telegram(text)
    log.info("=== 완료 ===")


if __name__ == "__main__":
    main()