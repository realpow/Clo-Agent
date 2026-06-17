"""
Korean Stock Screener with Envelope Filter (Claude API 없는 무료 버전)
조건: 이동평균 엔벨로프 + MA Squeeze + 거래량 필터
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

# ────────────────────────────────────────────
# 설정값 (GitHub Secrets → 환경변수)
# ────────────────────────────────────────────
GMAIL_USER       = os.environ.get("GMAIL_USER", "")
GMAIL_PASS       = os.environ.get("GMAIL_PASS", "")
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ────────────────────────────────────────────
# 파라미터 (여기서 조건 조정 가능)
# ────────────────────────────────────────────
ENV_PERIOD       = 20    # 엔벨로프 이동평균 기간 (일)
ENV_BAND         = 0.06  # 엔벨로프 밴드폭 ±6%
MA_SHORT         = 5     # 단기 이동평균
MA_LONG          = 20    # 장기 이동평균
SIDEWAYS_DAYS    = 10    # 횡보 판정 기간 (일)
SIDEWAYS_RANGE   = 0.05  # 횡보 범위 ±5%
VOLUME_RATIO_MIN = 1.5   # 거래량 전일 대비 최소 배수

SIGNAL_KR = {
    "lower_touch":  "하단밴드 터치 후 반등",
    "mid_breakout": "중심선 상향 돌파",
    "upper_break":  "상단밴드 돌파 (강세)",
}


# ────────────────────────────────────────────
# 데이터 수집
# ────────────────────────────────────────────
def get_ticker_list() -> list:
    today  = datetime.today().strftime("%Y%m%d")
    kospi  = pykrx_stock.get_market_ticker_list(today, market="KOSPI")
    kosdaq = pykrx_stock.get_market_ticker_list(today, market="KOSDAQ")
    tickers = list(kospi) + list(kosdaq)
    log.info(f"전체 종목 수: {len(tickers)}")
    return tickers


def fetch_ohlcv(ticker: str, days: int = 60) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=days * 2)
    df = pykrx_stock.get_market_ohlcv_by_date(
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        ticker,
    )
    if df is None or len(df) < 20:
        return pd.DataFrame()
    return df.tail(days).copy()


# ────────────────────────────────────────────
# 지표 계산
# ────────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 엔벨로프
    df["env_mid"]   = df["종가"].rolling(ENV_PERIOD).mean()
    df["env_upper"] = df["env_mid"] * (1 + ENV_BAND)
    df["env_lower"] = df["env_mid"] * (1 - ENV_BAND)
    # 이동평균
    df["ma_short"]  = df["종가"].rolling(MA_SHORT).mean()
    df["ma_long"]   = df["종가"].rolling(MA_LONG).mean()
    return df


def envelope_signal(df: pd.DataFrame):
    """엔벨로프 신호 탐지. 없으면 None 반환."""
    if len(df) < ENV_PERIOD + 5:
        return None
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    if pd.isna(latest["env_mid"]):
        return None

    close     = latest["종가"]
    env_mid   = latest["env_mid"]
    env_upper = latest["env_upper"]
    env_lower = latest["env_lower"]
    pos_pct   = (close - env_mid) / env_mid * 100

    signal = None
    if prev["종가"] <= prev["env_lower"] and close > env_lower:
        signal = "lower_touch"
    elif prev["종가"] < prev["env_mid"] and close >= env_mid:
        signal = "mid_breakout"
    elif prev["종가"] < prev["env_upper"] and close >= env_upper:
        signal = "upper_break"

    if signal is None:
        return None

    return {
        "signal":       signal,
        "close":        int(close),
        "env_mid":      int(env_mid),
        "env_upper":    int(env_upper),
        "env_lower":    int(env_lower),
        "position_pct": round(pos_pct, 1),
    }


def is_sideways(df: pd.DataFrame) -> bool:
    recent = df["종가"].tail(SIDEWAYS_DAYS)
    if len(recent) < SIDEWAYS_DAYS:
        return False
    return (recent.max() - recent.min()) / recent.mean() <= SIDEWAYS_RANGE * 2


def volume_surge(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev_vol = df["거래량"].iloc[-2]
    if prev_vol == 0:
        return False
    return df["거래량"].iloc[-1] / prev_vol >= VOLUME_RATIO_MIN


def golden_cross(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    cur  = df.iloc[-1]
    prev = df.iloc[-2]
    return prev["ma_short"] <= prev["ma_long"] and cur["ma_short"] > cur["ma_long"]


# ────────────────────────────────────────────
# 스크리닝
# ────────────────────────────────────────────
def screen_ticker(ticker: str):
    try:
        df = fetch_ohlcv(ticker)
        if df.empty:
            return None
        df  = calc_indicators(df)
        env = envelope_signal(df)
        if env is None:
            return None
        if not volume_surge(df):
            return None
        sw = is_sideways(df)
        gc = golden_cross(df)
        if not (sw or gc):
            return None
        name = pykrx_stock.get_market_ticker_name(ticker)
        return {**env, "ticker": ticker, "name": name, "sideways": sw, "golden_cross": gc}
    except Exception as e:
        log.debug(f"{ticker} 오류: {e}")
        return None


def run_screener() -> list:
    tickers = get_ticker_list()
    results = []
    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            log.info(f"진행 중: {i}/{len(tickers)}")
        r = screen_ticker(ticker)
        if r:
            results.append(r)
        time.sleep(0.05)
    log.info(f"조건 충족 종목: {len(results)}개")
    return results


# ────────────────────────────────────────────
# 메시지 포맷 (AI 없이 텍스트로 정리)
# ────────────────────────────────────────────
def format_message(results: list) -> str:
    today = datetime.today().strftime("%Y-%m-%d")
    lines = [
        f"📊 엔벨로프 조건검색 결과 [{today}]",
        f"조건: {ENV_PERIOD}일 SMA ±{int(ENV_BAND*100)}% 엔벨로프 + 거래량 {VOLUME_RATIO_MIN}배 이상",
        f"총 {len(results)}종목 발견\n",
        "─" * 35,
    ]
    for r in results:
        gc_mark  = "🟡골든크로스" if r["golden_cross"] else ""
        sw_mark  = "📐횡보돌파"   if r["sideways"]     else ""
        marks    = " ".join(filter(None, [gc_mark, sw_mark]))
        lines += [
            f"\n▶ {r['name']} ({r['ticker']})",
            f"   신호: {SIGNAL_KR.get(r['signal'], r['signal'])}",
            f"   현재가: {r['close']:,}원  |  중심선대비: {r['position_pct']:+.1f}%",
            f"   엔벨로프 상단: {r['env_upper']:,}원 / 하단: {r['env_lower']:,}원",
            f"   {marks}",
        ]
    lines += [
        "\n─" * 35,
        "※ 본 정보는 투자 참고용이며 투자 판단의 책임은 본인에게 있습니다.",
    ]
    return "\n".join(lines)


# ────────────────────────────────────────────
# 알림 발송
# ────────────────────────────────────────────
def send_email(subject: str, body: str):
    if not GMAIL_USER or not GMAIL_PASS:
        log.warning("Gmail 환경변수 미설정 — 이메일 생략")
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = GMAIL_USER
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.send_message(msg)
    log.info("이메일 발송 완료")


def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram 미설정 — 텔레그램 생략")
        return
    for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=10)
    log.info("텔레그램 발송 완료")


# ────────────────────────────────────────────
# 실행
# ────────────────────────────────────────────
def main():
    log.info("=== 엔벨로프 스크리너 시작 ===")
    results = run_screener()
    today   = datetime.today().strftime("%Y-%m-%d")

    if not results:
        msg = f"📊 [{today}] 오늘은 엔벨로프 조건에 맞는 종목이 없습니다."
        send_email(f"📊 [{today}] 엔벨로프 스크리닝 — 해당 종목 없음", msg)
        send_telegram(msg)
        return

    body    = format_message(results)
    subject = f"📊 [{today}] 엔벨로프 조건검색 — {len(results)}종목 발견"
    send_email(subject, body)
    send_telegram(body)
    log.info("=== 완료 ===")


if __name__ == "__main__":
    main()
