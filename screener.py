"""
Korean Stock Screener with Envelope Filter
조건: 이동평균 엔벨로프 + 기존 MA Squeeze + 거래량 필터
"""

import os
import time
import logging
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

try:
    from pykrx import stock as pykrx_stock
except ImportError:
    pykrx_stock = None

try:
    import anthropic
except ImportError:
    anthropic = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ────────────────────────────────────────────
# 설정값 (GitHub Secrets → 환경변수)
# ────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GMAIL_USER        = os.environ.get("GMAIL_USER", "")
GMAIL_PASS        = os.environ.get("GMAIL_PASS", "")
TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")

# ────────────────────────────────────────────
# 엔벨로프 파라미터
# ────────────────────────────────────────────
ENV_PERIOD = 20        # 이동평균 기간 (일)
ENV_BAND   = 0.06      # 밴드폭 ±6% (원하는 값으로 조정)

# 기존 MA Squeeze 파라미터
MA_SHORT   = 5
MA_LONG    = 20
SIDEWAYS_DAYS    = 10   # 횡보 판정 기간
SIDEWAYS_RANGE   = 0.05 # 횡보 범위 ±5%
VOLUME_RATIO_MIN = 1.5  # 거래량 전일 대비 최소 배수


# ────────────────────────────────────────────
# 데이터 수집
# ────────────────────────────────────────────
def get_ticker_list() -> list[str]:
    """KOSPI + KOSDAQ 전 종목 코드 반환"""
    today = datetime.today().strftime("%Y%m%d")
    kospi  = pykrx_stock.get_market_ticker_list(today, market="KOSPI")
    kosdaq = pykrx_stock.get_market_ticker_list(today, market="KOSDAQ")
    tickers = kospi + kosdaq
    log.info(f"전체 종목 수: {len(tickers)}")
    return tickers


def fetch_ohlcv(ticker: str, days: int = 60) -> pd.DataFrame:
    """종목 OHLCV 데이터 조회 (최근 N일)"""
    end   = datetime.today()
    start = end - timedelta(days=days * 2)  # 주말·공휴일 여유
    df = pykrx_stock.get_market_ohlcv_by_date(
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        ticker,
    )
    if df is None or len(df) < days:
        return pd.DataFrame()
    return df.tail(days).copy()


# ────────────────────────────────────────────
# 지표 계산
# ────────────────────────────────────────────
def calc_envelope(df: pd.DataFrame, period: int = ENV_PERIOD, band: float = ENV_BAND) -> pd.DataFrame:
    """
    엔벨로프(Envelope) 계산
      - 중심선: period 일 단순이동평균(SMA)
      - 상단밴드: 중심선 × (1 + band)
      - 하단밴드: 중심선 × (1 - band)
    """
    df = df.copy()
    df["env_mid"]  = df["종가"].rolling(period).mean()
    df["env_upper"] = df["env_mid"] * (1 + band)
    df["env_lower"] = df["env_mid"] * (1 - band)
    return df


def calc_ma(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma_short"] = df["종가"].rolling(MA_SHORT).mean()
    df["ma_long"]  = df["종가"].rolling(MA_LONG).mean()
    return df


def envelope_signal(df: pd.DataFrame) -> dict | None:
    """
    엔벨로프 기반 진입 신호 탐지
    반환값: 신호 dict 또는 None (조건 미충족)

    신호 종류:
      lower_touch  : 하단밴드 터치 후 반등 (역추세 매수)
      mid_breakout : 중심선 상향 돌파 (추세 복귀 매수)
      upper_break  : 상단밴드 돌파 (강세 모멘텀)
    """
    if len(df) < ENV_PERIOD + 5:
        return None

    latest   = df.iloc[-1]
    prev     = df.iloc[-2]
    close    = latest["종가"]
    env_mid  = latest["env_mid"]
    env_upper = latest["env_upper"]
    env_lower = latest["env_lower"]

    if pd.isna(env_mid):
        return None

    position_pct = (close - env_mid) / env_mid * 100  # 중심선 대비 위치(%)

    signal = None

    # 1) 하단밴드 터치 반등
    #    전일 종가 ≤ 하단밴드, 당일 종가 > 하단밴드
    if prev["종가"] <= prev["env_lower"] and close > env_lower:
        signal = "lower_touch"

    # 2) 중심선 상향 돌파
    #    전일 종가 < 중심선, 당일 종가 ≥ 중심선
    elif prev["종가"] < prev["env_mid"] and close >= env_mid:
        signal = "mid_breakout"

    # 3) 상단밴드 돌파 (강세 모멘텀)
    #    전일 종가 < 상단밴드, 당일 종가 ≥ 상단밴드
    elif prev["종가"] < prev["env_upper"] and close >= env_upper:
        signal = "upper_break"

    if signal is None:
        return None

    return {
        "signal":       signal,
        "close":        close,
        "env_mid":      round(env_mid, 0),
        "env_upper":    round(env_upper, 0),
        "env_lower":    round(env_lower, 0),
        "position_pct": round(position_pct, 2),
    }


def is_sideways(df: pd.DataFrame) -> bool:
    """최근 SIDEWAYS_DAYS 동안 종가 변동폭이 ±SIDEWAYS_RANGE 이내"""
    recent = df["종가"].tail(SIDEWAYS_DAYS)
    if len(recent) < SIDEWAYS_DAYS:
        return False
    high = recent.max()
    low  = recent.min()
    mid  = recent.mean()
    return (high - low) / mid <= SIDEWAYS_RANGE * 2


def volume_surge(df: pd.DataFrame) -> bool:
    """당일 거래량이 전일 대비 VOLUME_RATIO_MIN 배 이상"""
    if len(df) < 2:
        return False
    today_vol = df["거래량"].iloc[-1]
    prev_vol  = df["거래량"].iloc[-2]
    if prev_vol == 0:
        return False
    return today_vol / prev_vol >= VOLUME_RATIO_MIN


def golden_cross(df: pd.DataFrame) -> bool:
    """단기 MA가 장기 MA를 상향 돌파 (골든크로스)"""
    if len(df) < 2:
        return False
    cur  = df.iloc[-1]
    prev = df.iloc[-2]
    return (prev["ma_short"] <= prev["ma_long"]) and (cur["ma_short"] > cur["ma_long"])


# ────────────────────────────────────────────
# 스크리닝 메인 로직
# ────────────────────────────────────────────
def screen_ticker(ticker: str) -> dict | None:
    """
    단일 종목 스크리닝
    조건: 엔벨로프 신호 + (횡보 OR 골든크로스) + 거래량 급증
    """
    try:
        df = fetch_ohlcv(ticker, days=60)
        if df.empty:
            return None

        df = calc_envelope(df)
        df = calc_ma(df)

        env = envelope_signal(df)
        if env is None:
            return None

        # 엔벨로프 신호 필수, 거래량 급증 필수
        if not volume_surge(df):
            return None

        # 횡보 또는 골든크로스 중 하나 이상
        sideways = is_sideways(df)
        gc       = golden_cross(df)
        if not (sideways or gc):
            return None

        # 종목명 조회
        name = pykrx_stock.get_market_ticker_name(ticker)

        return {
            "ticker":       ticker,
            "name":         name,
            "signal":       env["signal"],
            "close":        env["close"],
            "env_mid":      env["env_mid"],
            "env_upper":    env["env_upper"],
            "env_lower":    env["env_lower"],
            "position_pct": env["position_pct"],
            "sideways":     sideways,
            "golden_cross": gc,
        }

    except Exception as e:
        log.debug(f"{ticker} 처리 오류: {e}")
        return None


def run_screener() -> list[dict]:
    tickers = get_ticker_list()
    results = []
    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            log.info(f"진행 중: {i}/{len(tickers)}")
        result = screen_ticker(ticker)
        if result:
            results.append(result)
        time.sleep(0.05)  # API 과부하 방지
    log.info(f"조건 충족 종목: {len(results)}개")
    return results


# ────────────────────────────────────────────
# AI 요약 (Claude API)
# ────────────────────────────────────────────
SIGNAL_KR = {
    "lower_touch":  "하단밴드 터치 반등",
    "mid_breakout": "중심선 상향 돌파",
    "upper_break":  "상단밴드 돌파(강세)",
}

def summarize_with_claude(results: list[dict]) -> str:
    if not anthropic or not ANTHROPIC_API_KEY:
        return _format_plain(results)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    stocks_text = "\n".join(
        f"- {r['name']}({r['ticker']}): 신호={SIGNAL_KR.get(r['signal'], r['signal'])}, "
        f"현재가={r['close']:,}원, 중심선={r['env_mid']:,}원, "
        f"중심선대비={r['position_pct']:+.1f}%, "
        f"골든크로스={'O' if r['golden_cross'] else 'X'}, 횡보={'O' if r['sideways'] else 'X'}"
        for r in results
    )
    prompt = f"""오늘 엔벨로프 조건에 부합한 한국 주식 종목입니다.
엔벨로프 기간: {ENV_PERIOD}일, 밴드폭: ±{int(ENV_BAND*100)}%

{stocks_text}

각 종목의 신호 특성과 주목할 점을 간결하게 정리하고,
오늘 가장 주목할 종목 3개를 추천 이유와 함께 알려주세요."""

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _format_plain(results: list[dict]) -> str:
    lines = [f"[엔벨로프 조건검색 결과] {datetime.today().strftime('%Y-%m-%d')}\n"]
    lines.append(f"조건: {ENV_PERIOD}일 SMA ±{int(ENV_BAND*100)}% 엔벨로프 신호 + 거래량 급증\n")
    for r in results:
        lines.append(
            f"▶ {r['name']}({r['ticker']}) | {SIGNAL_KR.get(r['signal'], r['signal'])} | "
            f"현재가 {r['close']:,}원 | 중심선대비 {r['position_pct']:+.1f}% | "
            f"골든크로스: {'O' if r['golden_cross'] else 'X'}"
        )
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
        log.warning("Telegram 환경변수 미설정 — 텔레그램 생략")
        return
    # 텔레그램 4096자 제한
    for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=10)
    log.info("텔레그램 발송 완료")


# ────────────────────────────────────────────
# 엔트리포인트
# ────────────────────────────────────────────
def main():
    log.info("=== 엔벨로프 조건검색 스크리너 시작 ===")
    results = run_screener()

    if not results:
        msg = f"[{datetime.today().strftime('%Y-%m-%d')}] 오늘은 조건에 맞는 종목이 없습니다."
        send_email("📊 오늘의 엔벨로프 스크리닝 결과", msg)
        send_telegram(msg)
        return

    summary = summarize_with_claude(results)
    today   = datetime.today().strftime("%Y-%m-%d")
    subject = f"📊 [{today}] 엔벨로프 조건검색 — {len(results)}종목 발견"

    send_email(subject, summary)
    send_telegram(summary)
    log.info("=== 완료 ===")


if __name__ == "__main__":
    main()
