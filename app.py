import os
import json
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


# =========================
# 앱 설정
# =========================
st.set_page_config(page_title="STEP 변화 이메일 알림", layout="wide")

TOTAL_CAPITAL = 800_000_000
RISK_ASSETS = ["SOXX", "QQQ", "SPY", "BRK-B"]

TARGET_WEIGHTS = {
    "SOXX": 0.30,
    "QQQ": 0.25,
    "SPY": 0.20,
    "BRK-B": 0.10,
}

ASSET_TRIGGERS = {
    "SOXX": {"STEP1": -0.12, "STEP2": -0.25},
    "QQQ":  {"STEP1": -0.08, "STEP2": -0.18},
    "SPY":  {"STEP1": -0.06, "STEP2": -0.12},
    "BRK-B":{"STEP1": -0.05, "STEP2": -0.10},
}

STEP_ALLOC = {
    "STEP0": 0.30,
    "STEP1": 0.25,
    "STEP2": 0.30,
}

STATE_FILE = "step_state.json"
REFRESH_SECONDS = 120  # 2분마다 체크


# =========================
# Secrets에서 설정 읽기 (코드에 비밀값 0)
# =========================
# Streamlit Cloud → App settings → Secrets 에 아래를 넣어야 함:
# SENDGRID_API_KEY = "..."
# FROM_EMAIL = "..."
# TO_EMAIL = "ouikim@oui.kr"
def get_secret(name: str):
    v = st.secrets.get(name, None)
    if v is None:
        v = os.environ.get(name, None)
    return v


SENDGRID_API_KEY = get_secret("SENDGRID_API_KEY")
FROM_EMAIL = get_secret("FROM_EMAIL")
TO_EMAIL = get_secret("TO_EMAIL") or "ouikim@oui.kr"


# =========================
# 유틸
# =========================
def krw(x: float) -> str:
    return f"{x:,.0f}원"


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def decide_step(dd: float, ticker: str) -> str:
    if dd <= ASSET_TRIGGERS[ticker]["STEP2"]:
        return "STEP2"
    if dd <= ASSET_TRIGGERS[ticker]["STEP1"]:
        return "STEP1"
    return "STEP0"


def step_amount(ticker: str, step: str) -> float:
    # 해당 STEP에서 해당 티커에 들어갈 원화 금액
    return TOTAL_CAPITAL * STEP_ALLOC[step] * TARGET_WEIGHTS[ticker]


def send_email_sendgrid(subject: str, body: str):
    if not SENDGRID_API_KEY or not FROM_EMAIL or not TO_EMAIL:
        return False, "Secrets 누락: SENDGRID_API_KEY / FROM_EMAIL / TO_EMAIL"

    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=TO_EMAIL,
        subject=subject,
        plain_text_content=body
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        resp = sg.send(message)
        return True, f"SendGrid status={resp.status_code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# =========================
# 데이터 로드: 2년 일봉(고점) + 1분봉(현재가)
# =========================
@st.cache_data(ttl=3600)
def load_daily_2y(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"{ticker} 데이터 없음")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Close"] = df["Close"].astype(float)
    df["RollingHigh"] = df["Close"].cummax()
    return df


@st.cache_data(ttl=60)
def load_live_price(ticker: str):
    # 장중이면 1분봉 마지막값, 아니면 종가
    try:
        intraday = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.get_level_values(0)
        if not intraday.empty and intraday["Close"].dropna().shape[0] > 0:
            last_px = float(intraday["Close"].dropna().iloc[-1])
            last_ts = intraday.index[-1]
            return last_px, f"1m last ({last_ts.strftime('%Y-%m-%d %H:%M')})"
    except Exception:
        pass

    d = load_daily_2y(ticker)
    return float(d.iloc[-1]["Close"]), "daily close (fallback)"


# =========================
# 자동 새로고침
# =========================
components.html(
    "<script>setTimeout(function(){window.location.reload();}, "
    + str(REFRESH_SECONDS * 1000)
    + ");</script>",
    height=0
)


# =========================
# UI
# =========================
st.title("📩 STEP 변화 이메일 알림")
st.caption(f"받는 사람: {TO_EMAIL} | 자동 체크: {REFRESH_SECONDS}초 | STEP 변할 때만 발송")

if not (SENDGRID_API_KEY and FROM_EMAIL and TO_EMAIL):
    st.warning("현재 이메일 발송 비활성(Secrets 미설정). 하단 안내대로 Secrets만 넣으면 바로 활성화됨.")

st.divider()

state = load_state()
events = []

# 현재 상태 계산 + 변화 감지
for t in RISK_ASSETS:
    daily = load_daily_2y(t)
    live_px, src = load_live_price(t)

    rolling_high = float(daily["RollingHigh"].iloc[-1])
    dd = (live_px - rolling_high) / rolling_high
    new_step = decide_step(dd, t)

    prev_step = state.get(t)  # 이전 스텝(있으면 변화 감지 가능)

    st.write(f"**{t}** | ${live_px:,.2f} | DD {dd*100:.2f}% | STEP **{new_step}** ({src})")

    # 첫 실행(이전 값 없음)은 메일 발송 안 함: 기준값만 저장
    if prev_step and prev_step != new_step:
        amount = step_amount(t, new_step) if new_step in STEP_ALLOC else 0.0
        events.append({
            "ticker": t,
            "prev": prev_step,
            "new": new_step,
            "price": live_px,
            "dd": dd,
            "amount": amount,
        })

    # 상태 저장(다음 번 실행에서 비교용)
    state[t] = new_step

save_state(state)

st.divider()
st.subheader("✅ 감지 결과")

if not events:
    st.info("STEP 변화 없음 → 메일 발송 없음")
else:
    for e in events:
        subject = f"[STEP ALERT] {e['ticker']} {e['prev']} → {e['new']}"
        body = (
            f"티커: {e['ticker']}\n"
            f"STEP 변화: {e['prev']} → {e['new']}\n\n"
            f"현재 가격: ${e['price']:,.2f}\n"
            f"2년 고점 대비: {e['dd']*100:.2f}%\n\n"
            f"추천 매수 금액(해당 STEP): {krw(e['amount'])}\n\n"
            f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        ok, msg = send_email_sendgrid(subject, body)
        if ok:
            st.success(f"메일 발송 성공: {subject} ({msg})")
        else:
            st.error(f"메일 발송 실패: {subject} | {msg}")

st.divider()
st.subheader("🔧 너가 해야 할 것(최소)")

st.code(
    'SENDGRID_API_KEY = "여기에_키"\n'
    'FROM_EMAIL = "SendGrid에서_승인된_발신주소"\n'
    'TO_EMAIL = "ouikim@oui.kr"\n',
    language="toml"
)

st.write("Streamlit Cloud → App settings → Secrets에 위 3줄을 붙여넣으면 끝.")
