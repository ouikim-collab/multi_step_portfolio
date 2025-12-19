import json
import os
import time
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 페이지 설정
# =========================
st.set_page_config(page_title="멀티자산 STEP 투자 대시보드", layout="wide")

# =========================
# 고정값
# =========================
TOTAL_CAPITAL = 800_000_000  # 8억

TARGET_WEIGHTS = {
    "SOXX": 0.30,
    "QQQ": 0.25,
    "SPY": 0.20,
    "BRK-B": 0.10,
    "CASH": 0.15,
}

# 자산별 트리거(고점 대비 하락률 기준)
ASSET_TRIGGERS = {
    "SOXX": {"STEP1": -0.12, "STEP2": -0.25},
    "QQQ":  {"STEP1": -0.08, "STEP2": -0.18},
    "SPY":  {"STEP1": -0.06, "STEP2": -0.12},
    "BRK-B":{"STEP1": -0.05, "STEP2": -0.10},
}

# STEP별 자금 투입 비율(총자금 기준)
STEP_ALLOC = {
    "STEP0": 0.30,
    "STEP1": 0.25,
    "STEP2": 0.30,
    "STEP3": 0.15,   # 옵션카드 (수동/여유)
}

RISK_ASSETS = ["SOXX", "QQQ", "SPY", "BRK-B"]

# 매수 체크 저장 파일 (Streamlit Cloud에서도 동작, 단 재배포/환경 초기화 시 리셋될 수 있음)
STATE_FILE = "buy_state.json"

# 자동 새로고침(초) - 60초 추천
REFRESH_SECONDS = 60


# =========================
# 유틸
# =========================
def krw(x: float) -> str:
    return f"{x:,.0f}원"


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"executed": {}}  # executed[ticker] = {"STEP0": bool, "STEP1": bool, "STEP2": bool, "STEP3": bool}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"executed": {}}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_executed(state: dict, ticker: str) -> dict:
    ex = state.get("executed", {})
    if ticker not in ex:
        ex[ticker] = {"STEP0": False, "STEP1": False, "STEP2": False, "STEP3": False}
    return ex[ticker]


def set_executed(state: dict, ticker: str, step: str, value: bool) -> None:
    ex = state.get("executed", {})
    if ticker not in ex:
        ex[ticker] = {"STEP0": False, "STEP1": False, "STEP2": False, "STEP3": False}
    ex[ticker][step] = value
    state["executed"] = ex


def decide_step(dd: float, ticker: str) -> str:
    # dd는 음수(하락)일수록 작은 값
    if dd <= ASSET_TRIGGERS[ticker]["STEP2"]:
        return "STEP2"
    if dd <= ASSET_TRIGGERS[ticker]["STEP1"]:
        return "STEP1"
    return "STEP0"


# =========================
# 데이터 로드: (1) 2년 일봉으로 고점/차트, (2) 1일 1분봉으로 "현재가"
# =========================
@st.cache_data(ttl=3600)
def load_daily_2y(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Close"] = df["Close"].astype(float)
    df["RollingHigh"] = df["Close"].cummax()
    df["Drawdown"] = (df["Close"] - df["RollingHigh"]) / df["RollingHigh"]
    return df


@st.cache_data(ttl=60)
def load_intraday_price(ticker: str) -> tuple[float, str]:
    """
    장중이면 1분봉 마지막 가격을 '현재가'로 사용.
    장이 닫혀 있으면 마지막 종가(일봉 Close)를 사실상 현재가로 사용.
    """
    try:
        intraday = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.get_level_values(0)
        if not intraday.empty:
            last_px = float(intraday["Close"].dropna().iloc[-1])
            last_ts = intraday.index[-1]
            label = f"1m last ({last_ts.strftime('%Y-%m-%d %H:%M')})"
            return last_px, label
    except Exception:
        pass

    # fallback: 종가
    d = load_daily_2y(ticker)
    last_px = float(d.iloc[-1]["Close"])
    label = "daily close (fallback)"
    return last_px, label


# =========================
# 자동 새로고침 (진짜 '실시간' 느낌)
# =========================
components.html(
    f"""
    <script>
      setTimeout(function() {{
        window.location.reload();
      }}, {REFRESH_SECONDS * 1000});
    </script>
    """,
    height=0
)

# =========================
# UI 상단
# =========================
st.title("📊 멀티자산 STEP 투자 대시보드")
st.caption("SOXX / QQQ / SPY / BRK-B | 자산별 진입 타이밍 분리 | 장중 1분봉 기준 업데이트")

state = load_state()

# 상단 설정 바
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.write(f"⏱ 자동 새로고침: **{REFRESH_SECONDS}초**")
with c2:
    if st.button("🔄 지금 새로고침"):
        st.rerun()
with c3:
    st.write(f"🗓 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.divider()

# =========================
# 현재가 + 드로다운 계산(장중 가격 기준)
# =========================
asset_rows = []
asset_data = {}

for t in RISK_ASSETS:
    daily = load_daily_2y(t)
    live_px, live_label = load_intraday_price(t)

    rolling_high = float(daily["RollingHigh"].iloc[-1])
    dd_live = (live_px - rolling_high) / rolling_high

    step_now = decide_step(dd_live, t)

    asset_data[t] = {
        "daily": daily,
        "live_px": live_px,
        "live_label": live_label,
        "rolling_high": rolling_high,
        "dd_live": dd_live,
        "step_now": step_now,
    }

    asset_rows.append({
        "Ticker": t,
        "Live Price": live_px,
        "Live Source": live_label,
        "Rolling High(2y)": rolling_high,
        "Drawdown (live vs 2y high)": dd_live,
        "Step Now": step_now,
    })

df_table = pd.DataFrame(asset_rows)

# =========================
# 상단 카드(메트릭)
# =========================
cols = st.columns(4)
for i, t in enumerate(RISK_ASSETS):
    m = asset_data[t]
    with cols[i]:
        st.metric(
            label=t,
            value=f"${m['live_px']:,.2f}",
            delta=f"{m['dd_live']*100:.2f}% (vs 2y high)"
        )
        st.caption(m["live_label"])
        st.caption(f"Rolling high(2y): ${m['rolling_high']:,.2f}")

st.divider()

# =========================
# 표
# =========================
st.subheader("📌 현재 상태 요약")
st.dataframe(
    df_table.assign(**{
        "Drawdown (live vs 2y high)": (df_table["Drawdown (live vs 2y high)"] * 100).map(lambda x: f"{x:.2f}%")
    }),
    use_container_width=True
)

st.divider()

# =========================
# STEP별 투입 금액(총액) + 자산별 분배
# =========================
st.subheader("💰 STEP별 투입 금액 (총자금 8억 기준)")

for step, ratio in STEP_ALLOC.items():
    step_cap = TOTAL_CAPITAL * ratio
    with st.expander(f"{step} : {krw(step_cap)}", expanded=(step == "STEP0")):
        for asset, w in TARGET_WEIGHTS.items():
            if asset == "CASH":
                continue
            st.write(f"- {asset}: {krw(step_cap * w)}")
        st.write(f"- CASH(최종 목표): {krw(TOTAL_CAPITAL * TARGET_WEIGHTS['CASH'])}")

st.divider()

# =========================
# ✅ 매수 체크 버튼(자산별 STEP0/1/2/3)
# =========================
st.subheader("✅ STEP별 매수 체크 (버튼으로 기록)")

st.caption("각 자산마다 STEP이 따로 올 수 있어. (예: BRK가 STEP1인데 SOXX는 STEP0인 경우)")

for t in RISK_ASSETS:
    ex = get_executed(state, t)
    m = asset_data[t]

    box = st.container(border=True)
    with box:
        st.markdown(f"### {t}  |  현재 추천 STEP: **{m['step_now']}**  |  Drawdown: **{m['dd_live']*100:.2f}%**")

        b1, b2, b3, b4, b5 = st.columns([1,1,1,1,2])

        with b1:
            if st.button(f"{t} STEP0 {'✅' if ex['STEP0'] else '⬜'}", key=f"{t}_s0"):
                set_executed(state, t, "STEP0", not ex["STEP0"])
                save_state(state)
                st.rerun()

        with b2:
            if st.button(f"{t} STEP1 {'✅' if ex['STEP1'] else '⬜'}", key=f"{t}_s1"):
                set_executed(state, t, "STEP1", not ex["STEP1"])
                save_state(state)
                st.rerun()

        with b3:
            if st.button(f"{t} STEP2 {'✅' if ex['STEP2'] else '⬜'}", key=f"{t}_s2"):
                set_executed(state, t, "STEP2", not ex["STEP2"])
                save_state(state)
                st.rerun()

        with b4:
            if st.button(f"{t} STEP3 {'✅' if ex['STEP3'] else '⬜'}", key=f"{t}_s3"):
                set_executed(state, t, "STEP3", not ex["STEP3"])
                save_state(state)
                st.rerun()

        with b5:
            if st.button(f"{t} 전체 리셋", key=f"{t}_reset"):
                set_executed(state, t, "STEP0", False)
                set_executed(state, t, "STEP1", False)
                set_executed(state, t, "STEP2", False)
                set_executed(state, t, "STEP3", False)
                save_state(state)
                st.rerun()

st.divider()

# =========================
# 그래프 (업데이트는 새로고침마다 갱신됨)
# =========================
st.subheader("📉 Drawdown 그래프 (2년 일봉 + 현재가 기준선)")
st.caption("그래프 자체는 그릴 때마다 스냅샷이지만, 위에서 자동 새로고침으로 계속 업데이트됨.")

for t in RISK_ASSETS:
    d = asset_data[t]["daily"]
    live_dd = asset_data[t]["dd_live"]

    fig, ax = plt.subplots()
    ax.plot(d["Date"], d["Drawdown"] * 100, label="Daily drawdown (%)")
    ax.axhline(ASSET_TRIGGERS[t]["STEP1"] * 100, linestyle="--", label="STEP1")
    ax.axhline(ASSET_TRIGGERS[t]["STEP2"] * 100, linestyle="--", label="STEP2")

    # 현재가 drawdown 점 (장중 기준)
    ax.scatter([d["Date"].iloc[-1]], [live_dd * 100], label="Now (live vs 2y high)")

    ax.set_title(t)
    ax.legend()
    st.pyplot(fig)

st.divider()

# =========================
# 룰 설명
# =========================
st.subheader("📘 룰 요약")
st.markdown("""
- **STEP0 (30%)**: 자리 확보 (추격 매수 방지)
- **STEP1 (+25%)**: 의미 있는 조정 구간 (자산별 트리거 다름)
- **STEP2 (+30%)**: 공포 구간 (기대값 최고)
- **STEP3 (+15%)**: 옵션 카드 (바닥 다지기 확인 후 수동)
- **현금 15%**: 최종 목표로 남겨두는 옵션
""")
