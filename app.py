import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="멀티자산 STEP 투자 대시보드", layout="wide")

TOTAL_CAPITAL = 800_000_000  # 8억

TARGET_WEIGHTS = {
    "SOXX": 0.30,
    "QQQ": 0.25,
    "SPY": 0.20,
    "BRK-B": 0.10,
    "CASH": 0.15,
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

# =========================
# 안전한 데이터 로드 (에러 방지 핵심)
# =========================
@st.cache_data(ttl=3600)
def load_data(ticker):
    df = yf.download(
        ticker,
        period="2y",
        auto_adjust=True,
        progress=False
    )

    # 멀티 인덱스 방지
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    close = df["Close"].astype(float)
    high = close.cummax()

    df["Drawdown"] = (close - high) / high
    return df

# =========================
# UI
# =========================
st.title("📊 멀티자산 STEP 투자 대시보드")
st.caption("SOXX / QQQ / SPY / BRK-B | 자산별 진입 타이밍 분리")

tickers = ["SOXX", "QQQ", "SPY", "BRK-B"]
cols = st.columns(len(tickers))

asset_data = {}

for i, ticker in enumerate(tickers):
    df = load_data(ticker)
    price = df.iloc[-1]["Close"]
    dd = df.iloc[-1]["Drawdown"]

    asset_data[ticker] = (df, price, dd)

    with cols[i]:
        st.metric(
            label=ticker,
            value=f"${price:,.2f}",
            delta=f"{dd*100:.2f}%"
        )

st.divider()

# =========================
# STEP 판단
# =========================
st.subheader("🚦 자산별 STEP 판단")

for ticker, (df, price, dd) in asset_data.items():
    step = "STEP0"
    if dd <= ASSET_TRIGGERS[ticker]["STEP2"]:
        step = "STEP2"
    elif dd <= ASSET_TRIGGERS[ticker]["STEP1"]:
        step = "STEP1"

    st.write(f"**{ticker}** → 현재 **{step}**")

st.divider()

# =========================
# STEP별 금액
# =========================
st.subheader("💰 STEP별 투입 금액 (총자금 8억 기준)")

for step, ratio in STEP_ALLOC.items():
    step_cap = TOTAL_CAPITAL * ratio
    st.markdown(f"### {step} : {step_cap:,.0f}원")

    for asset, w in TARGET_WEIGHTS.items():
        if asset == "CASH":
            continue
        st.write(f"- {asset}: {step_cap * w:,.0f}원")

st.divider()

# =========================
# 그래프
# =========================
st.subheader("📉 Drawdown 그래프")

for ticker, (df, _, _) in asset_data.items():
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Drawdown"] * 100, label="Drawdown (%)")
    ax.axhline(ASSET_TRIGGERS[ticker]["STEP1"] * 100, linestyle="--", label="STEP1")
    ax.axhline(ASSET_TRIGGERS[ticker]["STEP2"] * 100, linestyle="--", label="STEP2")
    ax.set_title(ticker)
    ax.legend()
    st.pyplot(fig)

# =========================
# STEP 설명
# =========================
st.divider()
st.subheader("📘 STEP 룰 요약")

st.markdown("""
**STEP0 (30%)**
- 시장 참여권 확보
- 추격 매수 방지

**STEP1 (추가 25%)**
- 의미 있는 조정
- 자산별 타이밍 분리

**STEP2 (추가 30%)**
- 공포 구간
- 기대값 최고

**현금 15%**
- 끝까지 남기는 옵션
""")
