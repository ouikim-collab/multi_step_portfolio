import json
import os
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd


# =========================
# 페이지 설정
# =========================
st.set_page_config(page_title="멀티자산 STEP 투자 대시보드", layout="wide")

# =========================
# 고정값
# =========================
TOTAL_CAPITAL = 800_000_000  # 8억

# 자산 비중 (현금은 계산에서 제외하되, 참고용으로만 남겨둠)
TARGET_WEIGHTS = {
    "SOXX": 0.30,
    "QQQ": 0.25,
    "SPY": 0.20,
    "BRK-B": 0.10,
    "CASH": 0.15,  # 화면에는 "최종 목표" 같은 문구로 안 보여줌
}

RISK_ASSETS = ["SOXX", "QQQ", "SPY", "BRK-B"]

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
    "STEP3": 0.15,   # 옵션카드(지금은 버튼만 제공, 추천 로직에는 기본 포함 안 함)
}

# 자동 새로고침(초)
REFRESH_SECONDS = 60

# 매수 체크 저장 파일 (Streamlit Cloud에서도 보통 동작하지만, 환경 초기화 시 리셋될 수 있음)
STATE_FILE = "buy_state.json"


# =========================
# 유틸
# =========================
def krw(x: float) -> str:
    return f"{x:,.0f}원"


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {"executed": {}}
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
    if dd <= ASSET_TRIGGERS[ticker]["STEP2"]:
        return "STEP2"
    if dd <= ASSET_TRIGGERS[ticker]["STEP1"]:
        return "STEP1"
    return "STEP0"


def step_amount(step: str) -> float:
    return TOTAL_CAPITAL * STEP_ALLOC[step]


def allocation_amount(ticker: str, step: str) -> float:
    # 해당 STEP에서 이 티커에 들어갈 금액
    return step_amount(step) * TARGET_WEIGHTS[ticker]


def next_recommended_step(current_step: str, executed: dict) -> str | None:
    """
    '현 시점부터 시작' 전제로:
    - 현재 시장이 STEP2여도, STEP0/1을 안 했다면 먼저 STEP0부터 추천
    - 추천은 한 번에 한 스텝(첫 미실행 스텝)만 제시
    """
    order = ["STEP0", "STEP1", "STEP2"]
    current_idx = order.index(current_step)

    for s in order[: current_idx + 1]:
        if not executed.get(s, False):
            return s

    return None  # 지금 할 것 없음(이미 따라잡음)


# =========================
# 데이터 로드: 2년 일봉(rolling high) + 1일 1분봉(현재가)
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
def load_intraday_price(ticker: str) -> tuple[float, str]:
    """
    장중이면 1분봉 마지막값을 '현재가'로 사용.
    장이 닫혀 있으면 마지막 종가를 사실상 현재가로 사용.
    """
    try:
        intraday = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.get_level_values(0)
        if not intraday.empty and intraday["Close"].dropna().shape[0] > 0:
            last_px = float(intraday["Close"].dropna().iloc[-1])
            last_ts = intraday.index[-1]
            label = f"1m last ({last_ts.strftime('%Y-%m-%d %H:%M')})"
            return last_px, label
    except Exception:
        pass

    d = load_daily_2y(ticker)
    last_px = float(d.iloc[-1]["Close"])
    return last_px, "daily close (fallback)"


# =========================
# 자동 새로고침(실시간 느낌)
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
st.caption("SOXX / QQQ / SPY / BRK-B | 자산별 진입 타이밍 분리 | 버튼 상태 기반 ‘지금 추천’")

state = load_state()

top1, top2, top3 = st.columns([1, 1, 2])
with top1:
    st.write(f"⏱ 자동 새로고침: **{REFRESH_SECONDS}초**")
with top2:
    if st.button("🔄 지금 새로고침"):
        st.rerun()
with top3:
    st.write(f"🗓 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.divider()

# =========================
# 현재가 + 드로다운(장중가 기준) 계산
# =========================
asset_data = {}
rows = []

for t in RISK_ASSETS:
    daily = load_daily_2y(t)
    live_px, live_label = load_intraday_price(t)

    rolling_high = float(daily["RollingHigh"].iloc[-1])
    dd_live = (live_px - rolling_high) / rolling_high

    step_now = decide_step(dd_live, t)
    ex = get_executed(state, t)

    rec_step = next_recommended_step(step_now, ex)
    rec_amount = allocation_amount(t, rec_step) if rec_step else 0.0

    asset_data[t] = {
        "live_px": live_px,
        "live_label": live_label,
        "rolling_high": rolling_high,
        "dd_live": dd_live,
        "step_now": step_now,
        "executed": ex,
        "rec_step": rec_step,
        "rec_amount": rec_amount,
    }

    rows.append({
        "Ticker": t,
        "Live Price": live_px,
        "Drawdown (live vs 2y high)": dd_live,
        "Current Step": step_now,
        "Next Recommendation": rec_step if rec_step else "대기(이미 따라잡음)",
        "Recommended Buy (KRW)": rec_amount,
    })

df_table = pd.DataFrame(rows)

# =========================
# 상단 카드 (현재값)
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

st.divider()

# =========================
# “지금 추천” 요약
# =========================
st.subheader("✅ 지금 추천 (버튼 체크 상태 반영)")
total_rec = float(df_table["Recommended Buy (KRW)"].sum())
st.markdown(f"### 오늘 추천 총액: **{krw(total_rec)}**")

# 추천 리스트를 더 직관적으로
for t in RISK_ASSETS:
    m = asset_data[t]
    if m["rec_step"] is None:
        st.write(f"- **{t}**: 대기 (현재 {m['step_now']}까지 이미 실행 체크됨)")
    else:
        st.write(f"- **{t}**: 지금은 **{m['rec_step']}** 추천 → **{krw(m['rec_amount'])}**")

st.divider()

# =========================
# 표 (상태 요약)
# =========================
st.subheader("📌 상태 요약표")
df_show = df_table.copy()
df_show["Drawdown (live vs 2y high)"] = (df_show["Drawdown (live vs 2y high)"] * 100).map(lambda x: f"{x:.2f}%")
df_show["Recommended Buy (KRW)"] = df_show["Recommended Buy (KRW)"].map(lambda x: krw(float(x)))
st.dataframe(df_show, use_container_width=True)

st.divider()

# =========================
# STEP별 매수 체크 버튼(티커별)
# =========================
st.subheader("🧷 STEP 매수 체크 (누른 상태에 따라 ‘지금 추천’이 바뀜)")
st.caption("현 시점부터 시작 전제: STEP0/1/2는 순서대로 따라잡는 방식으로 추천함.")

for t in RISK_ASSETS:
    m = asset_data[t]
    ex = m["executed"]

    box = st.container(border=True)
    with box:
        st.markdown(
            f"### {t} | 현재 STEP: **{m['step_now']}** | Drawdown: **{m['dd_live']*100:.2f}%**"
        )

        # 버튼 줄
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])

        with c1:
            if st.button(f"STEP0 {'✅' if ex['STEP0'] else '⬜'}", key=f"{t}_s0"):
                set_executed(state, t, "STEP0", not ex["STEP0"])
                save_state(state)
                st.rerun()

        with c2:
            if st.button(f"STEP1 {'✅' if ex['STEP1'] else '⬜'}", key=f"{t}_s1"):
                set_executed(state, t, "STEP1", not ex["STEP1"])
                save_state(state)
                st.rerun()

        with c3:
            if st.button(f"STEP2 {'✅' if ex['STEP2'] else '⬜'}", key=f"{t}_s2"):
                set_executed(state, t, "STEP2", not ex["STEP2"])
                save_state(state)
                st.rerun()

        with c4:
            if st.button(f"STEP3 {'✅' if ex['STEP3'] else '⬜'}", key=f"{t}_s3"):
                set_executed(state, t, "STEP3", not ex["STEP3"])
                save_state(state)
                st.rerun()

        with c5:
            if st.button("전체 리셋", key=f"{t}_reset"):
                for s in ["STEP0", "STEP1", "STEP2", "STEP3"]:
                    set_executed(state, t, s, False)
                save_state(state)
                st.rerun()

        # 이 티커의 "지금 추천" 한 줄
        if m["rec_step"] is None:
            st.write(f"➡️ 지금 추천: **대기** (이미 {m['step_now']}까지 체크됨)")
        else:
            st.write(f"➡️ 지금 추천: **{m['rec_step']}** 실행 → **{krw(m['rec_amount'])}**")

st.divider()

# =========================
# 최하단: 티커별 자금 투입 기준 룰북(깔끔 정리)
# =========================
st.subheader("📘 티커별 자금 투입 기준 (룰북)")

# 룰북 테이블 생성
rule_rows = []
for t in RISK_ASSETS:
    step0_amt = allocation_amount(t, "STEP0")
    step1_amt = allocation_amount(t, "STEP1")
    step2_amt = allocation_amount(t, "STEP2")

    rule_rows.append({
        "Ticker": t,
        "STEP0 (지금) 매수액": krw(step0_amt),
        "STEP1 트리거(DD)": f"{ASSET_TRIGGERS[t]['STEP1']*100:.0f}%",
        "STEP1 매수액": krw(step1_amt),
        "STEP2 트리거(DD)": f"{ASSET_TRIGGERS[t]['STEP2']*100:.0f}%",
        "STEP2 매수액": krw(step2_amt),
    })

rule_df = pd.DataFrame(rule_rows)
st.dataframe(rule_df, use_container_width=True)

st.markdown("""
**해석 방법**
- DD(드로다운) = 2년 롤링 고점 대비 하락률  
- 예: SOXX의 STEP1이 –12%면, **2년 고점 대비 –12% 이하로 내려오면 STEP1 금액 투입**  
- 현 시점부터 시작이므로, **현재 STEP이 높아도 STEP0 → STEP1 → STEP2 순서로 따라잡도록 추천** (버튼 체크 기반)
""")
