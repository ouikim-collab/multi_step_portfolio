"""
멀티자산 STEP 운용 모니터 (자산별 진입 타이밍 분리 버전)
- 포트폴리오: SOXX / QQQ / SPY / BRK-B / CASH
- "최종 비중"은 하나지만, "진입 타이밍"은 자산별로 따로 판단
- 트리거: 각 자산의 Rolling High 대비 Drawdown(고점 대비 하락률)

핵심 구조
1) 총자금 8억 중, 목표 현금 15%는 끝까지 남기는 설계
2) 실제로 '굴릴 자금(Deployable)' = 85% = 6.8억
3) Deployable을 STEP 트랜치로 쪼갬: 30% / 25% / 30% / 15%
4) 각 트랜치(POOL)는 "자산별 트리거가 충족되는 순간" 해당 자산에만 집행
   -> 그래서 자산별 진입 타이밍이 달라짐 (네가 말한 그 방식)

설치:
  pip install yfinance pandas numpy matplotlib

실행:
  python multi_step_portfolio.py
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# =========================
# 0) 설정 (여기만 바꾸면 됨)
# =========================

TOTAL_CAPITAL = 800_000_000  # 8억 원

# 최종 목표 비중 (엔드게임)
TARGET_WEIGHTS = {
    "SOXX": 0.30,
    "QQQ": 0.25,
    "SPY": 0.20,
    "BRK-B": 0.10,
    "CASH": 0.15,
}

RISK_ASSETS = ["SOXX", "QQQ", "SPY", "BRK-B"]

# STEP 트랜치 비중 (Deployable capital 기준)
# - Deployable = TOTAL * (1 - CASH_WEIGHT)
STEP_POOLS = {
    "POOL_0": 0.30,  # 시작 (즉시 집행)
    "POOL_1": 0.25,  # 1차 조정
    "POOL_2": 0.30,  # 공포 구간
    "POOL_3": 0.15,  # 옵션 카드 (수동/조건)
}

# 자산별 트리거 (drawdown: 고점 대비 하락률, 음수)
# 네가 말한 "타이밍 분리"를 구현한 핵심.
# - SOXX: 더 깊게, 더 빨리
# - QQQ: 중간
# - SPY: 보수
# - BRK-B: 타이밍 의미 낮음(방어성) -> 트리거 완화/혹은 POOL_2에서 제외 가능
ASSET_TRIGGERS = {
    "SOXX": {"POOL_1": -0.12, "POOL_2": -0.25, "POOL_3": -0.25},
    "QQQ":  {"POOL_1": -0.08, "POOL_2": -0.18, "POOL_3": -0.18},
    "SPY":  {"POOL_1": -0.06, "POOL_2": -0.12, "POOL_3": -0.12},
    "BRK-B":{"POOL_1": -0.05, "POOL_2": -0.10, "POOL_3": -0.10},
}

# POOL_3(옵션카드) 자동 집행 여부
AUTO_EXECUTE_POOL_3 = False

# 데이터 기간
HISTORY_PERIOD = "2y"

# 실행 상태 저장(중복 집행 방지)
USE_EXECUTION_GUARD = True
STATE_FILE = "portfolio_step_state.json"

# 그래프
SHOW_PLOTS = True
SAVE_PLOTS = False
PLOT_DIR = "plots"
PLOT_DPI = 160


# =========================
# 1) 데이터 로드 & 계산
# =========================

def fetch_history(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"데이터를 못 가져왔어: {ticker}")

    df = df.reset_index()  # KeyError('Date') 방지 (인덱스를 컬럼으로 내림)
    if "Close" not in df.columns:
        raise RuntimeError(f"{ticker}: Close 컬럼이 없어. 컬럼={list(df.columns)}")

    df["rolling_high"] = df["Close"].cummax()
    df["drawdown"] = (df["Close"] - df["rolling_high"]) / df["rolling_high"]
    return df


def current_metrics(ticker: str) -> Tuple[float, float, pd.DataFrame]:
    df = fetch_history(ticker, HISTORY_PERIOD)
    price = float(df.iloc[-1]["Close"])
    dd = float(df.iloc[-1]["drawdown"])
    return price, dd, df


# =========================
# 2) 상태 관리 (중복 집행 방지)
# =========================

def load_state(path: str) -> Dict:
    if not os.path.exists(path):
        return {"executed": {}}  # executed[ticker] = [pool_names...]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "executed" not in data:
            data["executed"] = {}
        return data
    except Exception:
        return {"executed": {}}


def save_state(path: str, state: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def is_executed(state: Dict, ticker: str, pool: str) -> bool:
    return pool in set(state.get("executed", {}).get(ticker, []))


def mark_executed(state: Dict, ticker: str, pool: str) -> Dict:
    executed = state.get("executed", {})
    executed.setdefault(ticker, [])
    s = set(executed[ticker])
    s.add(pool)
    executed[ticker] = sorted(list(s))
    state["executed"] = executed
    return state


# =========================
# 3) 자금 구조 (Deployable / Pools / 자산별 목표)
# =========================

def krw(x: float) -> str:
    return f"{x:,.0f}원"


def compute_deployable_capital() -> float:
    return TOTAL_CAPITAL * (1.0 - TARGET_WEIGHTS["CASH"])


def normalized_risk_weights() -> Dict[str, float]:
    # 현금 제외한 4자산 비중 합 = 0.85
    s = sum(TARGET_WEIGHTS[t] for t in RISK_ASSETS)
    return {t: TARGET_WEIGHTS[t] / s for t in RISK_ASSETS}


def pool_amounts(deployable: float) -> Dict[str, float]:
    return {p: deployable * r for p, r in STEP_POOLS.items()}


# =========================
# 4) 집행 로직 (자산별 타이밍 분리)
# =========================

@dataclass
class ExecutionPlan:
    ticker: str
    pool: str
    should_execute: bool
    reason: str
    amount: float


def decide_execution_plans(state: Dict) -> Tuple[Dict[str, float], Dict[str, float], List[ExecutionPlan], Dict[str, Dict]]:
    """
    반환:
    - deployable_total
    - pool_amount_dict
    - execution_plans (자산별/풀별 집행 여부 + 금액)
    - metrics (ticker -> {price, drawdown})
    """
    deployable = compute_deployable_capital()
    pools = pool_amounts(deployable)
    w = normalized_risk_weights()

    metrics: Dict[str, Dict] = {}
    for t in RISK_ASSETS:
        price, dd, _df = current_metrics(t)
        metrics[t] = {"price": price, "drawdown": dd}

    plans: List[ExecutionPlan] = []

    # POOL_0: 시작은 즉시 집행 (가드가 켜져 있으면 1회만)
    for t in RISK_ASSETS:
        already = is_executed(state, t, "POOL_0") if USE_EXECUTION_GUARD else False
        amt = pools["POOL_0"] * w[t]
        plans.append(ExecutionPlan(
            ticker=t,
            pool="POOL_0",
            should_execute=not already,
            reason="시작 포지션(POOL_0): 즉시 집행" + (" (이미 실행됨)" if already else ""),
            amount=0.0 if already else amt,
        ))

    # POOL_1 / POOL_2 / POOL_3: 자산별 트리거 충족 시 집행
    for pool in ["POOL_1", "POOL_2", "POOL_3"]:
        for t in RISK_ASSETS:
            # POOL_3는 기본 수동. AUTO_EXECUTE_POOL_3가 False면 "조건 충족해도 대기"
            if pool == "POOL_3" and not AUTO_EXECUTE_POOL_3:
                plans.append(ExecutionPlan(
                    ticker=t,
                    pool=pool,
                    should_execute=False,
                    reason="POOL_3(옵션카드): 자동 집행 OFF (수동 실행용)",
                    amount=0.0,
                ))
                continue

            already = is_executed(state, t, pool) if USE_EXECUTION_GUARD else False
            dd = metrics[t]["drawdown"]
            thr = ASSET_TRIGGERS[t][pool]

            if already:
                plans.append(ExecutionPlan(
                    ticker=t,
                    pool=pool,
                    should_execute=False,
                    reason=f"{pool}: 이미 실행됨",
                    amount=0.0,
                ))
                continue

            if dd <= thr:
                amt = pools[pool] * w[t]
                plans.append(ExecutionPlan(
                    ticker=t,
                    pool=pool,
                    should_execute=True,
                    reason=f"{pool}: 트리거 충족 ({dd*100:.2f}% <= {thr*100:.2f}%)",
                    amount=amt,
                ))
            else:
                plans.append(ExecutionPlan(
                    ticker=t,
                    pool=pool,
                    should_execute=False,
                    reason=f"{pool}: 대기 ({dd*100:.2f}% > {thr*100:.2f}%)",
                    amount=0.0,
                ))

    return deployable, pools, plans, metrics


# =========================
# 5) 출력 (요약 + 상세 + 최하단 STEP 설명)
# =========================

def print_header(deployable: float, pools: Dict[str, float]) -> None:
    print("\n" + "=" * 72)
    print("📌 멀티자산 STEP 운용 모니터 (자산별 진입 타이밍 분리)")
    print("=" * 72)
    print(f"총자금: {krw(TOTAL_CAPITAL)}")
    print(f"목표 현금(15%): {krw(TOTAL_CAPITAL * TARGET_WEIGHTS['CASH'])}")
    print(f"굴릴 자금(Deployable 85%): {krw(deployable)}")
    print("-" * 72)
    print("STEP 트랜치(Deployable 기준):")
    for p, amt in pools.items():
        print(f"  - {p}: {STEP_POOLS[p]*100:.0f}%  =>  {krw(amt)}")
    print("-" * 72)
    print("최종 비중(엔드게임): " + ", ".join([f"{k} {v*100:.0f}%" for k, v in TARGET_WEIGHTS.items()]))
    print("=" * 72)


def print_metrics(metrics: Dict[str, Dict]) -> None:
    print("\n📊 현재 지표 (각 자산별)")
    print("-" * 72)
    for t, m in metrics.items():
        print(f"- {t:<6}  Price={m['price']:,.2f}   Drawdown={m['drawdown']*100:,.2f}%")
    print("-" * 72)


def print_triggers() -> None:
    print("\n🎯 자산별 트리거(고점 대비 하락률 기준)")
    print("-" * 72)
    for t in RISK_ASSETS:
        tr = ASSET_TRIGGERS[t]
        print(
            f"- {t:<6}  "
            f"POOL_1 {tr['POOL_1']*100:>6.1f}% | "
            f"POOL_2 {tr['POOL_2']*100:>6.1f}% | "
            f"POOL_3 {tr['POOL_3']*100:>6.1f}%"
        )
    print("-" * 72)
    print("※ 숫자는 음수(–)가 정상. 더 작을수록(더 하락) 트리거 충족.\n")


def summarize_plans(plans: List[ExecutionPlan]) -> None:
    to_exec = [p for p in plans if p.should_execute and p.amount > 0]
    pending = [p for p in plans if (not p.should_execute) and p.pool != "POOL_3"]

    print("\n✅ 오늘 기준 '집행 추천' (조건 충족 + 미실행)")
    print("-" * 72)
    if not to_exec:
        print("  - 없음 (현재 조건에서 자동으로 실행할 트랜치가 없음)")
    else:
        for p in to_exec:
            print(f"  - {p.pool} / {p.ticker:<6}  =>  {krw(p.amount)}   ({p.reason})")

    print("\n🕒 대기 중(조건 미충족 or 이미 실행)")
    print("-" * 72)
    # 너무 길어질 수 있어 간단히 표시
    for p in pending:
        if "대기" in p.reason:
            print(f"  - {p.pool} / {p.ticker:<6}  ({p.reason})")


def print_step_explanations() -> None:
    print("\n" + "#" * 72)
    print("STEP 룰 설명 (핵심 디테일)")
    print("#" * 72)

    print("\n[POOL_0] 시작 트랜치 (30%)")
    print("- 목적: 포지션 '존재' 확보. 추격 매수/기회비용 리스크를 줄임.")
    print("- 특징: 트리거 없음. 1회만 실행(가드 ON 시).")

    print("\n[POOL_1] 1차 조정 트랜치 (25%)")
    print("- 목적: 흔들릴 때 규칙대로 추가해 평균단가 분산.")
    print("- 핵심: 자산별 트리거가 다름 (SOXX가 더 먼저/깊게 충족).")

    print("\n[POOL_2] 공포 트랜치 (30%)")
    print("- 목적: 기대값이 가장 좋은 가격대에서 가장 큰 화력을 씀.")
    print("- 핵심: 역시 자산별 트리거가 다름. 시장 공포는 SPY가 늦게 반영됨.")

    print("\n[POOL_3] 옵션 카드 (15%)")
    print("- 목적: -25% 이후 횡보/바닥 다지기 확인 등 '확인 후' 쓰는 남은 탄약.")
    print("- 기본: 자동 집행 OFF. (AUTO_EXECUTE_POOL_3=False)")

    print("\n[왜 자산별 타이밍 분리?]")
    print("- SOXX는 가장 먼저 깊게 빠지고, 가장 먼저 반등할 때가 많음.")
    print("- SPY/BRK는 방어적이라 같은 타이밍에 강공하면 오히려 비합리적일 수 있음.")
    print("- 그래서 STEP(돈 관리)과 Trigger(진입 타이밍)를 분리해서 운용함.")

    print("#" * 72 + "\n")


# =========================
# 6) 그래프 (각 자산별: 가격/드로다운)
# =========================

def ensure_plot_dir() -> None:
    if SAVE_PLOTS and not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR, exist_ok=True)


def plot_asset(ticker: str, df: pd.DataFrame) -> None:
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    x = pd.to_datetime(df[date_col])

    close = df["Close"].astype(float)
    high = df["rolling_high"].astype(float)
    dd = (df["drawdown"].astype(float) * 100.0)

    # 1) Price + Rolling High
    plt.figure()
    plt.plot(x, close, label=f"{ticker} Close")
    plt.plot(x, high, label="Rolling High")
    plt.title(f"{ticker} Price & Rolling High")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(os.path.join(PLOT_DIR, f"{ticker}_price.png"), dpi=PLOT_DPI)

    # 2) Drawdown + Thresholds (POOL_1/2/3)
    plt.figure()
    plt.plot(x, dd, label="Drawdown (%)")

    tr = ASSET_TRIGGERS.get(ticker, {})
    for pool_name, thr in tr.items():
        # POOL_0는 트리거 없음이라 표시 안 함
        if pool_name in ["POOL_1", "POOL_2", "POOL_3"]:
            plt.axhline(thr * 100.0, label=f"{pool_name} thr ({thr*100:.0f}%)")

    plt.scatter([x.iloc[-1]], [dd.iloc[-1]], label="Now")
    plt.title(f"{ticker} Drawdown & Pool Triggers")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()

    if SAVE_PLOTS:
        plt.savefig(os.path.join(PLOT_DIR, f"{ticker}_drawdown.png"), dpi=PLOT_DPI)


# =========================
# 7) (선택) 실행 기록 저장
# =========================

def record_execution_if_you_want(state: Dict, plans: List[ExecutionPlan]) -> None:
    """
    실제로 매수 '하고 나서'만 실행 기록을 남기는 게 안전.
    여기서는 자동 저장 안 하고,
    아래 주석 블록을 풀어서 "실제 매수 후"에 사용하도록 둠.
    """
    pass
    # for p in plans:
    #     if p.should_execute and p.amount > 0:
    #         state = mark_executed(state, p.ticker, p.pool)
    # save_state(STATE_FILE, state)
    # print(f"✅ 실행 상태 저장 완료: {STATE_FILE}")


# =========================
# 8) MAIN
# =========================

def main() -> None:
    state = load_state(STATE_FILE) if (USE_EXECUTION_GUARD) else {"executed": {}}

    deployable, pools, plans, metrics = decide_execution_plans(state)

    print_header(deployable, pools)
    print_metrics(metrics)
    print_triggers()
    summarize_plans(plans)

    # 그래프
    if SHOW_PLOTS or SAVE_PLOTS:
        ensure_plot_dir()
        for t in RISK_ASSETS:
            _price, _dd, df = current_metrics(t)
            plot_asset(t, df)
        if SHOW_PLOTS:
            plt.show()

    # 최하단 디테일 설명(요청사항)
    print_step_explanations()

    # (선택) 실제 매수 후에만 기록 저장하도록 주석 처리해둠
    # if USE_EXECUTION_GUARD:
    #     record_execution_if_you_want(state, plans)

    print("끝. (POOL_3는 기본 수동이야. AUTO_EXECUTE_POOL_3로 자동화 가능)")


if __name__ == "__main__":
    main()
