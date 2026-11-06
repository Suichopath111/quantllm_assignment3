"""作业回测入口：仅导出区间级结果。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.engine import Config, run

ROOT = Path(__file__).resolve().parent
PERIODS = {
    "full": ("2025-01-02", "2026-09-30"),
    "2025": ("2025-01-02", "2025-12-31"),
    "2026": ("2026-01-01", "2026-09-30"),
}


def inputs(args, end):
    names = json.loads((ROOT / "universe.json").read_text(encoding="utf-8"))
    data = {}
    for key in ("open", "close", "high", "low", "cumadj", "amt"):
        frame = pd.read_feather(args.data / f"{key}.feather").set_index("date")
        frame.index = pd.to_datetime(frame.index)
        data[key] = frame.loc["2024-07-01":end, names].astype(float)
    close = data["close"]
    if not all(v.index.equals(close.index) and v.columns.equals(close.columns) for v in data.values()):
        raise ValueError("行情宽表未对齐")
    if close.index.max() != pd.Timestamp(end):
        raise ValueError("行情未覆盖所选区间末日")
    status = pd.read_parquet(args.status, filters=[
        ("date", ">=", pd.Timestamp("2024-07-01")), ("date", "<=", pd.Timestamp(end))])
    warm = pd.read_parquet(args.warmup_status)
    status = pd.concat([warm.loc[warm.date.lt(status.date.min())], status], ignore_index=True)
    if status.duplicated(["date", "symbol"]).any():
        raise ValueError("历史状态键重复")
    def matrix(column):
        return status.pivot(index="date", columns="symbol", values=column).reindex_like(close)
    st, trading = matrix("isST"), matrix("tradestatus")
    adj = close.where(close.gt(0)) * data["cumadj"].where(data["cumadj"].gt(0))
    r5 = (adj / adj.shift(5) - 1).where(adj.notna().rolling(6).sum().eq(6))
    eligible = st.eq(0) & trading.eq(1) & close.notna().cumsum().ge(60) & r5.notna() & data["amt"].gt(0)
    if "delisted" not in status:
        status["delisted"] = False
    sparse = status.loc[status.isST.eq(1) | status.tradestatus.eq(0) | status.delisted.eq(True)].copy()
    sparse["is_st"] = sparse.isST.eq(1)
    sparse["suspended"] = sparse.tradestatus.eq(0)
    sparse["buy_eligible"] = ~(sparse.is_st | sparse.suspended)
    missing = st.isna() & close.notna() & (close.index >= pd.Timestamp("2024-12-31"))[:, None]
    i, j = np.where(missing.to_numpy())
    absent = pd.DataFrame(dict(date=close.index[i], symbol=close.columns[j], buy_eligible=False))
    meta = pd.concat([sparse[["date", "symbol", "is_st", "suspended", "buy_eligible", "delisted"]], absent], ignore_index=True)
    return data, (-r5).where(eligible), meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--warmup-status", type=Path, required=True)
    parser.add_argument("--period", choices=PERIODS, default="full")
    parser.add_argument("--baseline", action="store_true", help="取消截尾，其他策略和撮合参数相同")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "strategy.json").read_text(encoding="utf-8"))
    trim = cfg.pop("trim")
    cfg["start"], cfg["end"] = PERIODS[args.period]
    data, score, metadata = inputs(args, cfg["end"])
    if not args.baseline:
        score = score.where(score.rank(axis=1, ascending=False, method="average", pct=True).gt(trim))
    active = data["close"].loc[cfg["start"]:cfg["end"]].index
    result_fields = run(data, Config(**cfg), metadata=metadata, scores=score, terminal_summary=True)
    if args.period == "2026":
        result_fields = {"net_return": result_fields["net_return"]}
    output = dict(strategy="baseline" if args.baseline else "T20_N2000_C10",
                  start=str(active[0].date()), end=str(active[-1].date()), sessions=len(active),
                  initial_cash=cfg["initial_cash"], **result_fields)
    result = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
