"""日频 A 股回测：昨收信号、今开撮合、现金和持仓逐日记账。"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd


@dataclass
class Config:
    initial_cash: float = 1_000_000
    rebalance: int = 5
    top_n: int = 50
    min_history: int = 60
    commission: float = 0.0003
    min_commission: float = 5
    slippage: float = 0.001
    participation: float = 0.01
    stale_days: int = 60
    drift: bool = False
    liquidate_at_end: bool = False
    liquidation_sessions: int = 1
    stale_policy: str = "writeoff"
    start: str = "2025-01-01"
    end: str = "2026-09-30"


def load_data(folder):
    data = {}
    for name in ("open", "close", "high", "low", "cumadj", "amt"):
        frame = pd.read_feather(Path(folder) / f"{name}.feather")
        frame = frame.set_index("date").sort_index()
        frame.index = pd.to_datetime(frame.index)
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError(f"{name}: 重复日期或代码")
        data[name] = frame
    base = data["close"]
    symbols = [s for s in base.columns if s.endswith((".SH", ".SZ")) and s[:1] in "036"]
    for name, frame in data.items():
        if not frame.index.equals(base.index) or set(frame.columns) != set(base.columns):
            raise ValueError(f"{name}: 日期或股票不一致")
        data[name] = frame[symbols].astype(float)
    return data


def limit_rate(symbol, date):
    if symbol.startswith("688"):
        return .20
    if symbol.startswith(("300", "301")) and date >= pd.Timestamp("2020-08-24"):
        return .20
    return .10


def fee(value, side, date, cfg):
    stamp = (0.0005 if date >= pd.Timestamp("2023-08-28") else .001) if side == "sell" else 0
    transfer = .00001 if date >= pd.Timestamp("2022-04-29") else .00002
    return max(cfg.min_commission, value * cfg.commission) + value * (stamp + transfer)


def buy_quantity(symbol, budget, price):
    # 科创板最低申报 200 股，之后可逐股递增；其他板块按 100 股整手。
    if symbol.startswith("688"):
        q = int(budget / price)
        return q if q >= 200 else 0
    return int(budget / price / 100) * 100


def run(data, cfg, metadata=None, scores=None, progress_callback=None, return_only=False, terminal_summary=False):
    if min(cfg.rebalance, cfg.top_n, cfg.min_history, cfg.stale_days) < 1:
        raise ValueError("窗口、数量和缺失阈值必须为正")
    if cfg.initial_cash <= 0 or not 0 < cfg.participation <= 1 or not 0 <= cfg.slippage < 1:
        raise ValueError("资金、成交参与率或滑点无效")
    if cfg.stale_policy not in ("writeoff", "carry"):
        raise ValueError("stale_policy 必须是 writeoff 或 carry")
    close = data["close"]
    dates, symbols = close.index, list(close.columns)
    arrays = {k: v.to_numpy() for k, v in data.items()}
    if scores is None:
        raise ValueError("必须提供外部信号 scores，回测引擎不生成默认信号")
    if not scores.index.equals(close.index) or not scores.columns.equals(close.columns):
        raise ValueError("信号矩阵必须与行情完全对齐")
    # 外部信号分数越高越优，缺失值不参与选股。
    signal = -scores.replace([np.inf, -np.inf], np.nan)
    history = close.notna().cumsum().to_numpy()
    sig = signal.to_numpy()
    if metadata is not None and metadata.assign(date=pd.to_datetime(metadata["date"])).duplicated(["date", "symbol"]).any():
        raise ValueError("元数据存在重复日期和代码")
    meta = {} if metadata is None else {(pd.Timestamp(r["date"]), r["symbol"]): r for r in metadata.to_dict("records")}
    n = len(symbols)
    qty = np.zeros(n)
    mark = np.full(n, np.nan)
    factor = np.full(n, np.nan)
    missing = np.zeros(n, dtype=int)
    impaired = np.zeros(n, dtype=bool)
    cash = cfg.initial_cash
    trades, events, curve = (DiscardLog(), DiscardLog(), DiscardLog()) if return_only or terminal_summary else ([], [], [])
    target = {}
    last_rebalance = None
    active_count = int(((dates >= pd.Timestamp(cfg.start)) & (dates <= pd.Timestamp(cfg.end))).sum())
    processed = 0
    terminal_orders = 0
    for t, date in enumerate(dates):
        if date < pd.Timestamp(cfg.start) or date > pd.Timestamp(cfg.end):
            continue
        processed += 1
        if progress_callback is not None and (processed == 1 or processed % 20 == 0 or processed == active_count):
            progress_callback(processed / max(active_count, 1))
        o, c, adj = (arrays[k][t] for k in ("open", "close", "cumadj"))
        # 累计复权因子只支持总收益等价持仓，不能区分现金分红、送转和配股。
        valid_factor = np.isfinite(adj) & (adj > 0)
        changed = valid_factor & np.isfinite(factor) & (factor > 0)
        ratio = np.ones(n)
        ratio[changed] = adj[changed] / factor[changed]
        qty *= ratio
        mark[changed] /= ratio[changed]
        factor[valid_factor] = adj[valid_factor]
        opening = np.where(np.isfinite(o) & (o > 0), o, mark)
        if cfg.stale_policy == "writeoff":
            opening[impaired & ~(np.isfinite(o) & (o > 0))] = 0
        equity_open = cash + np.nansum(qty * opening)
        liquidation_day = cfg.liquidate_at_end and active_count - processed < cfg.liquidation_sessions
        scheduled = liquidation_day or (t > 0 and (last_rebalance is None or t - last_rebalance >= cfg.rebalance))
        if scheduled:
            eligible = np.isfinite(sig[t-1]) & (history[t-1] >= cfg.min_history) & ~impaired
            # ST/退市状态仅使用信号日已知记录。
            for j, s in enumerate(symbols):
                m = meta.get((dates[t-1], s), {})
                if m.get("is_st", False) == True or m.get("delisted", False) == True:
                    eligible[j] = False
            candidates = np.flatnonzero(eligible)
            chosen = [] if liquidation_day else sorted(candidates, key=lambda j: (sig[t-1, j], symbols[j]))[:cfg.top_n]
            if liquidation_day:
                terminal_orders = int((qty > 1e-8).sum())
            target = {j: equity_open / len(chosen) for j in chosen} if chosen else {}
            if cfg.drift:
                # 保留继续入选的仓位；仅对新股分配一个名额的预算。
                target = {j: qty[j]*opening[j] if qty[j]>1e-8 else equity_open/cfg.top_n for j in chosen}
            held_at_check = qty.copy()
            planned_delta = {j:target.get(j,0)-qty[j]*o[j] for j in set(target)|set(np.flatnonzero(qty>1e-8))} if cfg.drift else {}
            last_rebalance = t
            # 先卖后买，一天内每只股票只走一个方向，天然满足股票 T+1。
            for side in ("sell", "buy"):
                for j in (range(n) if side == "sell" else target):
                    if side == "sell" and qty[j] <= 0:
                        continue
                    price = o[j]
                    m = meta.get((date, symbols[j]), {})
                    if m.get("delisted", False) == True:
                        continue
                    if cfg.drift:
                        if j in target and held_at_check[j]>1e-8:
                            continue
                        if (side=="buy" and planned_delta.get(j,0)<=0) or (side=="sell" and planned_delta.get(j,0)>=0):
                            continue
                    desired = target.get(j, 0)
                    delta = desired - qty[j] * price
                    if (side == "buy" and delta <= 0) or (side == "sell" and delta >= 0):
                        continue
                    # 买入同时检查执行日已生效的风险警示；卖出仍允许退出。
                    if side == "buy" and (m.get("is_st", False) == True or m.get("buy_eligible", True) == False):
                        events.append(dict(date=date, symbol=symbols[j], event="buy_ineligible", side=side))
                        continue
                    if not np.isfinite(price) or price <= 0 or m.get("suspended", False) == True:
                        events.append(dict(date=date, symbol=symbols[j], event="quote_block", side=side))
                        continue
                    if t == 0 or not np.isfinite(arrays["close"][t-1,j]) or not valid_factor[j]:
                        continue
                    prev_adj = arrays["cumadj"][t-1,j]
                    if not np.isfinite(prev_adj) or prev_adj <= 0:
                        continue
                    reference = arrays["close"][t-1,j] * prev_adj / adj[j]
                    rate = .05 if m.get("is_st", False) == True and not symbols[j].startswith(("688", "300", "301")) else limit_rate(symbols[j], date)
                    upper = m.get("limit_up", np.nan)
                    lower = m.get("limit_down", np.nan)
                    upper = upper if pd.notna(upper) else np.floor(reference * (1 + rate) * 100 + .5) / 100
                    lower = lower if pd.notna(lower) else np.floor(reference * (1 - rate) * 100 + .5) / 100
                    if (side == "buy" and price >= upper - .001) or (side == "sell" and price <= lower + .001):
                        events.append({"date": date, "symbol": symbols[j], "event": "limit_block", "side": side})
                        continue
                    # 使用前日成交额限制容量，不用当日完整成交额确定订单大小。
                    liquidity = arrays["amt"][t-1,j] * cfg.participation
                    if not np.isfinite(liquidity) or liquidity <= 0:
                        continue
                    execution = price * (1 + cfg.slippage if side == "buy" else 1 - cfg.slippage)
                    # 开盘撮合不读取当日收盘、全天成交额和事后高低价；仅受法定限价约束。
                    execution = min(execution, upper) if side == "buy" else max(execution, lower)
                    if not np.isfinite(execution) or execution <= 0:
                        continue
                    budget = min(abs(delta), liquidity)
                    if side == "buy":
                        budget = min(budget, max(0, cash - cfg.min_commission) / (1 + cfg.commission + .00002))
                        q = buy_quantity(symbols[j], budget, execution)
                    else:
                        q = min(qty[j], int(budget / execution / 100) * 100)
                        if desired == 0 and qty[j] * execution <= liquidity:
                            q = qty[j]
                    if q <= 0:
                        continue
                    value = q * execution
                    cost = fee(value, side, date, cfg)
                    if side == "buy" and value + cost > cash + 1e-8:
                        continue
                    cash += value - cost if side == "sell" else -value - cost
                    qty[j] += -q if side == "sell" else q
                    trades.append(dict(date=date, symbol=symbols[j], side=side, quantity=q, price=execution, value=value, fee=cost, signal_date=dates[t-1]))
                    if cfg.drift and side=="buy" and not np.isfinite(mark[j]):
                        mark[j]=price
        available = np.isfinite(c) & (c > 0)
        resumed = available & impaired & (qty > 0)
        for j in np.flatnonzero(resumed):
            events.append(dict(date=date, symbol=symbols[j], event="quote_resumed"))
        mark[available] = c[available]
        missing = np.where(available, 0, missing + 1)
        impaired[available] = False
        for j in np.flatnonzero((qty > 0) & (missing >= cfg.stale_days) & ~impaired):
            events.append(dict(date=date, symbol=symbols[j], event="stale_writeoff" if cfg.stale_policy == "writeoff" else "stale_carry", shares=qty[j], previous_mark=mark[j]))
            impaired[j] = True
        for j, s in enumerate(symbols):
            m = meta.get((date, s), {})
            if qty[j] > 0 and m.get("delisted", False) == True:
                recovery = m.get("recovery_price", 0)
                recovery = 0 if pd.isna(recovery) else float(recovery)
                cash += qty[j] * recovery
                events.append(dict(date=date, symbol=s, event="delisted_settlement", shares=qty[j], recovery_price=recovery))
                qty[j] = 0
        valuation = mark.copy()
        if cfg.stale_policy == "writeoff":
            valuation[impaired] = 0
        equity = cash + np.nansum(qty * valuation)
        curve.append(dict(date=date, equity=equity, cash=cash, holdings=int((qty > 1e-8).sum()), stale_value=float(np.nansum((qty * mark)[missing > 0]))))
        if cash < -1e-6 or (qty < -1e-6).any():
            raise AssertionError("现金或持仓出现负数")
    if return_only or terminal_summary:
        if not processed:
            raise ValueError("回测区间无数据")
        if terminal_summary:
            remaining = int((qty > 1e-8).sum())
            return dict(net_return=float(equity/cfg.initial_cash-1),
                        liquidation_orders=terminal_orders, residual_positions=remaining,
                        residual_market_value=float(np.nansum(qty * valuation)),
                        fully_liquidated=remaining == 0)
        return float(equity/cfg.initial_cash-1)
    nav = pd.DataFrame(curve).set_index("date")
    if nav.empty:
        raise ValueError("回测区间无数据")
    nav["nav"] = nav.equity / cfg.initial_cash
    nav["return"] = nav.equity.pct_change().fillna(nav.nav.iloc[0] - 1)
    nav["drawdown"] = nav.nav / nav.nav.cummax().clip(lower=1) - 1
    r = nav["return"]
    annual = nav.nav.iloc[-1] ** (252 / len(nav)) - 1
    summary = dict(config=asdict(cfg), start=str(nav.index[0].date()), end=str(nav.index[-1].date()), total_return=float(nav.nav.iloc[-1]-1), annual_return=float(annual), annual_volatility=float(r.std()*np.sqrt(252)), sharpe=float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else None, max_drawdown=float(nav.drawdown.min()), trades=len(trades), total_fees=float(sum(x["fee"] for x in trades)), final_equity=float(nav.equity.iloc[-1]), metadata_supplied=metadata is not None)
    positions = pd.DataFrame([dict(symbol=symbols[j], quantity=qty[j], last_price=mark[j], missing_days=missing[j], impaired=bool(impaired[j]), market_value=qty[j]*valuation[j]) for j in np.flatnonzero(qty>0)])
    return nav, pd.DataFrame(trades), pd.DataFrame(events), positions, summary


class DiscardLog:
    def append(self, row):
        pass


def save_result(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    nav, trades, events, positions, summary = result
    nav.to_csv(folder / "equity.csv", encoding="utf-8-sig")
    for name, table in (("trades", trades), ("events", events), ("positions", positions)):
        table.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(nav.index, nav.nav)
    axes[0].set_ylabel("NAV")
    axes[1].fill_between(nav.index, nav.drawdown, 0, alpha=.5)
    axes[1].set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(folder / "equity.png", dpi=150)
    plt.close(fig)
