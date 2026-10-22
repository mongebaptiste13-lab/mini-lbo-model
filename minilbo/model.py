"""Simplified leveraged buyout (LBO) model.

Cash-free / debt-free acquisition financed with a senior term loan (mandatory amortisation plus
cash sweep), a subordinated bullet loan and sponsor equity as the balancing item. Amounts are in
millions. Interest is charged on opening balances, which avoids circular references.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq


@dataclass(frozen=True)
class LBOInputs:
    # Entry
    ltm_revenue: float
    ltm_ebitda_margin: float
    entry_multiple: float
    transaction_fees_pct: float        # % of enterprise value
    financing_fees_pct: float          # % of total debt raised
    min_cash: float                    # cash left on the balance sheet at closing and kept as a floor
    # Financing
    senior_multiple: float             # x LTM EBITDA
    senior_rate: float
    senior_amortisation_pct: float     # mandatory repayment, % of original senior per year
    cash_sweep_pct: float              # share of excess cash used to repay senior debt
    sub_multiple: float
    sub_rate: float
    # Operating forecast
    revenue_growth: Sequence[float]
    ebitda_margin: Sequence[float]
    da_pct_revenue: float
    capex_pct_revenue: float
    nwc_pct_revenue: float
    tax_rate: float
    # Exit
    exit_multiple: float
    exit_year: int

    def __post_init__(self):
        if len(self.revenue_growth) != len(self.ebitda_margin):
            raise ValueError("revenue_growth and ebitda_margin must have the same length")
        if not 1 <= self.exit_year <= len(self.revenue_growth):
            raise ValueError("exit_year must be within the forecast horizon")

    @property
    def ltm_ebitda(self) -> float:
        return self.ltm_revenue * self.ltm_ebitda_margin

    @property
    def enterprise_value(self) -> float:
        return self.ltm_ebitda * self.entry_multiple


@dataclass(frozen=True)
class LBOResult:
    sources_uses: pd.DataFrame
    projections: pd.DataFrame           # one row per year (operating, debt, credit metrics)
    sponsor_equity: float
    exit_enterprise_value: float
    exit_net_debt: float
    exit_equity: float
    moic: float
    irr: float
    bridge: pd.Series


def sources_and_uses(i: LBOInputs) -> pd.DataFrame:
    senior = i.senior_multiple * i.ltm_ebitda
    sub = i.sub_multiple * i.ltm_ebitda
    debt = senior + sub
    uses = {
        "Purchase of enterprise": i.enterprise_value,
        "Transaction fees": i.transaction_fees_pct * i.enterprise_value,
        "Financing fees": i.financing_fees_pct * debt,
        "Cash to balance sheet": i.min_cash,
    }
    equity = sum(uses.values()) - debt
    if equity <= 0:
        raise ValueError("Debt exceeds total uses: sponsor equity would be negative")
    sources = {"Senior term loan": senior, "Subordinated loan": sub, "Sponsor equity": equity}
    rows = [(k, v, "Source") for k, v in sources.items()] + [(k, v, "Use") for k, v in uses.items()]
    df = pd.DataFrame(rows, columns=["item", "amount", "side"])
    df["pct_of_total"] = df["amount"] / df.groupby("side")["amount"].transform("sum")
    return df


def irr(cash_flows: Sequence[float]) -> float:
    """Internal rate of return of annual cash flows (first element at t=0)."""
    cf = np.asarray(cash_flows, dtype=float)
    f = lambda r: np.sum(cf / (1 + r) ** np.arange(len(cf)))
    return brentq(f, -0.99, 10.0)


def run_lbo(i: LBOInputs) -> LBOResult:
    su = sources_and_uses(i)
    amounts = su.set_index("item")["amount"]
    senior0, sub0, equity0 = amounts["Senior term loan"], amounts["Subordinated loan"], amounts["Sponsor equity"]

    n = len(i.revenue_growth)
    rev_prev = i.ltm_revenue
    senior, sub, cash = senior0, sub0, i.min_cash
    rows = []
    for t in range(1, n + 1):
        revenue = rev_prev * (1 + i.revenue_growth[t - 1])
        ebitda = revenue * i.ebitda_margin[t - 1]
        da = revenue * i.da_pct_revenue
        ebit = ebitda - da
        interest_senior = senior * i.senior_rate
        interest_sub = sub * i.sub_rate
        interest = interest_senior + interest_sub
        taxes = max(ebit - interest, 0.0) * i.tax_rate
        capex = revenue * i.capex_pct_revenue
        d_nwc = i.nwc_pct_revenue * (revenue - rev_prev)
        fcf_pre_debt = ebitda - taxes - capex - d_nwc               # before interest and repayment
        cash_flow_after_interest = fcf_pre_debt - interest

        mandatory = min(senior, i.senior_amortisation_pct * senior0)
        cash_after_mandatory = cash + cash_flow_after_interest - mandatory
        excess = max(cash_after_mandatory - i.min_cash, 0.0)
        sweep = min(senior - mandatory, i.cash_sweep_pct * excess)
        senior_end = senior - mandatory - sweep
        cash_end = cash_after_mandatory - sweep

        net_debt = senior_end + sub - cash_end
        rows.append(dict(
            year=t, revenue=revenue, ebitda=ebitda, ebit=ebit, interest=interest, taxes=taxes,
            capex=capex, delta_nwc=d_nwc, free_cash_flow_after_interest=cash_flow_after_interest,
            mandatory_repayment=mandatory, cash_sweep=sweep, senior_debt=senior_end,
            subordinated_debt=sub, cash=cash_end, net_debt=net_debt,
            net_debt_to_ebitda=net_debt / ebitda, interest_cover=ebitda / interest if interest else np.nan,
            liquidity_shortfall=cash_after_mandatory < 0,
        ))
        senior, cash, rev_prev = senior_end, cash_end, revenue

    proj = pd.DataFrame(rows).set_index("year")
    exit_row = proj.loc[i.exit_year]
    exit_ev = exit_row["ebitda"] * i.exit_multiple
    exit_net_debt = exit_row["net_debt"]
    exit_equity = exit_ev - exit_net_debt
    moic = exit_equity / equity0
    cash_flows = [-equity0] + [0.0] * (i.exit_year - 1) + [exit_equity]
    result_irr = irr(cash_flows) if exit_equity > 0 else -1.0

    res = LBOResult(su, proj, equity0, exit_ev, exit_net_debt, exit_equity, moic, result_irr, pd.Series(dtype=float))
    return replace(res, bridge=value_creation_bridge(i, res))


def value_creation_bridge(i: LBOInputs, r: LBOResult) -> pd.Series:
    """Decompose the equity gain: EBITDA growth + multiple expansion + deleveraging - fees."""
    exit_ebitda = r.projections.loc[i.exit_year, "ebitda"]
    entry_net_debt = r.sources_uses.set_index("item").loc[["Senior term loan", "Subordinated loan"], "amount"].sum() - i.min_cash
    fees = i.transaction_fees_pct * i.enterprise_value + (
        i.financing_fees_pct * (i.senior_multiple + i.sub_multiple) * i.ltm_ebitda)
    return pd.Series({
        "EBITDA growth": (exit_ebitda - i.ltm_ebitda) * i.entry_multiple,
        "Multiple expansion": (i.exit_multiple - i.entry_multiple) * exit_ebitda,
        "Net debt reduction": entry_net_debt - r.exit_net_debt,
        "Fees": -fees,
    })


def sensitivity_entry_exit(i: LBOInputs, entry_multiples: Iterable[float], exit_multiples: Iterable[float],
                           metric: str = "irr") -> pd.DataFrame:
    """IRR or MOIC for entry multiple x exit multiple. Debt stays fixed in x EBITDA, equity adjusts."""
    entry_multiples, exit_multiples = list(entry_multiples), list(exit_multiples)
    grid = np.array([[getattr(run_lbo(replace(i, entry_multiple=e, exit_multiple=x)), metric)
                      for x in exit_multiples] for e in entry_multiples])
    return pd.DataFrame(grid, index=pd.Index(entry_multiples, name="Entry EV/EBITDA"),
                        columns=pd.Index(exit_multiples, name="Exit EV/EBITDA"))


def sensitivity_exit_year(i: LBOInputs, years: Iterable[int], exit_multiples: Iterable[float]) -> pd.DataFrame:
    years, exit_multiples = list(years), list(exit_multiples)
    grid = np.array([[run_lbo(replace(i, exit_year=y, exit_multiple=x)).irr for x in exit_multiples] for y in years])
    return pd.DataFrame(grid, index=pd.Index(years, name="Exit year"), columns=pd.Index(exit_multiples, name="Exit EV/EBITDA"))


def load_inputs(path: str | Path) -> tuple[LBOInputs, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = d.pop("meta", {})
    d["revenue_growth"], d["ebitda_margin"] = tuple(d["revenue_growth"]), tuple(d["ebitda_margin"])
    return LBOInputs(**d), meta
