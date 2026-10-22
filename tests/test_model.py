from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from minilbo import (LBOInputs, irr, load_inputs, run_lbo, sensitivity_entry_exit, sensitivity_exit_year,
                     sources_and_uses)

DATA = Path(__file__).parent.parent / "data" / "sample_deal.json"


@pytest.fixture(scope="module")
def deal():
    return load_inputs(DATA)[0]


def flat_unlevered(**kw):
    """No debt, no fees, flat business: exit equity = entry EV + 5 years of accumulated free cash flow."""
    base = dict(
        ltm_revenue=100, ltm_ebitda_margin=0.25, entry_multiple=8.0, transaction_fees_pct=0, financing_fees_pct=0,
        min_cash=0, senior_multiple=0, senior_rate=0.07, senior_amortisation_pct=0.05, cash_sweep_pct=1.0,
        sub_multiple=0, sub_rate=0.11, revenue_growth=(0,) * 5, ebitda_margin=(0.25,) * 5,
        da_pct_revenue=0.05, capex_pct_revenue=0.05, nwc_pct_revenue=0.0, tax_rate=0.25,
        exit_multiple=8.0, exit_year=5)
    base.update(kw)
    return LBOInputs(**base)


def test_sources_equal_uses(deal):
    su = sources_and_uses(deal)
    assert su.loc[su.side == "Source", "amount"].sum() == pytest.approx(su.loc[su.side == "Use", "amount"].sum())


def test_closed_form_unlevered_flat_case():
    i = flat_unlevered()
    ebit = 100 * (0.25 - 0.05)
    fcf = 100 * 0.25 - ebit * 0.25 - 100 * 0.05            # EBITDA - tax - capex = 20 - ... = 25 - 5 - 5 = 15
    r = run_lbo(i)
    ev = 25 * 8.0
    assert r.sponsor_equity == pytest.approx(ev)
    assert r.exit_equity == pytest.approx(ev + 5 * fcf)
    assert r.moic == pytest.approx(1 + 5 * fcf / ev)
    assert r.irr == pytest.approx(r.moic ** (1 / 5) - 1)


def test_irr_function_matches_closed_form_and_multi_flows():
    assert irr([-100, 0, 0, 0, 0, 200]) == pytest.approx(2 ** (1 / 5) - 1)
    assert irr([-100, 10, 10, 10, 110]) == pytest.approx(0.10, abs=1e-6)   # par bond: IRR equals the coupon


def test_value_creation_bridge_sums_to_equity_gain(deal):
    for variant in (deal, replace(deal, exit_multiple=11.0), replace(deal, exit_multiple=7.0, exit_year=3)):
        r = run_lbo(variant)
        assert r.bridge.sum() == pytest.approx(r.exit_equity - r.sponsor_equity)


def test_no_multiple_expansion_when_multiples_equal(deal):
    assert run_lbo(deal).bridge["Multiple expansion"] == pytest.approx(0.0)


def test_debt_never_negative_and_sweep_accelerates_paydown(deal):
    r = run_lbo(deal).projections
    assert (r["senior_debt"] >= -1e-9).all()
    no_sweep = run_lbo(replace(deal, cash_sweep_pct=0.0)).projections
    assert (r["senior_debt"].iloc[1:] < no_sweep["senior_debt"].iloc[1:]).all()
    assert run_lbo(deal).irr > run_lbo(replace(deal, cash_sweep_pct=0.0)).irr - 0.02


def test_leverage_falls_over_time(deal):
    lev = run_lbo(deal).projections["net_debt_to_ebitda"]
    assert lev.is_monotonic_decreasing


def test_shortfall_flag_when_cash_flow_is_negative(deal):
    bad = replace(deal, ebitda_margin=(0.08,) * 5, revenue_growth=(0.0,) * 5)
    assert run_lbo(bad).projections["liquidity_shortfall"].any()


def test_sensitivities_move_in_the_right_direction(deal):
    t = sensitivity_entry_exit(deal, [8.0, 9.0, 10.0], [8.0, 9.0, 10.0])
    assert (t.diff(axis=1).dropna(axis=1) > 0).all().all()      # higher exit multiple -> higher IRR
    assert (t.diff(axis=0).dropna() < 0).all().all()            # higher entry multiple -> lower IRR
    m = sensitivity_entry_exit(deal, [8.0, 9.0], [9.0, 10.0], metric="moic")
    assert (m > 0).all().all()
    y = sensitivity_exit_year(deal, [3, 4, 5], [9.0])
    assert y.shape == (3, 1)


def test_too_much_debt_raises(deal):
    with pytest.raises(ValueError):
        run_lbo(replace(deal, senior_multiple=9.0, sub_multiple=1.0))


def test_exit_year_validation(deal):
    with pytest.raises(ValueError):
        replace(deal, exit_year=9)
