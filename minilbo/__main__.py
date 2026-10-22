"""Command line: python -m minilbo --input data/sample_deal.json --out docs/lbo_overview.png"""
import argparse

import pandas as pd

from .model import load_inputs, run_lbo, sensitivity_entry_exit
from .report import plot_overview


def main() -> None:
    p = argparse.ArgumentParser(description="Simplified LBO model")
    p.add_argument("--input", default="data/sample_deal.json")
    p.add_argument("--out", default="docs/lbo_overview.png")
    args = p.parse_args()

    i, meta = load_inputs(args.input)
    r = run_lbo(i)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.width", 200)
    print(f"{meta.get('name', 'LBO')} ({meta.get('currency', '')})\n")
    print("Sources and uses\n", r.sources_uses.to_string(index=False), "\n")
    print("Projections\n", r.projections.T.to_string(), "\n")
    print("Value creation\n", r.bridge.to_string(), "\n")
    print(f"Sponsor equity {r.sponsor_equity:,.1f} | Exit equity {r.exit_equity:,.1f} | "
          f"MOIC {r.moic:.2f}x | IRR {r.irr:.1%}")
    plot_overview(i, args.out)
    print(f"Chart written to {args.out}")


if __name__ == "__main__":
    main()
