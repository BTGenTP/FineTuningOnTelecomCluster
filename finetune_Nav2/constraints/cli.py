from __future__ import annotations

import argparse
from pathlib import Path

from finetune_Nav2.catalog.catalog_io import default_catalog_path, load_catalog
from finetune_Nav2.constraints.steps_gbnf import build_steps_json_gbnf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate constraints artifacts from the Nav2 skills catalog.")
    p.add_argument(
        "--catalog",
        type=str,
        default=str(default_catalog_path()),
        help="Path to bt_nodes_catalog.json (default: BT_Navigator/script/bt_nodes_catalog.json).",
    )
    p.add_argument("--out-gbnf", type=str, default=None, help="Write steps JSON GBNF grammar to this path.")
    args = p.parse_args(argv)

    catalog = load_catalog(args.catalog)
    gbnf = build_steps_json_gbnf(catalog)

    if args.out_gbnf:
        out = Path(args.out_gbnf).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(gbnf, encoding="utf-8")
        print(f"Wrote GBNF: {out}")
    else:
        print(gbnf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

