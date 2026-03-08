from __future__ import annotations

import argparse
from pathlib import Path

from finetune_Nav2_XML.catalog.catalog_io import default_catalog_path, load_catalog
from finetune_Nav2_XML.constraints.xml_gbnf import build_nav2_bt_xml_gbnf
from finetune_Nav2_XML.constraints.xml_prefix_fn import _build_allowed_tags, _build_xml_regex


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate constraints artifacts for Nav2 XML-direct (catalog-driven).")
    p.add_argument(
        "--catalog",
        type=str,
        default=str(default_catalog_path()),
        help="Path to bt_nodes_catalog.json (default: finetune_Nav2_XML/catalog/bt_nodes_catalog.json).",
    )
    p.add_argument(
        "--reference-dir",
        type=str,
        default=None,
        help="Optional reference BT directory to enrich the allowlist for regex constraint.",
    )
    p.add_argument("--out-gbnf", type=str, default=None, help="Write XML GBNF grammar to this path.")
    p.add_argument("--out-regex", type=str, default=None, help="Write XML regex constraint to this path.")
    args = p.parse_args(argv)

    catalog = load_catalog(args.catalog)
    gbnf = build_nav2_bt_xml_gbnf(catalog)
    ref = Path(args.reference_dir).expanduser().resolve() if args.reference_dir else None
    allowed = _build_allowed_tags(catalog, ref)
    regex = _build_xml_regex(allowed_tags=allowed)

    if args.out_gbnf:
        out = Path(args.out_gbnf).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(gbnf, encoding="utf-8")
        print(f"Wrote GBNF: {out}")
    else:
        print(gbnf)

    if args.out_regex:
        out = Path(args.out_regex).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(regex + "\n", encoding="utf-8")
        print(f"Wrote regex: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

