#!/usr/bin/env python3
"""生成交付给评测端的 conditions.yaml（只列正式录像）。录制目录名决定条件名，对照表在 holdout1_paths.py。

    python3 make_conditions.py <recordings_holdout1 目录>

目录名是 YYYYmmdd_HHMMSS_<路径>[_<tag>]：S 形用 --tag sceneA/B/C 区分场景，其余正式录像不带 tag。
带 board、rehearsal 等其他 tag 的目录不应该放进批次目录，遇到就报错。每个条件应有 3 条，不是 3 条时报错但仍不写文件。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import holdout1_paths as hp

NAME = re.compile(r"^\d{8}_\d{6}_(.+)$")
PER_CONDITION = 3


def condition_of(directory: str) -> str | None:
    """录制目录名对应的条件名；不是正式录像时为 None。"""
    m = NAME.match(directory)
    if not m:
        return None
    rest = m.group(1)
    if rest in hp.CONDITION_OF:
        return hp.CONDITION_OF[rest]
    if rest.startswith("s_path_"):
        return hp.S_SCENE_TAGS.get(rest[len("s_path_"):])
    return None


def build(root: Path) -> tuple[dict[str, str], list[str]]:
    """(完整录制名 → 条件名, 问题列表)。"""
    conditions, problems = {}, []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "calib"):
        name = condition_of(d.name)
        if name is None:
            problems.append(f"{d.name}：不是正式录像的目录名，不应该放在批次目录里")
        else:
            conditions[d.name] = name
    for name, count in sorted(Counter(conditions.values()).items()):
        if count != PER_CONDITION:
            problems.append(f"条件 {name} 有 {count} 条，应该是 {PER_CONDITION} 条")
    return conditions, problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="批次目录，例如 data/recordings/recordings_holdout1")
    args = parser.parse_args(argv)
    conditions, problems = build(args.root)
    for m in problems:
        print(f"问题：{m}")
    if problems or not conditions:
        print("没有写文件")
        return 1
    text = "conditions:\n" + "".join(f"  {k}: {v}\n" for k, v in conditions.items())
    (args.root / "conditions.yaml").write_text(text, encoding="utf-8", newline="\n")
    print(f"已写 {args.root / 'conditions.yaml'}：{len(conditions)} 条，{len(set(conditions.values()))} 个条件")
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
