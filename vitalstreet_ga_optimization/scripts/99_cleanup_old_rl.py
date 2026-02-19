#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""可选：输出“可删除清单”，不自动删除旧 RL 相关文件。"""
from pathlib import Path

# 仓库根目录（本工程上一级）
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 与旧 RL 相关的路径（相对仓库根）
CLEANUP_CANDIDATES = [
    "playground/rl/",
    "playground/env/",
    "playground/objective/",
    "playground/logging/",
    "playground/main.py",
    "playground/configs/",
    "playground/tests/test_transition.py",
    "playground/tests/test_reward_termination.py",
    "playground/tests/test_action_mask.py",
]

def main():
    print("可删除清单（迁移至 vitalstreet_ga_optimization 后，以下为旧 RL 相关）：")
    print()
    for rel in CLEANUP_CANDIDATES:
        p = REPO_ROOT / rel
        if p.exists():
            print(f"  [存在] {rel}")
        else:
            print(f"  [不存在] {rel}")
    print()
    print("默认不自动删除。确认后请手动删除上述路径。")


if __name__ == "__main__":
    main()
