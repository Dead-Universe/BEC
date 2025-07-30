#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_buildingsbench_university2oov.py
--------------------------------------
按 7 : 1 : 2 生成训练 / 验证 / 测试三套建筑列表，并分别输出
    oov_val.txt   ——  验证集  “dataset:building_id”
    oov_test.txt  ——  测试集  “dataset:building_id”（含 university）
"""
import csv, random, re
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ---------- 配置 ----------
ROOT = Path("/home/hadoop/bec/buildings-bench/v2.0.0/BuildingsBench").resolve()
OUT_DIR = Path(".")
SEED = 2025
TRAIN_R, VAL_R, TEST_R = 0.7, 0.1, 0.2  # === 修改点：7 : 1 : 2
EXCLUDE = "buildingsbench_with_outliers"
UNIV_DIR = "university"  # university -> 强制测试集

# 若需要保留固定 LCL 栈，可在此填入；默认留空
FIXED_LCL: Set[str] = {
    "MAC000109",
    "MAC000141",
    "MAC000408",
    "MAC000418",
    "MAC000635",
    "MAC000666",
    "MAC000672",
    "MAC000675",
    "MAC000688",
    "MAC000700",
    "MAC000702",
    "MAC000714",
    "MAC000722",
    "MAC000759",
    "MAC000769",
    "MAC000777",
    "MAC000778",
    "MAC000790",
    "MAC001655",
    "MAC001688",
    "MAC001707",
    "MAC001715",
    "MAC001758",
    "MAC001770",
    "MAC001772",
    "MAC001777",
    "MAC002124",
    "MAC002133",
    "MAC002136",
    "MAC002146",
    "MAC002281",
    "MAC002290",
    "MAC002292",
    "MAC002334",
    "MAC002858",
    "MAC002873",
    "MAC002878",
    "MAC003178",
    "MAC003196",
    "MAC003202",
    "MAC003298",
    "MAC003504",
    "MAC003509",
    "MAC003516",
    "MAC003521",
    "MAC003530",
    "MAC003532",
    "MAC003542",
    "MAC003548",
    "MAC004645",
    "MAC004658",
    "MAC004904",
    "MAC004906",
    "MAC004913",
    "MAC004915",
    "MAC004920",
    "MAC004927",
    "MAC004975",
    "MAC004982",
    "MAC004991",
    "MAC004996",
    "MAC005002",
    "MAC005087",
    "MAC005103",
    "MAC005107",
    "MAC005113",
    "MAC005126",
    "MAC005439",
    "MAC005440",
    "MAC005468",
    "MAC005480",
    "MAC002162",
    "MAC002842",
    "MAC003973",
    "MAC005462",
}
FIXED_LCL: Set[str] = set()

# ---------- 正则 ----------
VALID_CSV = re.compile(r"_clean=\d{4}\.csv$", re.IGNORECASE)
ID_FROM_FN = re.compile(r"([^/\\]+?)_clean=\d{4}\.csv$", re.IGNORECASE)

# (dataset, building_id) → [csv_path, ...]
b2f: Dict[Tuple[str, str], List[str]] = {}


def add(ds: str, bid: str, path: str):
    b2f.setdefault((ds, bid), []).append(path)


# ═══════════════════════════════════════════════
# 1. 遍历并收集 (dataset, building) 映射
# ═══════════════════════════════════════════════
for csv_path in ROOT.rglob("*.csv"):
    if EXCLUDE in csv_path.parts:
        continue
    if "weather" in csv_path.name.lower():
        continue
    if not VALID_CSV.search(csv_path.name):
        continue

    dataset = csv_path.relative_to(ROOT).parts[0]

    # 检查列数
    try:
        with open(csv_path, newline="") as fp:
            header = next(csv.reader(fp))
    except Exception:
        continue
    if len(header) <= 1:
        continue

    if len(header) <= 2:  # 文件级
        m = ID_FROM_FN.match(csv_path.name)
        if m:
            add(dataset, m.group(1), str(csv_path))
    else:  # 多列
        for col in header[1:]:
            if col.strip():
                add(dataset, col.strip(), str(csv_path))

total_blds = len(b2f)
print(f"✓ 收集到 {total_blds} 栋建筑记录")

# ═══════════════════════════════════════════════
# 2. 划分集合
# ═══════════════════════════════════════════════
all_blds: Set[Tuple[str, str]] = set(b2f)

# 2‑A university → 测试集
univ_blds = {k for k in all_blds if k[0].lower() == UNIV_DIR.lower()}
remaining = all_blds - univ_blds

# 2‑B 固定 LCL（可选）
fixed_train = {(ds, bid) for ds, bid in remaining if ds == "LCL" and bid in FIXED_LCL}
remaining -= fixed_train

# 2‑C 计算目标数量（向下取整后再补齐余数给训练集）
train_target = int(total_blds * TRAIN_R) - len(fixed_train)
val_target = int(total_blds * VAL_R)
# 保证总和 <= total_blds
if train_target < 0:
    raise ValueError("FIXED_LCL 数量超过训练集目标，请调整比例或 FIXED_LCL")

# 2‑D 随机抽样
random.seed(SEED)
val_blds = set(random.sample(list(remaining), val_target))
remaining -= val_blds
extra_train_needed = train_target - len(fixed_train)
extra_train = set(random.sample(list(remaining), extra_train_needed))
train_blds = fixed_train | extra_train
test_blds = (all_blds - train_blds - val_blds) | univ_blds  # 测试 = 剩余 + university

# 统计
print(
    f"→ 训练集 {len(train_blds)} 栋"
    f" | 验证集 {len(val_blds)} 栋"
    f" | 测试集 {len(test_blds)} 栋 (university {len(univ_blds)})"
)

# ═══════════════════════════════════════════════
# 3. 输出 oov_val.txt / oov_test.txt
# ═══════════════════════════════════════════════
OUT_DIR.mkdir(exist_ok=True)


def dump(oov_set: Set[Tuple[str, str]], fname: str):
    lines = [f"{d}:{b}" for d, b in sorted(oov_set)]
    with open(OUT_DIR / fname, "w") as fp:
        fp.write("\n".join(lines))
    print(f"✅ 已生成 {fname}  ({len(oov_set)} 行)")


dump(val_blds, "oov_val.txt")
dump(test_blds, "oov_test.txt")
