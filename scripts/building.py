import random

# 固定随机种子，保证可复现
random.seed(2025)

cofactor_type = {
    "Kindergarten": [
        "building6396",
        "building6398",
        "building6402",
        "building6405",
        "building6406",
        "building6407",
        "building6409",
        "building6415",
        "building6419",
        "building6421",
        "building6422",
        "building6425",
        "building6426",
        "building6428",
        "building6429",
        "building6433",
        "building6434",
        "building6437",
        "building6439",
        "building6443",
    ],
    "School": [
        "building6397",
        "building6400",
        "building6404",
        "building6408",
        "building6413",
        "building6414",
        "building6416",
        "building6418",
        "building6420",
        "building6424",
        "building6431",
        "building6432",
        "building6438",
        "building6440",
        "building6444",
        "building6445",
    ],
    "NursingHome": [
        "building6399",
        "building6410",
        "building6412",
        "building6417",
        "building6423",
        "building6436",
        "building6442",
    ],
    "Office": [
        "building6411",
        "building6441",
    ],
}

# >>> 你提供的学生公寓两栋
student_apartment = ["A", "B"]

selected = {
    "Kindergarten": random.choice(cofactor_type["Kindergarten"]),
    "School": random.choice(cofactor_type["School"]),
    "NursingHome": random.choice(cofactor_type["NursingHome"]),
    "Office": random.choice(cofactor_type["Office"]),
    "university": random.choice(student_apartment),
}

print("✅ 本次用于 LHS 微调实验的 5 栋楼：\n")
for k, v in selected.items():
    print(f"  {k:<16} →  {v}")

with open("selected_buildings.txt", "w") as f:
    for k, v in selected.items():
        f.write(f"{k},{v}\n")

print("\n已写出 → selected_buildings.txt")
