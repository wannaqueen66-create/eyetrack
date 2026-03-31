import os, re, glob, subprocess, shutil
import csv
from collections import defaultdict

# ============================================================
# 配置区
# ============================================================
SCRIPT = "/content/eyetrack/scripts/batch_heatmap_groups.py"
COMPOSE_SCRIPT = "/content/eyetrack/scripts/compose_scene_panel.py"

SCENES_ROOT = "/content/drive/MyDrive/映射"
MANIFEST_ORIG = "/content/drive/MyDrive/映射/group_manifest.csv"
from datetime import datetime
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DRIVE_SAVE_ROOT = f"/content/drive/MyDrive/映射/热力图输出_{_TS}"

TEMP_ROOT = "/content/temp_merged"
MANIFEST_FILTERED = "/content/manifest_filtered.csv"
SCREEN_W, SCREEN_H = 1748, 2064

# QC 排除名单（和 configs/excluded_participants_qc.csv 一致）
QC_EXCLUDED = {"孙校聪", "康少勇", "张钰鹏", "杨可", "洪婷婷", "陈韬", "高梓楠", "赵国宏"}

# 投稿图配置
FIG_TITLE = "Fig. X. Fixation heatmaps across six visual conditions"
FIG_NOTE = "Note: Heatmaps represent fixation-duration-weighted density across all participants (after QC). Warmer colors indicate higher visual attention concentration."

os.makedirs(DRIVE_SAVE_ROOT, exist_ok=True)

# ============================================================
# 生成过滤后的 manifest（去掉 csv_path 列 + 排除 QC 黑名单）
# ============================================================
print("==== 生成过滤后的 manifest ====\n")

with open(MANIFEST_ORIG, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    orig_fields = reader.fieldnames

    # 找 name 列
    name_col = None
    for c in ["name", "participant_id"]:
        if c in orig_fields:
            name_col = c
            break
    if name_col is None:
        raise SystemExit("manifest 缺少 name 或 participant_id 列")

    # 只保留 name + Experience（去掉 csv_path 和其他无关列）
    keep_cols = [name_col, "Experience"]
    kept = 0
    excluded = 0

    with open(MANIFEST_FILTERED, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=keep_cols)
        writer.writeheader()
        for row in reader:
            pid = row[name_col].strip()
            if pid in QC_EXCLUDED:
                excluded += 1
                continue
            writer.writerow({c: row[c] for c in keep_cols})
            kept += 1

print(f"原始 manifest: {kept + excluded} 人")
print(f"QC 排除: {excluded} 人")
print(f"过滤后 manifest: {kept} 人 → {MANIFEST_FILTERED}\n")

# ============================================================
# 工具函数
# ============================================================
def pick_bg(scene_dir: str):
    imgs = []
    for ext in ("png", "jpg", "jpeg"):
        imgs += glob.glob(os.path.join(scene_dir, f"*.{ext}"))
        imgs += glob.glob(os.path.join(scene_dir, f"*.{ext.upper()}"))
    if not imgs:
        return None
    return max(imgs, key=lambda p: os.path.getsize(p))

def extract_condition(folder_name: str):
    """从文件夹名中提取条件标签，如 C0W15、C1W45 等"""
    m = re.search(r"(C[01]W(?:WR)?(?:15|45|75))", folder_name, re.IGNORECASE)
    if m:
        return m.group(1).upper().replace("WWR", "W")
    return None

def is_excluded(csv_path: str) -> bool:
    """检查 CSV 文件名是否包含 QC 排除名单中的姓名"""
    fname = os.path.basename(csv_path)
    for name in QC_EXCLUDED:
        if name in fname:
            return True
    return False

# ============================================================
# 扫描并按条件合并
# ============================================================
EXCLUDE_DIRS = {"输出结果", "__MACOSX", ".ipynb_checkpoints"}
# 自动排除所有 热力图输出_ 开头的旧结果目录
EXCLUDE_PREFIXES = ("热力图输出",)

all_dirs = sorted([
    d for d in os.listdir(SCENES_ROOT)
    if os.path.isdir(os.path.join(SCENES_ROOT, d))
    and not d.startswith(".")
    and d not in EXCLUDE_DIRS
    and not any(d.startswith(p) for p in EXCLUDE_PREFIXES)
])

# 按条件分组（C0W15 -> [组1文件夹, 组2文件夹]）
condition_groups = defaultdict(list)
for d in all_dirs:
    cond = extract_condition(d)
    if cond:
        condition_groups[cond].append(d)
    else:
        print(f"[跳过-无法识别条件] {d}")

print(f"识别到 {len(condition_groups)} 个条件：")
for cond, dirs in sorted(condition_groups.items()):
    print(f"  {cond}: {dirs}")
print()

# ============================================================
# 逐条件：合并 CSV（排除 QC 黑名单）→ 跑聚合热力图
# ============================================================
print("==== 开始批量处理 ====\n")

for cond in sorted(condition_groups.keys()):
    dirs = condition_groups[cond]
    drive_outdir = os.path.join(DRIVE_SAVE_ROOT, cond)

    # 断点：已有则跳过
    done_marker = os.path.join(drive_outdir, "participants_summary.csv")
    if os.path.exists(done_marker):
        print(f"[跳过-已完成] {cond}")
        continue

    # 合并多个组的 CSV 到临时文件夹（排除 QC 黑名单）
    merged_dir = os.path.join(TEMP_ROOT, cond)
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    os.makedirs(merged_dir, exist_ok=True)

    bg_img = None
    csv_count = 0
    excluded_count = 0

    for d in dirs:
        scene_dir = os.path.join(SCENES_ROOT, d)

        if bg_img is None:
            bg_img = pick_bg(scene_dir)

        for csv_file in glob.glob(os.path.join(scene_dir, "*.csv")):
            if is_excluded(csv_file):
                excluded_count += 1
                continue
            fname = os.path.basename(csv_file)
            dst = os.path.join(merged_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(csv_file, dst)
                csv_count += 1

    if bg_img is None:
        print(f"[跳过-缺背景图] {cond}")
        continue
    if csv_count == 0:
        print(f"[跳过-缺CSV] {cond}")
        continue

    print(f"===== 正在处理: {cond} =====")
    print(f"  合并自: {dirs}")
    print(f"  底图: {os.path.basename(bg_img)}")
    print(f"  CSV数: {csv_count} (已排除QC: {excluded_count})")

    cmd = [
        "python", SCRIPT,
        "--manifest", MANIFEST_FILTERED,
        "--csv_dir", merged_dir,
        "--screen_w", str(SCREEN_W),
        "--screen_h", str(SCREEN_H),
        "--background_img", bg_img,
        "--outdir", drive_outdir,
        "--title_mode", "none",
        "--skip_individual",
    ]
    subprocess.run(cmd, check=True)
    print(f"  ✔️ 已保存: {cond}\n")

    shutil.rmtree(merged_dir, ignore_errors=True)

shutil.rmtree(TEMP_ROOT, ignore_errors=True)

print("==== 全部场景处理完成 ====")

# ============================================================
# 拼 6 场景投稿图
# ============================================================
print("\n==== 开始拼合投稿图 ====\n")

fig_out = os.path.join(DRIVE_SAVE_ROOT, "Fig_heatmap_6scenes.png")

compose_cmd = [
    "python", COMPOSE_SCRIPT,
    "--scene_dirs", DRIVE_SAVE_ROOT,
    "--outfile", fig_out,
    "--title", FIG_TITLE,
    "--note", FIG_NOTE,
]
subprocess.run(compose_cmd, check=True)
print(f"\n✔️ 投稿组合图已保存: {fig_out}")

# 清理临时 manifest
if os.path.exists(MANIFEST_FILTERED):
    os.remove(MANIFEST_FILTERED)

print("\n==== 全部完成 ====")
