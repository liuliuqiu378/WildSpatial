"""ModelScope 数据集批量下载器（空间感知多模态数据台账）

============================ 目的 ============================
本项目此前只有 TUM fr1/desk（室内 RGB-D）+ 4Seasons（夜间驾驶）两套真实数据，
**对场景/目标/方法的理解偏窄**：缺 (a) 单目/双目深度真值、(b) 恶劣环境（雾/夜/水下/地下）、
(c) 大规模 3D 重建基准、(d) 检测/分割语义数据。
本脚本从 **ModelScope（国内、~2MB/s 可达）** 批量拉取，建立**多模态数据台账**。

============================ 用法 ============================
    conda activate wildspatial
    # 看清单（不下载）
    PYTHONPATH=src python scripts/fetch_datasets.py --list
    # 下某一批（tier: A/B/C/D）
    PYTHONPATH=src python scripts/fetch_datasets.py --tier A
    # 下全部（按 A→D 顺序）
    PYTHONPATH=src python scripts/fetch_datasets.py --tier all

下载到 data/raw/ms/<owner>__<name>/，并写 data/raw/ms/manifest.json 台账。

============================ 诚实声明 ============================
· 体积为 ModelScope 标注，实际以磁盘为准；大集（>30G）建议确认后再下。
· 仅用 `repo_type='dataset'`；`allow_patterns` 可选以减小体积（如只取样例）。
"""
import os
import sys
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 分层清单：A=小体积高价值优先；B=深度真值；C=多模态/恶劣；D=大基准
TIERS = {
    "A": [
        # owner, name, 用途, 备注
        ("OmniData", "NightCity", "夜间驾驶（恶劣光照）", "2.1G"),
        ("AI-ModelScope", "ScanNet-Absolute-Camera", "室内绝对相机位姿（前馈模型评测）", "0.42G"),
        ("AI-ModelScope", "TartanAir-Absolute-Camera", "合成多场景+绝对位姿（SLAM 评测）", "小(按场景)"),
        ("OmniData", "ModelNet40-C", "3D 形状分类（点云表示理解）", "2.28G"),
        ("Mriris", "remote-sensing-change-detection", "遥感变化检测（无人机视角）", "2.06G"),
    ],
    "B": [
        ("OmniData", "SUN_RGB-D", "室内 RGB-D 深度真值（单目深度对比）", "8.7G"),
        ("vikhyatk", "nyu_depth_v2", "室内深度真值（经典 NYUv2）", "小"),
    ],
    "C": [
        ("OmniData", "KITTI_depth_completion", "驾驶 LiDAR 深度补全（点云×深度）", "21G"),
        ("OmniData", "Foggy_Cityscapes", "雾天街景（恶劣天气语义）", "53G!"),
        ("OmniData", "NDISPark_Night_and_Day_Instance_Segmented_etc", "夜间/白天实例分割", "0.12G"),
        ("isLinXu", "rf100-vl-underwater-objects", "水下目标检测（恶劣环境）", "0.42G"),
    ],
    # 🆕 恶劣环境专批（用户 2026-09-12 决策：优先补"真正缺的恶劣环境"）
    "E": [
        ("OmniData", "NDISPark_Night_and_Day_Instance_Segmented_etc", "夜间/白天实例分割对照", "0.12G"),
        ("isLinXu", "rf100-vl-underwater-objects", "水下目标检测（散射/浑浊）", "0.42G"),
        ("LibreYOLO", "underwater-objects-5v7p8", "水下目标检测（补充）", "小"),
        ("Mriris", "remote-sensing-change-detection", "航拍/无人机视角", "2.06G"),
        ("OmniData", "KITTI_depth_completion", "驾驶 LiDAR 深度补全（视觉+激光融合真实数据）", "21G"),
    ],
    "D": [
        ("OpenDataLab", "ADE20K_2016", "场景解析语义分割", "6.05G"),
        ("PAI", "COCO2017", "检测/分割基准", "20.4G"),
        ("OmniData", "ScanNet", "室内 3D 重建大基准", "39G"),
    ],
    # 🆕 P1 四场景专批（2026-09-14）：按 docs/P1_field_projects.md 的四个真实场景补齐真实数据
    "F": [
        # —— 园区/室内服务机器人（动态行人 = P1 动态避障的真实输入）——
        ("OpenDataLab", "MOT17", "行人多目标跟踪（真实动态轨迹，替换 P1 模拟行人）", "5.8G"),
        ("OmniData", "CVC-14", "昼夜行人检测（园区夜巡场景）", "3.3G"),
        ("OmniData", "CrowdHuman", "拥挤行人检测（酒店/商场人流）", "10.2G"),
        ("DatatangBeijing", "190426ImagesofRobotCleanerPerspectiveCollectionData",
         "扫地机器人第一视角真实采集（园区扫地）", "13M"),
        ("DatatangBeijing", "76184Images-LiquidStainDataofRobotCleanerPerspective",
         "扫地机器人视角液体污渍（园区扫地任务真值）", "11M"),
        # —— 酒店送物机器人（真实具身操作数据）——
        ("RoboCOIN", "leju_robot_hotel_services_i", "酒店送物机器人真实采集（LeRobot 具身）", "2.1G"),
        ("RoboCOIN", "leju_robot_hotel_services_ah", "酒店送物机器人真实采集（小包）", "0.54G"),
        # —— 无人船（水面感知；⚠️ 优先选 login=False 公开可下）——
        ("isLinXu", "rf100-vl-floating-waste", "水面漂浮垃圾（无人船垃圾打捞，公开可下）", "604M"),
        ("irhawks", "floating-det", "漂浮物检测（公开可下）", "724M"),
        ("Flier123", "Ship", "船只检测（无人船避碰，公开可下）", "5.4G"),
        # ⚠️ 以下需登录（实测 401），仅备查，勿盲下：
        # ("Echo0174", "Trash_floater", "水面漂浮物（⚠️ 实测可下但有登录变体）", "3.2G"),
        # ("xiakeann", "Ship_Detection", "船只检测（⛔ 实测需登录，不可下）", "244M"),
    ],
}


def list_all():
    print("ModelScope 数据集下载清单（按 tier 分组）\n")
    for t in ("A", "B", "C", "D", "E", "F"):
        if t not in TIERS:
            continue
        print(f"=== Tier {t} ===")
        for owner, name, use, size in TIERS[t]:
            print(f"  {owner}/{name:48s} {size:10s} | {use}")
        print()


def fetch(owner, name, out_root, local_only=False):
    from modelscope import snapshot_download
    tag = f"{owner}__{name}"
    local_dir = os.path.join(out_root, tag)
    print(f"[↓] {owner}/{name} → {local_dir}")
    try:
        p = snapshot_download(
            f"{owner}/{name}",
            repo_type="dataset",
            local_dir=local_dir,
        )
        print(f"[✓] {tag} 完成")
        return {"owner": owner, "name": name, "local_dir": local_dir, "ok": True}
    except Exception as e:
        print(f"[✗] {tag} 失败: {str(e)[:160]}")
        return {"owner": owner, "name": name, "local_dir": local_dir, "ok": False,
                "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="A", help="A/B/C/D/all")
    ap.add_argument("--list", action="store_true", help="只列清单")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "raw", "ms"))
    args = ap.parse_args()

    if args.list:
        list_all(); return 0

    tiers = ["A", "B", "C", "D", "E", "F"] if args.tier == "all" else [args.tier]
    items = [it for t in tiers for it in TIERS[t]]
    os.makedirs(args.out, exist_ok=True)

    results = []
    for owner, name, use, size in items:
        r = fetch(owner, name, args.out)
        r["use"] = use; r["size_hint"] = size
        results.append(r)
        # 每下一个就写台账（便于他人接手）
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in results if r["ok"])
    print(f"\n[✓] 批次 {args.tier}：成功 {ok}/{len(results)}，台账 → {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
