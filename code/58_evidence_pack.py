"""生成误差归因的证据包，全部写入 测试\\调试\\。

包含:
  1. 逐样本误差表 CSV
  2. 误差统计（含 bootstrap 95% CI）
  3. 两类误差各一条示例：原图 / 人工修图 / 机器修图 + 数据对比
  4. 12 张样本总拼图
  5. 文字报告
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

Image.MAX_IMAGE_PIXELS = None
DBG = P.TEST_DBG
AN = P.OUTPUTS / "误差分析"
METRICS = [("Length(cm)", "根总长", "cm"), ("SurfArea(cm2)", "表面积", "cm²"),
           ("RootVolume(cm3)", "根体积", "cm³"), ("AvgDiam(mm)", "平均直径", "mm")]


def font(sz, b=False):
    for n in (("msyhbd.ttc", "msyh.ttc") if b else ("msyh.ttc",)):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def thumb(p: Path, w: int, h: int, block_min: bool = True) -> Image.Image:
    """缩略图: 默认可分块取最暗，保住细根（普通缩放会把细根平均掉）。"""
    im = Image.open(p).convert("RGB")
    a = np.asarray(im)
    if block_min:
        k = max(1, int(np.ceil(max(a.shape[:2]) / max(w, h))))
        Hc, Wc = (a.shape[0] // k) * k, (a.shape[1] // k) * k
        s = a[:Hc, :Wc].reshape(Hc // k, k, Wc // k, k, 3).min(axis=(1, 3)).astype(np.uint8)
        im = Image.fromarray(s)
    k2 = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * k2)), max(1, int(im.height * k2))), Image.LANCZOS)
    bg = Image.new("RGB", (w, h), (255, 255, 255))
    bg.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return bg


def find_man(stem: str) -> Path | None:
    for cand in [P.TEST_MAN / f"{stem}-man.tif", P.TEST_MAN / f"{stem}-man.jpg",
                 P.TEST_MAN_99GAI / f"{stem}-man.tif", P.TEST_MAN_99GAI / f"{stem}-man.jpg",
                 P.TEST_MAN_99GAI / f"{stem}-man-9.22.tif"]:
        if cand.exists():
            return cand
    hits = list(P.TEST_MAN.rglob(f"{stem}-man*"))
    return hits[0] if hits else None


def find_mac(stem: str) -> Path | None:
    for cand in [P.TEST_MAC / f"{stem}-mac.tif", P.TEST_MAC_99GAI / f"{stem}-mac.tif"]:
        if cand.exists():
            return cand
    hits = list(P.TEST_MAC.rglob(f"{stem}-mac*"))
    return hits[0] if hits else None


def find_orig(stem: str) -> Path | None:
    for root in (P.RAW_D2, P.RAW_D1):
        if not root.exists():
            continue
        for e in (".jpg", ".jpeg", ".tif", ".tiff", ".png"):
            hits = [p for p in root.rglob(stem + e) if "9.9改" not in p.parts]
            if hits:
                return hits[0]
    return None


def example_figure(stem: str, tag: str, note: str, pairs_row: dict, attr_row: dict) -> Path:
    """出一张示例图：原图 / 人工 / 机器 / 数据表。"""
    CW, CH = 700, 500
    o, m, k = find_orig(stem), find_man(stem), find_mac(stem)
    tiles = []
    for p, t in [(o, "① 原图（扫描件）"), (m, "② 人工修图"), (k, "③ 机器修图")]:
        if p and p.exists():
            bg = thumb(p, CW, CH, block_min=(t != "① 原图（扫描件）"))
        else:
            bg = Image.new("RGB", (CW, CH), (250, 235, 235))
            ImageDraw.Draw(bg).text((20, CH // 2), f"缺: {t}", fill=(200, 0, 0), font=font(20))
        tiles.append((t, bg))

    fig = Image.new("RGB", (CW * 3 + 40, (CH + 40) + 60 + 200 + 60), (255, 255, 255))
    dr = ImageDraw.Draw(fig)
    dr.text((16, 12), f"{tag}　样本 {stem}", fill=(0, 0, 0), font=font(30, True))
    dr.text((16, 50), note, fill=(150, 0, 0), font=font(20))
    y = 84
    for i, (t, bg) in enumerate(tiles):
        fig.paste(bg, (16 + i * (CW + 6), y + 30))
        dr.text((16 + i * (CW + 6), y + 6), t, fill=(0, 0, 0), font=font(22, True))
        dr.rectangle([16 + i * (CW + 6), y + 30, 16 + i * (CW + 6) + CW, y + 30 + CH],
                     outline=(210, 210, 210))
    y += 30 + CH + 26
    dr.text((16, y), "四项重点指标：人工 vs 机器", fill=(0, 0, 0), font=font(24, True))
    y += 40
    colw = (CW * 3) // 4
    for j, (key, cn, unit) in enumerate(METRICS):
        x = 16 + j * colw
        a = pairs_row.get(f"人工_{cn}", "")
        b = pairs_row.get(f"机器_{cn}", "")
        e = pairs_row.get(f"误差%_{cn}", "")
        dr.text((x, y), f"{cn}（{unit}）", fill=(0, 0, 0), font=font(21, True))
        dr.text((x, y + 30), f"人工  {a}", fill=(70, 70, 70), font=font(20))
        dr.text((x, y + 58), f"机器  {b}", fill=(0, 90, 160), font=font(20))
        col = (0, 140, 0) if abs(float(e or 0)) < 10 else (
            (210, 120, 0) if abs(float(e or 0)) < 20 else (205, 0, 0))
        dr.text((x, y + 86), f"误差  {e}%", fill=col, font=font(22, True))
    y += 130
    gy = attr_row.get("根灰度", "?")
    sh = attr_row.get("阴影占比%", "?")
    wd = attr_row.get("根宽度px", "?")
    dr.text((16, y), f"该样本形态特征：根灰度 {gy}（越小越黑）　阴影占比 {sh}%　根宽 {wd}px"
                     f"　四项绝对误差中位 {attr_row.get('四项绝对误差中位','?')}%",
            fill=(90, 90, 90), font=font(19))
    out = DBG / f"示例_{tag.split('：')[0]}_{stem}.png"
    fig.save(out, optimize=True)
    return out


def main() -> int:
    DBG.mkdir(parents=True, exist_ok=True)
    pairs = list(csv.DictReader((AN / "pairs.csv").open(encoding="utf-8-sig")))
    attr = {r["样本"]: r for r in csv.DictReader((AN / "误差归因.csv").open(encoding="utf-8-sig"))}
    summary = json.loads((AN / "summary.json").read_text(encoding="utf-8"))
    grp = json.loads((AN / "归因分组.json").read_text(encoding="utf-8"))

    # ---- 1. 逐样本表 ----
    shutil.copy2(AN / "pairs.csv", DBG / "01_逐样本误差表.csv")
    shutil.copy2(AN / "误差归因.csv", DBG / "02_形态特征与误差.csv")
    shutil.copy2(AN / "summary.json", DBG / "03_误差统计含置信区间.json")
    print(f"已拷贝 3 份数据表 -> {DBG}")

    # ---- 2. 挑示例 ----
    # 阴影太深: 阴影占比最高、且误差偏正（机器偏大）的那张
    def pick_shadow():
        cand = [r for r in pairs if r["样本"] in grp["阴影太深组"]]
        best = None
        for r in cand:
            try:
                e = float(r["误差%_表面积"])
            except Exception:  # noqa: BLE001
                continue
            if best is None or e > best[1]:
                best = (r, e)
        return best[0] if best else cand[0]

    def pick_pale():
        cand = [r for r in pairs if r["样本"] in grp["根太淡组"]]
        best = None
        for r in cand:
            try:
                e = float(r["误差%_根总长"])
            except Exception:  # noqa: BLE001
                continue
            if best is None or e < best[1]:
                best = (r, e)
        return best[0] if best else cand[0]

    ps, pp = pick_shadow(), pick_pale()
    print(f"阴影太深示例: {ps['样本']}   根太淡示例: {pp['样本']}")

    f1 = example_figure(
        ps["样本"], "误差类型一：阴影太深",
        f"该样本阴影占比 {attr[ps['样本']].get('阴影占比%')}%（12 张中最高档之一）。"
        f"机器把阴影误判成根 → 表面积误差 {ps.get('误差%_表面积')}%、根体积 {ps.get('误差%_根体积')}%（偏大）",
        ps, attr[ps["样本"]])
    f2 = example_figure(
        pp["样本"], "误差类型二：根太淡",
        f"该样本根灰度 {attr[pp['样本']].get('根灰度')}（越浅越难判）。"
        f"机器漏掉淡细根 → 根总长误差 {pp.get('误差%_根总长')}%、表面积 {pp.get('误差%_表面积')}%（偏小）",
        pp, attr[pp["样本"]])
    print(f"  -> {f1}\n  -> {f2}")

    # ---- 3. 12 张总拼图 ----
    CW, CH = 430, 310
    cols = 3
    rws = (len(pairs) + cols - 1) // cols
    sheet = Image.new("RGB", (CW * cols, (CH + 62) * rws), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, r in enumerate(pairs):
        s = r["样本"]
        c, rr = i % cols, i // cols
        x, y = c * CW, rr * (CH + 62)
        k = find_mac(s)
        if k:
            sheet.paste(thumb(k, CW - 8, CH), (x + 4, y + 40))
        dr.text((x + 4, y + 2), f"{i+1}. {s}", fill=(0, 0, 0), font=font(20, True))
        e1, e2 = r.get("误差%_根总长", ""), r.get("误差%_表面积", "")
        dr.text((x + 4, y + CH + 42),
                f"根总长 {e1}%　表面积 {e2}%", fill=(150, 0, 0), font=font(17))
        dr.rectangle([x + 4, y + 40, x + CW - 4, y + 40 + CH], outline=(220, 220, 220))
    o = DBG / "04_十二样本拼图.png"
    sheet.save(o, optimize=True)
    print(f"  -> {o}")

    # ---- 4. 报告 ----
    m = summary["metrics"]
    lines = []
    A = lines.append
    A("# 机器修图 vs 人工修图：端到端误差分析")
    A("")
    A(f"数据来源：`测试\\测试数据.TXT`（RHIZO 2016a），**可配对样本 {summary['n_pairs']} 对**。")
    A("同一张原图、同一台扫描仪、同一套阈值。差异 = (机器 − 人工) / 人工。")
    A("")
    A("## 一、总体误差（含 95% 置信区间）")
    A("")
    A("| 指标 | 中位误差 | 95% CI（bootstrap 10000 次） | 绝对误差中位 | 单张范围 | 是否显著偏离 0 |")
    A("|---|---|---|---|---|---|")
    for k, cn, _ in METRICS + [("ProjArea(cm2)", "投影面积", ""), ("Tips", "根尖数", ""),
                               ("Forks", "分叉数", ""), ("Crossings", "交叉数", "")]:
        d = m.get(cn)
        if not d:
            continue
        ci = d["ci95"]
        A(f"| {cn} | {d['median']:+.2f}% | [{ci[0]:+.2f}%, {ci[1]:+.2f}%] | {d['median_abs']:.2f}% | "
          f"[{d['min']:+.1f}%, {d['max']:+.1f}%] | {'**是**' if d['significant_from_0'] else '否'} |")
    A("")
    A("**读法：**")
    A("- 四项重点指标的 95% CI **都跨过 0**，说明机器修图**没有系统性偏差**——")
    A("  在样本层面，机器与人工的测量值在统计上不可区分。")
    A("- 但**单张离散度很大**：绝对误差中位 5.9%~12.7%，最坏单张超过 40%。")
    A("  → **均值可信、单张不可信**。")
    A("- **分叉数与交叉数有显著系统性偏差**（−50.8% / −86.7%，CI 不跨 0）——")
    A("  这两个拓扑指标**当前不可用**。")
    A("")
    A("## 二、误差归因：两类相反的系统偏差")
    A("")
    A(f"按形态特征把 12 张分成两组（各取最极端 {grp['n_grp']} 张）：")
    A("")
    A(f"- **阴影太深组**：{', '.join(grp['阴影太深组'])}")
    A(f"- **根太淡组**：{', '.join(grp['根太淡组'])}")
    A("")
    A("| 指标 | 阴影太深组 中位 [95%CI] | 根太淡组 中位 [95%CI] |")
    A("|---|---|---|")
    for k, cn, _ in METRICS:
        a = grp["阴影组统计"].get(cn)
        b = grp["淡根组统计"].get(cn)
        if not a or not b:
            continue
        A(f"| {cn} | {a['median']:+.1f}% [{a['ci95'][0]:+.1f}, {a['ci95'][1]:+.1f}] | "
          f"{b['median']:+.1f}% [{b['ci95'][0]:+.1f}, {b['ci95'][1]:+.1f}] |")
    A("")
    A("**结论：两类误差方向相反**")
    A("")
    A("- **阴影太深 → 机器偏大**：阴影被判成根，表面积 +13.7%、根体积 +20.0%、直径 +8.2%。")
    A("- **根太淡 → 机器偏小**：淡细根被漏掉，根总长 −12.4%（**该 CI 不跨 0，显著偏低**）、表面积 −9.0%。")
    A("")
    A("这解释了为什么一套全局参数修不好全部：两类的修正方向是**相反**的。")
    A("")
    A("## 三、示例证据")
    A("")
    A(f"- **阴影太深**：`示例_误差类型一：阴影太深_{ps['样本']}.png`")
    A(f"  - 阴影占比 {attr[ps['样本']].get('阴影占比%')}%　根灰度 {attr[ps['样本']].get('根灰度')}")
    A(f"- **根太淡**：`示例_误差类型二：根太淡_{pp['样本']}.png`")
    A(f"  - 根灰度 {attr[pp['样本']].get('根灰度')}　阴影占比 {attr[pp['样本']].get('阴影占比%')}%")
    A("")
    A("每张示例含三格图（原图 / 人工修图 / 机器修图）+ 四项指标数据。")
    A("")
    A("## 四、需要注意的局限")
    A("")
    A("1. **n=12，置信区间较宽**。上表 CI 是用 bootstrap（10000 次重采样）算的中位数区间；")
    A("   分组后每组只有 3~4 张，分组结论只能当**方向性判断**，不能当定量结论。")
    A("2. **66-003 同时落进两组**（既阴影重、根又淡），说明两类并非互斥。")
    A("3. **分析区不一致**（人工 vs 机器的 WinRHIZO 自动分析区不同，如 28-CL-003 差 2.6 倍）")
    A("   这一混杂因素**尚未排除**，可能影响部分样本的数值。")
    A("4. 分叉/交叉数的显著偏差提示：机器掩膜把相邻根并连了，拓扑结构被破坏。")
    A("")
    A("## 五、文件清单")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `01_逐样本误差表.csv` | 12 对样本的全部指标：人工值 / 机器值 / 误差% |")
    A("| `02_形态特征与误差.csv` | 每张原图的形态特征（根灰度、阴影占比、根宽…）与误差 |")
    A("| `03_误差统计含置信区间.json` | 总体统计 + bootstrap CI + 分组统计 |")
    A("| `04_十二样本拼图.png` | 12 张机器修图总览 + 各自误差 |")
    A(f"| `示例_误差类型一：阴影太深_{ps['样本']}.png` | 阴影太深类证据 |")
    A(f"| `示例_误差类型二：根太淡_{pp['样本']}.png` | 根太淡类证据 |")
    (DBG / "误差分析报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {DBG/'误差分析报告.md'}")

    print(f"\n=== 调试文件夹内容 ===")
    for p in sorted(DBG.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
