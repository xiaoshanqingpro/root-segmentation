"""阶段二 · 10 张工作集选样（分层抽样，可复现）。

选样依据（全部来自阶段一实测数据）:
  - 覆盖 4 个批次，并保证 0814 批次（原 720 dpi，已被重采样）至少有 2 张
  - 按"阴影嫌疑占比"分层: 高 4 张 / 中 3 张 / 低 3 张，避免只挑好图或只挑坏图
  - 强制纳入阶段一发现的疑难图（水渍、褶皱最重、根偏淡、尺子裁过）

产出:
  config/phase2_images.json        工作集清单（含每张的阴影/根占比、来源批次、是否重采样过）
  outputs/阶段二/phase2_contact.png 缩略图总表（供你确认选样）
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

PROJ = P.PROJ
ROOT = Path.home() / "Desktop" / "扫描"
STAGE1 = PROJ / "outputs" / "阶段一"
OUT_DIR = PROJ / "outputs" / "阶段二"
CFG_DIR = PROJ / "config"
SEED = 42
N = 10

# 阶段一明确发现的疑难图，强制纳入（rel_path 为原图相对路径）
MUST_INCLUDE = [
    "图片\\0808\\F120-001.jpg",   # 左下角环形水渍，最典型
    "图片\\0807\\G27-003.jpg",    # 纸面褶皱最重
    "图片\\0814\\G218-002.jpg",   # 根整体偏淡 + 背景大面积不匀 + 幅面异常
    "图片\\0807\\G66-001.jpg",    # 左边缘粉色刻度尺最宽(11.68mm)，裁边最狠
]
QUOTA = {"高": 4, "中": 3, "低": 3}


def font(sz: int):
    for n in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def main() -> int:
    man = {r["rel_path"]: r for r in
           csv.DictReader((STAGE1 / "preprocess_manifest.csv").open(encoding="utf-8-sig"))}
    sur = {r["rel_path"]: r for r in
           csv.DictReader((STAGE1 / "survey.csv").open(encoding="utf-8-sig"))}

    pool = [rel for rel, r in man.items() if r["excluded"] == "False"]
    shadow = {rel: float(sur[rel]["proxy_shadow_pct"]) for rel in pool if rel in sur}
    ranked = sorted(shadow, key=lambda r: -shadow[r])

    strata = {
        "高": ranked[: max(1, len(ranked) // 3)],
        "中": ranked[len(ranked) // 3: 2 * len(ranked) // 3],
        "低": ranked[2 * len(ranked) // 3:],
    }

    rng = np.random.default_rng(SEED)
    picked: list[str] = [r for r in MUST_INCLUDE if r in man and man[r]["excluded"] == "False"]
    for name, quota in QUOTA.items():
        cand = [r for r in strata[name] if r not in picked]
        need = quota - sum(1 for p in picked if p in strata[name])
        if need <= 0 or not cand:
            continue
        idx = rng.choice(len(cand), size=min(need, len(cand)), replace=False)
        picked += [cand[i] for i in idx]

    # 覆盖约束：0814 批次至少 2 张；且至少有 1 张是从 720 dpi 重采样来的（校验重采样路径）
    def ensure(pred, k, pool_sorted):
        have = sum(1 for p in picked if pred(p))
        if have >= k:
            return
        extra = [r for r in pool_sorted if pred(r) and r not in picked]
        for e in extra[: k - have]:
            # 从"低"层里替换掉一张，保持总数与分层结构
            low = [p for p in picked if p in strata["低"] and p not in MUST_INCLUDE]
            if low:
                picked.remove(low[-1])
            picked.append(e)

    ensure(lambda p: "\\0814\\" in p, 2, ranked)
    ensure(lambda p: man[p]["resampled"] == "True", 1, ranked)

    picked = list(dict.fromkeys(picked))[:N]

    items = []
    for rel in picked:
        m, s = man[rel], sur.get(rel, {})
        items.append({
            "rel_path": rel,
            "batch": rel.split("\\")[1] if rel.count("\\") >= 2 else "_根目录",
            "processed_path": m["out_path"],
            "out_w": int(m["out_w"]), "out_h": int(m["out_h"]), "out_dpi": int(m["out_dpi"]),
            "resampled": m["resampled"] == "True",
            "crop_total_mm": round(sum(float(m[f"crop_{s2}_mm"]) for s2 in
                                       ["left", "right", "top", "bottom"]), 2),
            "proxy_shadow_pct": float(s.get("proxy_shadow_pct", 0)),
            "proxy_root_pct": float(s.get("proxy_root_pct", 0)),
        })

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (CFG_DIR / "phase2_images.json").write_text(
        json.dumps({"seed": SEED, "n": len(items), "quota": QUOTA,
                    "must_include": MUST_INCLUDE, "items": items},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # 缩略图总表
    COLS, TW = 5, 430
    rows_n = (len(items) + COLS - 1) // COLS
    thumbs = []
    for it in items:
        p = PROJ / it["processed_path"]
        im = Image.open(p)
        im.draft("RGB", (TW, TW))
        im = im.convert("RGB")
        im = im.resize((TW, int(im.height * TW / im.width)), Image.LANCZOS)
        thumbs.append((it, im))
    th = max(t[1].height for t in thumbs)
    sheet = Image.new("RGB", (COLS * TW, rows_n * (th + 52)), (245, 245, 245))
    dr = ImageDraw.Draw(sheet)
    f, f2 = font(19), font(16)
    for i, (it, im) in enumerate(thumbs):
        r, c = divmod(i, COLS)
        x, y = c * TW, r * (th + 52)
        sheet.paste(im, (x, y))
        dr.rectangle([x, y + th, x + TW, y + th + 52], fill=(255, 255, 255))
        dr.text((x + 6, y + th + 3), f'{i+1}. ' + it["rel_path"].split("\\")[-1], fill=(0, 0, 0), font=f)
        dr.text((x + 6, y + th + 27),
                f'阴影{it["proxy_shadow_pct"]:.1f}% 根{it["proxy_root_pct"]:.1f}% '
                f'{"[重采样]" if it["resampled"] else ""} 裁{it["crop_total_mm"]}mm',
                fill=(120, 0, 0), font=f2)
    sheet.save(OUT_DIR / "phase2_contact.png", optimize=True)

    print(f"[phase2] 选出 {len(items)} 张（seed={SEED}）")
    for i, it in enumerate(items, 1):
        print(f'  {i:2d}. {it["rel_path"]:<28} 阴影{it["proxy_shadow_pct"]:6.2f}%  '
              f'根{it["proxy_root_pct"]:5.2f}%  {"重采样" if it["resampled"] else "      "}  '
              f'裁{it["crop_total_mm"]:5.1f}mm')
    bat = {}
    for it in items:
        bat[it["batch"]] = bat.get(it["batch"], 0) + 1
    print(f"  批次覆盖: {bat}")
    print(f"  清单: config/phase2_images.json   缩略图: outputs/阶段二/phase2_contact.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
