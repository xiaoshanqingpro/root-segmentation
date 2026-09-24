"""阶段一 · 裁剪方案验证 v2。

v1 的问题: 根掩膜含"有管状响应"的项, 而边缘暗带恰好也有管状响应 -> 每张图都报"会切根", 假阳性。
v2 修正:
  根掩膜改为纯色彩判据 (比局部背景暗 且 带褐色饱和度)  -> 排除灰/黑的边缘暗带
  边框暗度改为相对中央区域的固定落差 -> 不再用分位数, 避免尺度漂移
另外对问题图输出"裁剪线可视化", 由人眼最终确认。

产出: outputs/阶段一/crop_check.csv , outputs/阶段一/crop_preview/*.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

IMAGE_DIR = P.RAW_D1
OUT_DIR = Path(r"D:\根系分割项目\outputs\阶段一")
PREVIEW_DIR = OUT_DIR / "crop_preview"
WORK_LONG_SIDE = 1100
EXCLUDE = {"树根004.jpg", "树叶003.jpg"}
MM_CROP = 2.5


def analyze(path: Path):
    with Image.open(path) as raw:
        full_w, full_h = raw.size
        dpi = float((raw.info.get("dpi") or (600, 600))[0])
    im = Image.open(path)
    im.draft("RGB", (WORK_LONG_SIDE, WORK_LONG_SIDE))
    im = im.convert("RGB")
    s = WORK_LONG_SIDE / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    disp_w = im.width
    to_full = full_w / disp_w
    rgb = np.asarray(im)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    illum = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(np.float32)
    depth = illum - gray

    root = (depth > 14) & (sat >= 0.12)
    root = cv2.morphologyEx(root.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)).astype(bool)

    h, w = gray.shape
    bg = float(np.median(gray[h // 4:3 * h // 4, w // 4:3 * w // 4]))
    dark = gray < (bg - 30)

    def ring_frac(mask, k):
        r = np.zeros_like(mask, bool)
        r[:k, :] = r[-k:, :] = r[:, :k] = r[:, -k:] = True
        return mask[r], r.sum()

    m = max(2, int(min(h, w) * 0.02))
    dk, n1 = ring_frac(dark, m)
    before = float(dk.mean() * 100) if n1 else 0.0

    crop_px_full = int(round(MM_CROP / 25.4 * dpi))
    crop_disp = max(1, int(round(crop_px_full / to_full)))
    inner_dark = dark[crop_disp:h - crop_disp, crop_disp:w - crop_disp]
    m2 = max(2, int(min(inner_dark.shape) * 0.02))
    r2 = np.zeros_like(inner_dark, bool)
    r2[:m2, :] = r2[-m2:, :] = r2[:, :m2] = r2[:, -m2:] = True
    after = float(inner_dark[r2].mean() * 100)

    ys, xs = np.nonzero(root)
    if len(ys):
        dist_disp = min(ys.min(), xs.min(), h - 1 - ys.max(), w - 1 - xs.max())
        dist_full = dist_disp * to_full
    else:
        dist_full = float("nan")
    return {
        "rgb": rgb, "root": root, "to_full": to_full, "crop_disp": crop_disp,
        "crop_px": crop_px_full, "before": before, "after": after,
        "dist_full": dist_full, "dpi": dpi,
    }


def preview(path: Path, res: dict) -> None:
    img = Image.fromarray(res["rgb"]).convert("RGB")
    dr = ImageDraw.Draw(img)
    c = res["crop_disp"]
    w, h = img.size
    dr.rectangle([c, c, w - c, h - c], outline=(0, 200, 0), width=4)
    dr.text((c + 8, c + 8), f"crop {MM_CROP}mm", fill=(0, 200, 0))
    ov = np.asarray(img).copy()
    ov[res["root"]] = [255, 0, 0]
    Image.fromarray(ov).save(PREVIEW_DIR / f"{path.stem}_cropline.png")


def main() -> int:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(IMAGE_DIR.rglob("*.jpg")) if p.name not in EXCLUDE]
    rows = []
    for i, p in enumerate(files, 1):
        res = analyze(p)
        cut = (not np.isnan(res["dist_full"])) and res["dist_full"] < res["crop_px"]
        rows.append({
            "file": p.name,
            "rel_path": str(p.relative_to(IMAGE_DIR.parent)),
            "dpi": res["dpi"],
            "crop_px": res["crop_px"],
            "root_min_border_px": None if np.isnan(res["dist_full"]) else int(round(res["dist_full"])),
            "root_would_be_cut": bool(cut),
            "border_dark_before": round(res["before"], 2),
            "border_dark_after": round(res["after"], 2),
        })
        if res["before"] > 8:
            preview(p, res)
        if i % 20 == 0:
            print(f"  ... {i}/{len(files)}", flush=True)

    with (OUT_DIR / "crop_check.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    ds = np.array([r["root_min_border_px"] for r in rows if r["root_min_border_px"] is not None])
    cut = [r for r in rows if r["root_would_be_cut"]]
    b_before = np.array([r["border_dark_before"] for r in rows])
    b_after = np.array([r["border_dark_after"] for r in rows])
    hard = sorted([r for r in rows if r["border_dark_before"] > 8],
                  key=lambda r: -r["border_dark_before"])

    print(f"\n=== 裁 {MM_CROP} mm（{len(rows)} 张）===")
    print(f"  根到画幅边缘的最近距离: 中位 {np.median(ds):.0f}px, 最小 {ds.min():.0f}px "
          f"(裁剪量 600dpi=59px / 720dpi=71px)")
    print(f"  边框暗像素: 裁剪前 中位 {np.median(b_before):.2f}% -> 裁剪后 中位 {np.median(b_after):.2f}%")
    print(f"\n  边缘暗带 >8% 的 {len(hard)} 张:")
    for r in hard:
        print(f"    {r['file']:<16} {r['border_dark_before']:6.2f}% -> {r['border_dark_after']:6.2f}%"
              f"   根距边缘 {r['root_min_border_px']}px")
    print(f"\n  裁剪会切到根的: {len(cut)} 张")
    for r in sorted(cut, key=lambda z: z["root_min_border_px"])[:30]:
        print(f"    {r['file']:<16} 根距边缘 {r['root_min_border_px']:>4}px < 裁剪量 {r['crop_px']}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
