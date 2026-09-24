"""解析 测试数据.TXT 里的四条 RHIZO 记录，并拼四栏对比图。

栏位: ① 原图(无法扫描)  ② 人工修图  ③ 机器修图-jpg(收边1px)  ④ 机器修图-auto2(收边2px)
每栏: 上=图像  中=WinRHIZO 扫描截图  下=数据对比表
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import sys

# --- 统一路径（见 paths.py；数据搬家只改那里）---
sys.path.insert(0, str(Path(__file__).parent))
import paths as P  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SRC = P.TEST
OUT = Path(r"D:\根系分割项目\outputs\7-003三方对比")

# ---- 解析 RHIZO 文本 ----
WANT = ["Length(cm)", "ProjArea(cm2)", "SurfArea(cm2)", "AvgDiam(mm)",
        "RootVolume(cm3)", "Tips", "Forks", "Crossings"]
LABEL = {"Length(cm)": "根总长 Length(cm)", "ProjArea(cm2)": "投影面积 ProjArea(cm²)",
         "SurfArea(cm2)": "表面积 SurfArea(cm²)", "AvgDiam(mm)": "平均直径 AvgDiam(mm)",
         "RootVolume(cm3)": "根体积 Volume(cm³)", "Tips": "根尖数 Tips",
         "Forks": "分叉数 Forks", "Crossings": "交叉数 Crossings"}


def parse_txt(p: Path):
    txt = p.read_text(encoding="gbk", errors="replace")
    lines = [l.rstrip("\n") for l in txt.splitlines()]
    hdr = next(l for l in lines if "SoilVol(m3)" in l)
    names = [h.strip() for h in hdr.split("\t")]
    recs = {}
    for l in lines:
        if not l.startswith("7003"):
            continue
        f = l.split("\t")
        sid = f[0].strip()
        row = {}
        for k in WANT:
            try:
                row[k] = f[names.index(k)].strip()
            except ValueError:
                row[k] = ""
        recs[sid] = row
    return recs


def find(name: str) -> Path:
    """在 测试\\ 下（含 原图/人工/机器 子目录）按文件名查找。"""
    p = SRC / name
    if p.exists():
        return p
    hits = list(SRC.rglob(name))
    return hits[0] if hits else p


def font(sz, bold=False):
    for n in (("msyhbd.ttc", "msyh.ttc", "simhei.ttf") if bold else ("msyh.ttc", "simhei.ttf")):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def load_fit(p: Path, w: int, h: int) -> Image.Image:
    im = Image.open(p).convert("RGB")
    k = min(w / im.width, h / im.height)
    im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
    bg = Image.new("RGB", (w, h), (255, 255, 255))
    bg.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return bg


def main() -> int:
    recs = parse_txt(SRC / "测试数据.TXT")
    man = recs.get("7003man", {})
    a1 = recs.get("7003auto", {})
    a2 = recs.get("7003auto2-tiff", {})
    print("7003man        ", man)
    print("7003auto       ", a1)
    print("7003auto2-tiff ", a2)

    def pct(new, base):
        try:
            return f"{(float(new)/float(base)-1)*100:+.1f}%"
        except Exception:  # noqa: BLE001
            return ""

    OUT.mkdir(parents=True, exist_ok=True)
    CW, IH, SH, HDR = 760, 560, 470, 46
    cols = [
        ("① 原图", find("7-003.tif"), None, "原图无法扫描\n（背景不干净，\nWinRHIZO 无法分析）"),
        ("② 人工修图  (7-003-man.tif)", find("7-003-man.tif"), find("7003-man-scan.png"), None),
        ("③ 机器修图-jpg  (7-003_auto.jpg)", find("7-003_auto.jpg"), find("7003-auto-scan.png"), None),
        ("④ 机器修图-auto2  (7-003_auto2.tif)", find("7-003_auto2.tif"), find("7003-auto2-scan.png"), None),
    ]
    imgs, scans = [], []
    for _, ip, sp, ph in cols:
        imgs.append(load_fit(ip, CW, IH))
        if sp and sp.exists():
            scans.append(load_fit(sp, CW, SH))
        else:
            box = Image.new("RGB", (CW, SH), (245, 245, 245))
            d = ImageDraw.Draw(box)
            for i, ln in enumerate((ph or "（无）").split("\n")):
                f = font(28 if i == 0 else 22)
                tw = d.textlength(ln, font=f)
                d.text(((CW - tw) / 2, SH // 2 - 40 + i * 38), ln,
                       fill=(150, 0, 0) if i == 0 else (90, 90, 90), font=f)
            scans.append(box)

    row1 = Image.new("RGB", (CW * 4, HDR + IH), (255, 255, 255))
    row2 = Image.new("RGB", (CW * 4, HDR + SH), (255, 255, 255))
    d1, d2 = ImageDraw.Draw(row1), ImageDraw.Draw(row2)
    for i, ((t, _, _, _), im, sc) in enumerate(zip(cols, imgs, scans)):
        row1.paste(im, (i * CW, HDR))
        d1.text((i * CW + 10, 8), t, fill=(0, 0, 0), font=font(23, True))
        row2.paste(sc, (i * CW, HDR))
        d2.text((i * CW + 10, 8), "WinRHIZO 扫描结果" if i else "扫描结果",
                fill=(0, 0, 0), font=font(23, True))
        d1.line([(i * CW, 0), (i * CW, HDR + IH)], fill=(200, 200, 200), width=2)
        d2.line([(i * CW, 0), (i * CW, HDR + SH)], fill=(200, 200, 200), width=2)

    # 数据表
    rows = len(WANT)
    TH = 56 + (rows + 1) * 44 + 30
    tab = Image.new("RGB", (CW * 4, TH), (255, 255, 255))
    dt = ImageDraw.Draw(tab)
    dt.text((12, 6), "WinRHIZO 扫描数据对比（RHIZO 2016a，同一张原图 / 同一台扫描仪 / 同一套阈值）",
            fill=(0, 0, 0), font=font(26, True))
    colx = [16, CW + 16, CW * 2 + 16, CW * 3 + 16, CW * 3 + 330]
    heads = ["指标", "② 人工修图", "③ 机器修图-jpg", "④ 机器修图-auto2", "④ vs ② 差异"]
    y = 50
    for x, h in zip(colx, heads):
        dt.text((x, y), h, fill=(0, 0, 0), font=font(22, True))
    dt.line([(0, y + 32), (CW * 4, y + 32)], fill=(120, 120, 120), width=2)
    for i, k in enumerate(WANT):
        yy = y + 42 + i * 44
        if i % 2 == 1:
            dt.rectangle([0, yy - 6, CW * 4, yy + 36], fill=(246, 246, 250))
        dt.text((colx[0], yy), LABEL[k], fill=(0, 0, 0), font=font(22))
        dt.text((colx[1], yy), man.get(k, ""), fill=(0, 0, 0), font=font(22))
        dt.text((colx[2], yy), a1.get(k, ""), fill=(0, 0, 0), font=font(22))
        dt.text((colx[3], yy), a2.get(k, ""), fill=(0, 0, 0), font=font(22))
        d = pct(a2.get(k, "nan"), man.get(k, "nan"))
        dt.text((colx[4], yy), d, fill=(200, 0, 0) if d.startswith("+") else (0, 90, 200),
                font=font(22, True))

    note = ("说明：原图背景不干净，WinRHIZO 无法直接分析。③ 与 ④ 都是机器修图，唯一区别是收边量"
            "（③ 收边1px、④ 收边2px），④ 的掩膜面积更贴人工修图。")
    canvas = Image.new("RGB", (CW * 4, HDR + IH + 14 + HDR + SH + TH + 56), (255, 255, 255))
    yy = 0
    canvas.paste(row1, (0, yy)); yy += HDR + IH + 14
    canvas.paste(row2, (0, yy)); yy += HDR + SH + 14
    canvas.paste(tab, (0, yy)); yy += TH
    ImageDraw.Draw(canvas).text((14, yy + 10), note, fill=(90, 90, 90), font=font(21))
    o = OUT / "7-003_四栏扫描对比总图.png"
    canvas.save(o, optimize=True)
    print(f"-> {o}  {canvas.size}")
    return 0


if __name__ == "__main__":
    main()
