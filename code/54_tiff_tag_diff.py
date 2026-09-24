"""对比"WinRHIZO 能打开"与"打不开"的 TIFF，逐项列出 TIFF 标签差异。"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
TAGS = {
    254: "NewSubfileType", 255: "SubfileType", 256: "ImageWidth", 257: "ImageLength",
    258: "BitsPerSample", 259: "Compression", 262: "PhotometricInterpretation",
    263: "Threshholding", 266: "FillOrder", 270: "ImageDescription", 271: "Make",
    272: "Model", 273: "StripOffsets", 274: "Orientation", 277: "SamplesPerPixel",
    278: "RowsPerStrip", 279: "StripByteCounts", 282: "XResolution", 283: "YResolution",
    284: "PlanarConfiguration", 296: "ResolutionUnit", 305: "Software", 306: "DateTime",
    317: "Predictor", 320: "ColorMap", 338: "ExtraSamples", 339: "SampleFormat",
    700: "XMP", 34665: "ExifIFD", 34675: "ICCProfile",
}


def dump(p: Path) -> dict:
    with Image.open(p) as im:
        t = im.tag_v2
        d = {}
        for k, v in t.items():
            name = TAGS.get(k, f"Tag{k}")
            if name == "StripOffsets" or name == "StripByteCounts":
                v = f"<{len(v) if hasattr(v,'__len__') else v} 项>"
            if name == "ICCProfile":
                v = f"<{len(v)} bytes>"
            if name == "XMP":
                v = f"<{len(v)} bytes>"
            d[f"{name}({k})"] = v
        d["_size"] = im.size
        d["_mode"] = im.mode
        d["_file_MB"] = round(p.stat().st_size / 1024**2, 2)
        return d


def main() -> int:
    files = sys.argv[1:]
    data = {}
    for s in files:
        p = Path(s)
        if not p.exists():
            print(f"缺失 {s}")
            continue
        data[s] = dump(p)
    keys = sorted({k for d in data.values() for k in d})
    w = max(len(k) for k in keys) + 2
    for s in data:
        print(f"\n=== {s} ===")
    print()
    print(f"{'标签':<{w}}" + "".join(f"{Path(s).name[:26]:<28}" for s in data))
    for k in keys:
        vals = [str(data[s].get(k, "—")) for s in data]
        mark = "  <<< 不同" if len(set(vals)) > 1 else ""
        print(f"{k:<{w}}" + "".join(f"{v[:26]:<28}" for v in vals) + mark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
