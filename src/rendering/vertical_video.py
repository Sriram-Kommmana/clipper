"""FFmpeg filter-graph builders for 9:16 layouts. Pure string functions (unit-testable without FFmpeg)."""
from __future__ import annotations

from config import RenderConfig

BG_DOWNSCALE = 6   # background is blurred at 1/6 resolution (fast), then scaled back up


def _num(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def lerp_expression(points: list[tuple[float, float]]) -> str:
    """Piece-wise linear function of `t` through (t, value) points, as an FFmpeg expression."""
    if not points:
        raise ValueError("need at least one point")
    if len(points) == 1 or all(abs(p[1] - points[0][1]) < 0.5 for p in points):
        return _num(round(points[0][1], 1))
    expr = _num(round(points[-1][1], 1))
    for (t0, v0), (t1, v1) in reversed(list(zip(points, points[1:]))):
        if t1 <= t0:
            continue
        seg = f"{_num(round(v0, 1))}+({_num(round(v1 - v0, 1))})*(t-{_num(round(t0, 2))})/{_num(round(t1 - t0, 2))}"
        expr = f"if(lt(t,{_num(round(t1, 2))}),{seg},{expr})"
    return expr


def blur_layout(src_w: int, src_h: int, out_w: int, out_h: int, cfg: RenderConfig, *, w: int | None = None, h: int | None = None) -> str:
    """Blurred, darkened, enlarged copy as background; the untouched-aspect video centred on top."""
    if abs(src_w / src_h - out_w / out_h) < 0.01:
        return f"scale={out_w}:{out_h}:flags=lanczos,setsar=1"
    k = BG_DOWNSCALE
    sigma = max(0.5, cfg.blur_sigma / k)
    return (
        f"split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h},"
        f"scale=iw/{k}:ih/{k},gblur=sigma={sigma:.1f},scale={out_w}:{out_h}:flags=bilinear,eq=brightness=-{cfg.background_dim:.2f}[bg];"
        f"[fgsrc]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def crop_layout(src_w: int, src_h: int, out_w: int, out_h: int, x_points: list[tuple[float, float]] | None) -> str:
    """Fill the frame by cropping. x_points = [(t_seconds, left_edge_px)] for horizontal tracking."""
    target = out_w / out_h
    if src_w / src_h >= target:  # wider than target: crop sides, optionally following the subject
        cw = min(src_w, int(round(src_h * target / 2)) * 2)
        max_x = src_w - cw
        pts = [(t, max(0.0, min(float(max_x), x))) for t, x in (x_points or [(0.0, max_x / 2)])]
        x = lerp_expression(pts)
        crop = f"crop=w={cw}:h={src_h}:x='{x}':y=0"
    else:                        # taller than target: crop top/bottom (centre)
        ch = min(src_h, int(round(src_w / target / 2)) * 2)
        crop = f"crop=w={src_w}:h={ch}:x=0:y={(src_h - ch) // 2}"
    return f"{crop},scale={out_w}:{out_h}:flags=lanczos,setsar=1"


def original_layout(out_w: int, out_h: int) -> str:
    """Keep the source framing; just make dimensions even and fit inside (out_w x out_h)."""
    return f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,setsar=1"
