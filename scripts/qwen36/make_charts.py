#!/usr/bin/env python3
# Renders the benchmark charts of the Qwen3.6 section in README.md as plain SVG (light and dark variant of each)
# into docs/qwen36/. No dependencies. The numbers are the measurements quoted in the README.
#
# usage: python3 scripts/qwen36/make_charts.py

import math
import os
from xml.sax.saxutils import escape

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "qwen36")

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#6f6e69", grid="#e1e0d9", base="#c3c2b7",
                  s1="#2a78d6", s2="#eb6834", gray="#9a9993"),
    "dark":  dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#9a9993", grid="#2c2c2a", base="#4a4a46",
                  s1="#3987e5", s2="#d95926", gray="#7a7973"),
}

FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
W = 760
PAD = 16


class Svg:
    def __init__(self, t, w, h, title, desc):
        self.t, self.w, self.h = t, w, h
        self.el = [f'<rect width="{w}" height="{h}" rx="8" fill="{t["surface"]}"/>']
        self.title, self.desc = title, desc

    def text(self, x, y, s, size=12, color="ink2", anchor="start", weight=400, baseline="middle"):
        self.el.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{self.t[color]}" '
                       f'text-anchor="{anchor}" dominant-baseline="{baseline}">{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, color, width=1):
        self.el.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{self.t[color]}" '
                       f'stroke-width="{width}"/>')

    def hbar(self, x0, x1, yc, thick, color, r=4):
        # square at the baseline (x0), 4 px rounded data end (x1)
        r = min(r, (x1 - x0) / 2, thick / 2)
        y0, y1 = yc - thick / 2, yc + thick / 2
        d = (f"M{x0:.1f},{y0:.1f} H{x1 - r:.1f} Q{x1:.1f},{y0:.1f} {x1:.1f},{y0 + r:.1f} V{y1 - r:.1f} "
             f"Q{x1:.1f},{y1:.1f} {x1 - r:.1f},{y1:.1f} H{x0:.1f} Z")
        self.el.append(f'<path d="{d}" fill="{self.t[color]}"/>')

    def polyline(self, pts, color):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.el.append(f'<polyline points="{p}" fill="none" stroke="{self.t[color]}" stroke-width="2" '
                       f'stroke-linejoin="round" stroke-linecap="round"/>')

    def dot(self, x, y, color):
        # 8 px marker with a 2 px surface ring
        self.el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{self.t[color]}" stroke="{self.t["surface"]}" '
                       f'stroke-width="2"/>')

    def swatch(self, x, y, color, kind):
        if kind == "bar":
            self.el.append(f'<rect x="{x:.1f}" y="{y - 6:.1f}" width="12" height="12" rx="3" fill="{self.t[color]}"/>')
        else:
            self.line(x - 2, y, x + 18, y, color, 2)
            self.dot(x + 8, y, color)

    def save(self, path):
        body = "\n".join(self.el)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
               f'font-family="{FONT}" role="img" aria-labelledby="t d">\n<title id="t">{escape(self.title)}</title>\n'
               f'<desc id="d">{escape(self.desc)}</desc>\n{body}\n</svg>\n')
        with open(path, "w") as f:
            f.write(svg)


def text_w(s, size):
    # rough width of proportional sans text, used only to space legend entries
    return len(s) * size * 0.56


def header(svg, title, subtitle, legend=None, kind="bar"):
    svg.text(PAD, 26, title, size=16, color="ink", weight=600)
    svg.text(PAD, 47, subtitle, size=12.5, color="ink2")
    if legend:
        x = PAD
        for label, color in legend:
            svg.swatch(x, 74, color, kind)
            off = 18 if kind == "bar" else 26
            svg.text(x + off, 74, label, size=12.5, color="ink2")
            x += off + text_w(label, 12.5) + 22


def hbar_chart(name, mode, title, subtitle, cats, series, xmax, xstep, note, fmt="{:.1f}", desc=""):
    t = THEMES[mode]
    n_s = len(series)
    thick, gap = 18, 2
    band = 22 + n_s * thick + (n_s - 1) * gap
    top = 96 if n_s > 1 else 68
    label_w = 236
    x0, x1 = PAD + label_w, W - 64
    plot_h = band * len(cats)
    h = top + plot_h + 74
    svg = Svg(t, W, h, title, desc or title)
    header(svg, title, subtitle, [(s[0], s[1]) for s in series] if n_s > 1 else None)

    def sx(v):
        return x0 + (x1 - x0) * v / xmax

    for v in range(0, xmax + 1, xstep):
        svg.line(sx(v), top, sx(v), top + plot_h, "grid")
        svg.text(sx(v), top + plot_h + 14, f"{v}", size=11, color="muted", anchor="middle")
    svg.text((x0 + x1) / 2, top + plot_h + 34, "tokens per second", size=11, color="muted", anchor="middle")
    for ci, cat in enumerate(cats):
        yc = top + band * ci + band / 2
        svg.text(x0 - 12, yc, cat, size=12.5, color="ink", anchor="end")
        for si, (_, color, values) in enumerate(series):
            y = yc + (si - (n_s - 1) / 2) * (thick + gap)
            v = values[ci]
            svg.hbar(x0, sx(v), y, thick, color)
            svg.text(sx(v) + 6, y, fmt.format(v), size=11.5, color="ink2")
    svg.line(x0, top, x0, top + plot_h, "base")
    svg.text(PAD, h - 14, note, size=11, color="muted")
    svg.save(os.path.join(OUT, f"{name}-{mode}.svg"))


def line_chart(name, mode, title, subtitle, xs, xticks, series, ymax, ystep, yfmt, ylabel, note, logx=False, vline=None,
               desc=""):
    t = THEMES[mode]
    top, h = 116, 436
    x0, x1 = PAD + 48, W - 150
    y0, y1 = h - 76, top
    svg = Svg(t, W, h, title, desc or title)
    header(svg, title, subtitle, [(s[0], s[1]) for s in series], kind="line")
    f = (lambda v: math.log2(v)) if logx else (lambda v: v)
    lo, hi = f(xs[0]), f(xs[-1])
    lo, hi = lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.04

    def sx(v):
        return x0 + (x1 - x0) * (f(v) - lo) / (hi - lo)

    def sy(v):
        return y0 - (y0 - y1) * v / ymax

    n = int(round(ymax / ystep))
    for i in range(n + 1):
        v = i * ystep
        svg.line(x0, sy(v), x1, sy(v), "base" if i == 0 else "grid")
        svg.text(x0 - 8, sy(v), yfmt.format(v), size=11, color="muted", anchor="end")
    svg.text(PAD, y1 - 14, ylabel, size=11, color="muted")
    for v, label in xticks:
        svg.text(sx(v), y0 + 16, label, size=11, color="muted", anchor="middle")
    if vline:
        vx, vlabel = vline
        svg.line(sx(vx), y1, sx(vx), y0, "base")
        svg.text(sx(vx) + 6, y1 + 8, vlabel, size=11, color="ink2")
    for _, color, ys, end_label in series:
        pts = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
        svg.polyline(pts, color)
        for x, y in pts:
            svg.dot(x, y, color)
        svg.text(pts[-1][0] + 12, pts[-1][1], end_label, size=12, color="ink2")
    svg.text(PAD, h - 36, note[0], size=11, color="muted")
    svg.text(PAD, h - 16, note[1], size=11, color="muted")
    svg.save(os.path.join(OUT, f"{name}-{mode}.svg"))


def main():
    os.makedirs(OUT, exist_ok=True)
    stages = ["Stock llama.cpp, best flags", "+ MTP draft head (2 tokens)", "+ n-gram drafts", "+ expert prefetch (this fork)"]
    for mode in THEMES:
        hbar_chart(
            "decode-new-code", mode,
            "Decode speed: writing new code and text",
            "Qwen3.6-35B-A3B Q8_0, 128K context, RTX 3060 12 GB + 32 GB DDR4-3200",
            stages,
            [("Greedy", "s1", [30.6, 39.1, 39.0, 42.1]),
             ("Default sampling", "s2", [30.6, 36.7, 37.4, 39.0])],
            50, 10,
            "Stock has no speculative decoding, so its speed is the same for every prompt and sampler.",
            desc="Greedy: 30.6, 39.1, 39.0, 42.1 tok/s. Default sampling: 30.6, 36.7, 37.4, 39.0 tok/s.")
        hbar_chart(
            "decode-rewrite", mode,
            "Decode speed: rewriting existing code",
            "The model re-emits a 1500-token file it was given, with one rename (a typical agentic edit)",
            stages,
            [("Rewrite", "s1", [30.6, 40.3, 106.9, 108.3])],
            120, 20,
            "n-gram drafts copy up to 30 tokens at a time from the context; 1443 of 1443 drafted tokens were accepted.",
            desc="30.6, 40.3, 106.9, 108.3 tok/s.")
        line_chart(
            "fa-kernel", mode,
            "Attention time per layer for one decoded token",
            "CUDA flash attention over a q8_0 K / q4_0 V cache, head size 256, RTX 3060",
            [16, 64, 128], [(16, "16K"), (64, "64K"), (128, "128K")],
            [("Stock (vector kernel)", "gray", [273, 1027, 2034], "Stock  2034 µs"),
             ("This fork (tensor cores)", "s1", [63, 202, 389], "This fork  389 µs")],
            2500, 500, "{:.0f}", "µs per attention layer",
            ("x-axis: tokens in the KV cache. 10 of the 40 layers are attention layers, so at 128K this saves",
             "about 16 ms per token. Measured decode at 68K depth (262K profile): 29.0 -> 33.4 tok/s."),
            desc="Stock: 273, 1027, 2034 us at 16K, 64K, 128K. This fork: 63, 202, 389 us.")
        line_chart(
            "prompt-offload", mode,
            "Short prompts: where to run the experts",
            "Time to process a new prompt chunk, 128K profile",
            [32, 64, 128, 192, 256, 512, 1024], [(v, str(v)) for v in (32, 64, 128, 256, 512, 1024)],
            [("Experts streamed to the GPU", "s1", [0.354, 0.489, 0.641, 0.722, 0.808, 1.010, 1.315], "GPU  1.32 s"),
             ("Experts on the CPU", "s2", [0.229, 0.344, 0.534, 0.704, 0.867, 1.694, 3.172], "CPU  3.17 s")],
            3.5, 0.5, "{:.1f}", "seconds",
            ("x-axis: new prompt tokens (log scale). Streaming the experts over PCIe has a fixed cost of ~0.3 s,",
             "so chunks below 192 tokens are faster on the CPU (an agentic turn is often only 15-600 tokens)."),
            logx=True, vline=(192, "GGML_OP_OFFLOAD_MIN_BATCH=192"),
            desc="Tokens 32, 64, 128, 192, 256, 512, 1024. GPU: 0.354, 0.489, 0.641, 0.722, 0.808, 1.010, 1.315 s. "
                 "CPU: 0.229, 0.344, 0.534, 0.704, 0.867, 1.694, 3.172 s.")
        hbar_chart(
            "prefill-ubatch", mode,
            "Prefill speed by micro-batch size",
            "llama-bench pp2048: the experts are streamed over PCIe once per micro-batch",
            ["-ub 512  (262K profile)", "-ub 1024  (128K profile)", "-ub 2048"],
            [("Prefill", "s1", [487, 761, 1135])],
            1200, 200,
            "Real agentic turns (3.7K to 11K new tokens): 351-377 tok/s at 262K context, 563-617 tok/s at 128K.",
            fmt="{:.0f}", desc="487, 761, 1135 tok/s.")


if __name__ == "__main__":
    main()
