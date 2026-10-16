"""Figure 8: the generative model of Section 6, in the visual language of Figures 1 and 2.

Two panels: how a frame becomes six numbers and comes back as an image, and how the input drives
the two-state NODE to generate a sequence. The helpers, the palette and the NODE block are taken
from `make_framework_figure_v2.py`; the thumbnails come from `fig8_assets_v3.py`.

The SVG is written here and rasterised with Inkscape; the Playwright route of Figures 1 and 2 is
not available in this environment.
"""
import base64
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "figures_sec6" / "fig8_v3"
DESTS = [HERE.parent.parent / "Manuscript" / "figures", HERE / "figures_sec6"]
STEM = "Figure_8_generative_model"

W, H = 1800, 1180
OR, ORF = "#D98A1C", "#FFF5E6"   # pool (orange)
BL, BLF = "#2F6FC4", "#EEF4FD"   # thermal (blue)
GR, GRF = "#23926A", "#E8F6EF"   # learned (green)
GY, GYF = "#8A94A6", "#F3F4F6"   # frozen (grey)
TX, TX2 = "#1F2937", "#4B5563"
SANS = "DejaVu Sans, Liberation Sans, Arial, sans-serif"
SERIF = "DejaVu Serif, STIX Two Text, Times New Roman, serif"
o = []


def add(s):
    o.append(s)


def rect(x, y, w, h, stroke, fill, rx=14, sw=2.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=24, fill=TX, weight="normal", anchor="start", family=SANS):
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>')


def arrow(x1, y1, x2, y2, color=TX2, sw=2.5, dash=None, mid="ah"):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} '
        f'marker-end="url(#{mid})"/>')


def it(s):
    return f'<tspan font-family="{SERIF}" font-style="italic">{s}</tspan>'


def sb(s, size=24):
    return f'<tspan dy="7" font-size="{int(size * 0.68)}">{s}</tspan><tspan dy="-7"></tspan>'


def hat_over(x, y, w=14, h=7, sw=2.2, c=None):
    """A circumflex drawn as a path, because the fonts here have no combining accent."""
    add(f'<path d="M{x - w / 2} {y} L{x} {y - h} L{x + w / 2} {y}" fill="none" '
        f'stroke="{c or TX}" stroke-width="{sw}" stroke-linecap="round" '
        f'stroke-linejoin="round"/>')


def dot_over(x, y, r=2.8):
    """The dot of a time derivative, drawn as a circle. The combining accent over delta has no
    glyph in the fonts available here."""
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{BL}"/>')


def dot(x, y, c=TX2):
    add(f'<circle cx="{x}" cy="{y}" r="5" fill="{c}"/>')


def junction(x, y, sign, r=22):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#FFFFFF" stroke="{GY}" stroke-width="2.5"/>')
    text(x, y + 10, sign, 30, TX, "normal", "middle")


def thumb(x, y, side, name, label=None, lab_size=20):
    data = base64.b64encode((ASSETS / f"{name}.png").read_bytes()).decode()
    add(f'<image x="{x}" y="{y}" width="{side}" height="{side}" '
        f'xlink:href="data:image/png;base64,{data}" preserveAspectRatio="none"/>')
    add(f'<rect x="{x}" y="{y}" width="{side}" height="{side}" fill="none" stroke="{GY}" '
        f'stroke-width="2"/>')
    if label:
        text(x + side / 2, y + side + lab_size + 6, label, lab_size, TX2, "normal", "middle")


add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
    f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>')
for mid, col in [("ah", TX2), ("ahO", OR), ("ahB", BL), ("ahG", GR)]:
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{col}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# ---------------------------------------------------------------- (a) representation
rect(20, 20, 1760, 520, "#CBD5E1", "#FAFBFC", rx=18, sw=2)
text(44, 64, "(a) Representation", 30, TX, "bold")

thumb(44, 96, 132, "frame")
text(192, 252, "recorded frame", 19, TX2)
arrow(186, 162, 236, 162)
rect(236, 128, 190, 70, GY, GYF, rx=12, sw=2)
text(331, 160, "DINOv2", 23, TX, "normal", "middle")
text(331, 186, "frozen", 19, GY, "normal", "middle")
arrow(426, 162, 476, 162)
rect(476, 132, 130, 62, GR, GRF, rx=12, sw=2)
text(541, 171, "head " + it("h"), 24, TX, "normal", "middle")
arrow(606, 162, 656, 162)
rect(656, 132, 170, 62, OR, ORF, rx=12, sw=2.5)
text(741, 171, it("z") + ", 6 values", 22, TX, "normal", "middle")
arrow(826, 162, 876, 162)
rect(876, 124, 250, 78, GR, GRF, rx=12, sw=2)
text(1001, 154, "structure decoder", 21, GR, "bold", "middle")
text(1001, 186, it("D") + "(" + it("z") + ") = " + it("Ī") + " + Δ(" + it("z") + ")",
     23, TX, "normal", "middle")
arrow(1126, 162, 1176, 162)
thumb(1176, 96, 132, "structure")

# the residual and its path
junction(132, 300, "&#8722;")
add(f'<polyline points="132,228 132,278" fill="none" stroke="{TX2}" stroke-width="2.5"/>')
# the subtraction is fed from the side of the structure thumbnail, clear of every label
add(f'<polyline points="1308,162 1348,162 1348,300 154,300" fill="none" stroke="{TX2}" '
    f'stroke-width="2.5" marker-end="url(#ah)"/>')
arrow(132, 322, 132, 364)
thumb(66, 364, 132, "residual")
text(166, 342, it("r") + " = " + it("I") + " &#8722; " + it("D") + "(" + it("z") + ")",
     21, TX2)
arrow(198, 430, 248, 430)
rect(248, 400, 150, 62, GR, GRF, rx=12, sw=2)
text(323, 439, "encoder", 22, TX, "normal", "middle")
arrow(398, 430, 448, 430)
thumb(448, 364, 132, "codes")
text(514, 352, "codes, 24 &#215; 24 of 512", 20, TX2, "normal", "middle")
arrow(580, 430, 630, 430)
rect(630, 400, 170, 62, GR, GRF, rx=12, sw=2)
text(715, 439, "decoder " + it("G"), 22, TX, "normal", "middle")
arrow(800, 430, 884, 430)
junction(906, 430, "+")
# D(z) joins the sum from above, on its own vertical
add(f'<polyline points="1348,300 1348,352 906,352 906,406" fill="none" stroke="{TX2}" '
    f'stroke-width="2.5" marker-end="url(#ah)"/>')
arrow(928, 430, 1000, 430)
thumb(1000, 364, 132, "reconstruction",
      it("D") + "(" + it("z") + ") + " + it("G") + "(" + it("c") + ")", 19)

# ---------------------------------------------------------------- (b) generation
rect(20, 560, 1760, 600, "#CBD5E1", "#FAFBFC", rx=18, sw=2)
text(44, 604, "(b) Generation from the input and the first frame", 30, TX, "bold")

thumb(44, 742, 118, "frame", "first frame", 19)
arrow(162, 801, 206, 801)
rect(206, 772, 130, 58, GR, GRF, rx=12, sw=2)
text(271, 809, "encoder", 21, TX, "normal", "middle")
arrow(336, 801, 380, 801)
rect(380, 768, 150, 66, OR, ORF, rx=12, sw=2.5)
text(455, 798, it("ẑ") + sb("0", 22), 23, TX, "normal", "middle", SERIF)
text(455, 824, it("δ̂") + sb("0", 20) + " = 0", 20, BL, "normal", "middle", SERIF)
rect(206, 886, 130, 58, GY, GYF, rx=12, sw=2)
text(271, 923, "speed " + it("u"), 21, TX, "normal", "middle")
add(f'<polyline points="336,915 530,915" fill="none" stroke="{TX2}" stroke-width="2.5"/>')
add(f'<polyline points="530,801 586,801" fill="none" stroke="{TX2}" stroke-width="2.5" '
    f'marker-end="url(#ah)"/>')
add(f'<polyline points="530,915 560,915 560,862 586,862" fill="none" stroke="{TX2}" '
    f'stroke-width="2.5" marker-end="url(#ah)"/>')

# the NODE block, drawn as in Figure 1(b)
rect(600, 636, 700, 400, "#64748B", "#FFFFFF", rx=16, sw=2.5)
rect(626, 662, 440, 180, "#CBD5E1", "#F8FAFC", rx=12, sw=2)
text(646, 694, "relaxation form", 21, TX2)
text(646, 752, '<tspan font-style="italic">ż</tspan> = &#8722;<tspan font-style="italic">Γ</tspan>'
     ' [ <tspan font-style="italic">z</tspan> &#8722; <tspan font-style="italic">z</tspan>'
     '<tspan dy="8" font-size="19">eq</tspan><tspan dy="-8"> ]</tspan>',
     29, OR, "normal", "start", SERIF)
text(646, 812, '<tspan font-style="italic">δ</tspan> = <tspan font-style="italic">λ</tspan>'
     ' [ <tspan font-style="italic">φ</tspan> &#8722; <tspan font-style="italic">δ</tspan> ]',
     29, BL, "normal", "start", SERIF)
dot_over(654, 783)
rect(626, 870, 440, 136, GR, GRF, rx=12, sw=2)
text(646, 906, "equilibria and rates", 21, GR, "bold")
text(646, 944, it("z") + sb("eq", 23) + "(" + it("u") + ", " + it("δ") + "),  " + it("Γ")
     + "(" + it("z") + ", " + it("u") + ", " + it("δ") + ")", 23, TX, "normal", "start", SERIF)
text(646, 982, it("φ") + "(" + it("z") + ", " + it("u") + "),  " + it("λ"),
     23, TX, "normal", "start", SERIF)
arrow(846, 870, 846, 846, GR, 2.5, None, "ahG")
rect(1116, 716, 140, 150, "#475569", "#F1F5F9", rx=10, sw=2.5)
text(1186, 802, "∫ " + '<tspan font-style="italic">dt</tspan>', 38, TX, "normal", "middle", SERIF)
arrow(1066, 756, 1114, 756)

# the two predicted states, as in Figure 1(b)
arrow(1256, 746, 1318, 746, BL, 3, None, "ahB")
arrow(1256, 846, 1318, 846, OR, 3, None, "ahO")
rect(1320, 716, 140, 60, BL, BLF, rx=12, sw=2.5)
text(1390, 758, it("δ") + sb("k", 24), 25, BL, "bold", "middle", SERIF)
hat_over(1384, 735, c=BL)
rect(1320, 816, 140, 60, OR, ORF, rx=12, sw=2.5)
text(1390, 858, it("ẑ") + sb("k", 24), 25, OR, "bold", "middle", SERIF)

# the current state fed back is both of them
add(f'<polyline points="1460,746 1700,746" fill="none" stroke="{TX2}" stroke-width="2.5"/>')
add(f'<polyline points="1460,846 1700,846" fill="none" stroke="{TX2}" stroke-width="2.5"/>')
add(f'<polyline points="1700,846 1700,616 846,616 846,634" fill="none" stroke="{TX2}" '
    f'stroke-width="2.5" marker-end="url(#ah)"/>')
dot(1700, 746)
text(1270, 608, "current state", 20, TX2, "normal", "middle")

# the generated sequence hangs straight below the pool state, clear of every other line
rect(1320, 912, 300, 56, GR, GRF, rx=12, sw=2)
text(1470, 950, it("D") + "(" + it("ẑ") + ") + retrieved detail", 21, TX, "normal", "middle")
arrow(1390, 876, 1390, 910)
times = (ASSETS / "times.txt").read_text().split()
for i, t in enumerate(times):
    thumb(1314 + i * 108, 988, 96, f"gen{i}", f"{it('t')} = {t} s", 18)
arrow(1470, 968, 1470, 986)

add('</svg>')
svg_text = "\n".join(o)
for d in DESTS:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{STEM}.svg").write_text(svg_text)

src = DESTS[1] / f"{STEM}.svg"
# Inkscape does not pick up the italic serif face, so the SVG is rendered with Chrome, the same
# engine the Playwright route of Figures 1 and 2 uses.
scale = 180.0 / 25.4 * 600 / W
png = DESTS[1] / f"{STEM}.png"
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                "--default-background-color=FFFFFF",
                f"--force-device-scale-factor={scale:.4f}",
                f"--window-size={W},{H}", f"--screenshot={png}", src.resolve().as_uri()],
               check=True, capture_output=True)
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={DESTS[1] / f'{STEM}.pdf'}", src.resolve().as_uri()],
               check=True, capture_output=True)
for ext in ("png", "pdf"):
    shutil.copy(DESTS[1] / f"{STEM}.{ext}", DESTS[0] / f"{STEM}.{ext}")
print(f"written {STEM}.{{svg,png,pdf}}")
