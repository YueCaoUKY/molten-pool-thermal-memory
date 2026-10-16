"""Figure 8: the generative model behind Section 6, in the visual language of Figures 1 and 2.

Three bands: how a frame becomes a six-number pool state and comes back as an image, how the
input drives the two-state NODE, and how the two are joined to generate a sequence from the
input alone. Thumbnails come from fig8_assets.py.

Writes Figure_8_generative_model.{svg,png,pdf} into Manuscript/figures and figures_sec6.
The SVG is written here and rasterised with Inkscape, which is installed on this machine; the
Playwright route used for Figures 1 and 2 is not available in this environment.
"""
import base64
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "figures_sec6" / "fig8"
DESTS = [HERE.parent.parent / "Manuscript" / "figures", HERE / "figures_sec6"]
STEM = "Figure_8_generative_model"

W, H = 1800, 1180
OR, ORF = "#D98A1C", "#FFF5E6"          # the pool equation, as in Figure 2
BL, BLF = "#2F6FC4", "#EEF4FD"          # the thermal equation
GR, GRF = "#23926A", "#E8F6EF"          # the representation
GY, GYF = "#64748B", "#F1F5F9"          # frozen parts
TX, TX2 = "#1F2937", "#4B5563"
SANS = "DejaVu Sans, Liberation Sans, Arial, sans-serif"
SERIF = "DejaVu Serif, STIX Two Text, Times New Roman, serif"

o = []


def add(s):
    o.append(s)


def rect(x, y, w, h, stroke, fill, rx=10, sw=2.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=22, fill=TX, weight="normal", anchor="start", family=SANS):
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>')


def line(pts, color=TX2, sw=2.4, arrow=True, mid="ah", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = f' marker-end="url(#{mid})"' if arrow else ''
    add(f'<polyline points="{" ".join(f"{x},{y}" for x, y in pts)}" fill="none" '
        f'stroke="{color}" stroke-width="{sw}"{d}{m}/>')


def dot(x, y, c=TX2):
    add(f'<circle cx="{x}" cy="{y}" r="5" fill="{c}"/>')


def hat(x, y, w=13, h=7, sw=2.0):
    """A circumflex drawn as a path. The combining accent over delta has no glyph in the
    fonts available here, so the two strokes are placed by hand."""
    add(f'<path d="M{x - w / 2} {y} L{x} {y - h} L{x + w / 2} {y}" fill="none" stroke="{TX}" '
        f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')


def I(s):
    return f'<tspan font-family="{SERIF}" font-style="italic">{s}</tspan>'


def sub(s, sz=15, dy=7):
    return f'<tspan dy="{dy}" font-size="{sz}">{s}</tspan><tspan dy="-{dy}">&#8203;</tspan>'


def thumb(x, y, side, name, label=None, stroke=GY):
    data = base64.b64encode((ASSETS / f"{name}.png").read_bytes()).decode()
    add(f'<image x="{x}" y="{y}" width="{side}" height="{side}" '
        f'xlink:href="data:image/png;base64,{data}" '
        f'preserveAspectRatio="none" image-rendering="auto"/>')
    add(f'<rect x="{x}" y="{y}" width="{side}" height="{side}" fill="none" stroke="{stroke}" '
        f'stroke-width="2"/>')
    if label:
        text(x + side / 2, y + side + 26, label, 19, TX2, "normal", "middle")


add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
    f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>')
for mid, col in [("ah", TX2), ("ahO", OR), ("ahB", BL), ("ahG", GR)]:
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{col}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# ----------------------------------------------------------------- (a) representation
add(f'<rect x="44" y="70" width="{W - 88}" height="330" rx="16" fill="{GRF}"/>')
text(64, 54, '(a)', 26, TX, "bold")
text(132, 54, 'representation: a frame to six numbers and back', 23, GR, "bold")

thumb(70, 104, 150, "frame", "recorded frame")
line([(232, 179), (292, 179)])

rect(292, 134, 180, 90, GY, GYF)
text(382, 170, 'DINOv2 ViT-B/14', 19, TX, "normal", "middle")
text(382, 198, 'frozen', 18, GY, "normal", "middle")
line([(472, 179), (528, 179)])

rect(528, 144, 120, 70, GR, GRF)
text(588, 187, 'head ' + I('h'), 22, TX, "normal", "middle")
line([(648, 179), (704, 179)])

rect(704, 149, 150, 60, GR, "#FFFFFF")
text(779, 187, I('z') + ' &#8712; &#8477;' + sub('6', 16, -10), 22, TX, "normal", "middle")
text(779, 232, 'pool state', 18, TX2, "normal", "middle")

line([(854, 179), (910, 179)])
rect(910, 144, 190, 70, GR, GRF)
text(1005, 180, 'structure decoder', 18, TX, "normal", "middle")
text(1005, 204, I('D') + '(' + I('z') + ') = ' + I('Ī') + ' + Δ(' + I('z') + ')', 20, TX,
     "normal", "middle")
thumb(1190, 104, 150, "structure")
line([(1100, 179), (1190, 179)])

# the residual path
line([(145, 254), (145, 318)], arrow=False)
line([(145, 318), (292, 318)])
text(160, 306, I('r') + ' = ' + I('I') + ' &#8722; ' + I('D') + '(' + I('z') + ')', 20, TX2)
rect(292, 288, 150, 62, GR, GRF)
text(367, 326, 'encoder', 19, TX, "normal", "middle")
line([(442, 318), (498, 318)])
rect(498, 288, 200, 62, GR, GRF)
text(598, 320, 'codebook', 19, TX, "normal", "middle")
text(598, 343, '24 &#215; 24, 512 entries', 16, TX2, "normal", "middle")
line([(698, 318), (754, 318)])
thumb(754, 272, 92, "codes")
line([(846, 318), (902, 318)])
rect(902, 288, 150, 62, GR, GRF)
text(977, 326, 'decoder ' + I('G'), 19, TX, "normal", "middle")
line([(1052, 318), (1203, 318)])
add(f'<circle cx="1227" cy="318" r="24" fill="{GYF}" stroke="{GY}" stroke-width="2.4"/>')
text(1227, 327, '+', 26, TX, "normal", "middle")
line([(1340, 179), (1392, 179), (1392, 270), (1251, 270), (1251, 300)], arrow=True)
dot(1340, 179)
line([(1251, 318), (1398, 318)])
thumb(1420, 243, 128, "reconstruction",
      I('D') + '(' + I('z') + ') + ' + I('G') + '(' + I('c') + ')')

# ----------------------------------------------------------------- (b) dynamics
add(f'<rect x="44" y="452" width="{W - 88}" height="300" rx="16" fill="#FFFFFF"/>')
text(64, 436, '(b)', 26, TX, "bold")
text(132, 436, 'dynamics: the two-state NODE driven by the travel speed', 23, TX, "bold")

add(f'<rect x="300" y="470" width="760" height="130" rx="14" fill="{ORF}"/>')
add(f'<rect x="300" y="616" width="760" height="112" rx="14" fill="{BLF}"/>')
text(1090, 500, 'pool state', 20, OR, "bold")
text(1090, 532, 'd' + I('z') + '/d' + I('t') + ' = &#8722;Γ(' + I('z') + ', ' + I('u')
     + ', ' + I('δ') + ') [ ' + I('z') + ' &#8722; ' + I('z') + sub('eq') + '(' + I('u')
     + ', ' + I('δ') + ') ]', 21, TX)
text(1090, 646, 'thermal state', 20, BL, "bold")
text(1090, 678, 'd' + I('δ') + '/d' + I('t') + ' = λ [ ' + I('φ') + '(' + I('z') + ', '
     + I('u') + ') &#8722; ' + I('δ') + ' ]', 21, TX)

rect(64, 568, 150, 60, GY, GYF)
text(139, 597, 'travel speed ' + I('u'), 19, TX, "normal", "middle")
line([(214, 598), (258, 598)], arrow=False)
line([(258, 512), (258, 672)], arrow=False)
dot(258, 598)
for y in (512, 672):
    line([(258, y), (326, y)])

rect(326, 482, 170, 60, OR, "#FFFFFF")
text(411, 519, I('z') + sub('eq') + '(' + I('u') + ', ' + I('δ') + ')', 21, TX, "normal",
     "middle")
rect(326, 548, 170, 44, OR, "#FFFFFF")
text(411, 578, 'Γ(' + I('z') + ', ' + I('u') + ', ' + I('δ') + ')', 20, TX, "normal", "middle")
rect(326, 642, 170, 60, BL, "#FFFFFF")
text(411, 679, I('φ') + '(' + I('z') + ', ' + I('u') + ')', 21, TX, "normal", "middle")

line([(496, 512), (566, 512)], OR, mid="ahO")
line([(496, 570), (566, 570)], OR, mid="ahO")
line([(496, 672), (566, 672)], BL, mid="ahB")
add(f'<circle cx="596" cy="541" r="30" fill="{ORF}" stroke="{OR}" stroke-width="2.4"/>')
text(596, 551, '&#8722;', 28, TX, "normal", "middle")
add(f'<circle cx="596" cy="672" r="30" fill="{BLF}" stroke="{BL}" stroke-width="2.4"/>')
text(596, 682, '&#8722;', 28, TX, "normal", "middle")
line([(626, 541), (688, 541)], OR, mid="ahO")
line([(626, 672), (688, 672)], BL, mid="ahB")

rect(688, 511, 120, 60, OR, ORF)
text(748, 548, '&#8747; d' + I('t'), 22, TX, "normal", "middle")
rect(688, 642, 120, 60, BL, BLF)
text(748, 679, '&#8747; d' + I('t'), 22, TX, "normal", "middle")
line([(808, 541), (900, 541)], OR, mid="ahO")
line([(808, 672), (900, 672)], BL, mid="ahB")
rect(900, 511, 130, 60, OR, "#FFFFFF")
text(965, 548, I('ẑ') + sub('k'), 22, TX, "normal", "middle")
rect(900, 642, 130, 60, BL, "#FFFFFF")
text(965, 679, I('δ') + sub('k'), 22, TX, "normal", "middle")
hat(961, 659)

# state feedback
line([(965, 571), (965, 600), (286, 600), (286, 530), (326, 530)], TX2, dash="7 6")
line([(965, 702), (965, 724), (286, 724), (286, 660), (326, 660)], TX2, dash="7 6")

# ----------------------------------------------------------------- (c) generation
add(f'<rect x="44" y="802" width="{W - 88}" height="330" rx="16" fill="{GYF}"/>')
text(64, 786, '(c)', 26, TX, "bold")
text(132, 786, 'generation: a sequence from the input and the first frame alone', 23, TX, "bold")

rect(70, 862, 180, 64, GY, "#FFFFFF")
text(160, 902, 'input sequence', 19, TX, "normal", "middle")
rect(70, 946, 180, 64, GY, "#FFFFFF")
text(160, 986, 'first frame', 19, TX, "normal", "middle")
line([(250, 894), (300, 894), (300, 936)], arrow=False)
line([(250, 978), (300, 978), (300, 936)], arrow=False)
line([(300, 936), (356, 936)])

rect(356, 902, 170, 68, OR, ORF)
text(441, 944, 'two-state NODE', 19, TX, "normal", "middle")
line([(526, 936), (582, 936)])
rect(582, 906, 150, 60, OR, "#FFFFFF")
text(657, 944, I('ẑ') + ' sequence', 20, TX, "normal", "middle")

line([(732, 936), (788, 936)])
rect(788, 876, 180, 56, GR, GRF)
text(878, 911, I('D') + '(' + I('ẑ') + ')', 21, TX, "normal", "middle")
rect(788, 946, 180, 56, GR, GRF)
text(878, 974, 'retrieved detail', 18, TX, "normal", "middle")
text(878, 996, 'from the data set', 16, TX2, "normal", "middle")
line([(968, 904), (1012, 904), (1012, 936)], arrow=False)
line([(968, 974), (1012, 974), (1012, 936)], arrow=False)
add(f'<circle cx="1044" cy="936" r="24" fill="#FFFFFF" stroke="{GY}" stroke-width="2.4"/>')
text(1044, 945, '+', 26, TX, "normal", "middle")
line([(1068, 936), (1118, 936)])

for i in range(4):
    thumb(1118 + i * 160, 862, 148, f"gen{i}")
text(1118 + 2 * 160, 1052, 'generated frame sequence', 19, TX2, "normal", "middle")

add('</svg>')
svg_text = "\n".join(o)
for d in DESTS:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{STEM}.svg").write_text(svg_text)

src = DESTS[1] / f"{STEM}.svg"
px = round(180.0 / 25.4 * 600)              # 180 mm at 600 dpi
subprocess.run(["inkscape", "--export-type=png", f"--export-width={px}",
                f"--export-filename={DESTS[1] / f'{STEM}.png'}", str(src)], check=True)
subprocess.run(["inkscape", "--export-type=pdf",
                f"--export-filename={DESTS[1] / f'{STEM}.pdf'}", str(src)], check=True)
for ext in ("png", "pdf"):
    shutil.copy(DESTS[1] / f"{STEM}.{ext}", DESTS[0] / f"{STEM}.{ext}")
print(f"written {STEM}.{{svg,png,pdf}} to", ", ".join(str(d) for d in DESTS))
