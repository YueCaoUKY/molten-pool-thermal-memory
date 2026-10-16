"""Figure 1: two-state reduction, physics-structured NODE (simplified), and the two sources of the pool state.

Writes Figure_1_framework.{svg,png,pdf} next to this script. Requires Python with Playwright and a
Chromium browser (python -m playwright install chromium). The dot over delta is drawn as a circle
placed by measuring the glyph in the browser, so the final SVG is the one written by the render step.
"""
import os, asyncio
from playwright.async_api import async_playwright
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figure_1_framework")
# Figure 1: two-state reduction -> physics-structured NODE -> two sources of the pool state
W, H = 1800, 1010
OR, ORF = "#D98A1C", "#FFF5E6"   # pool (orange)
BL, BLF = "#2F6FC4", "#EEF4FD"   # thermal (blue)
GR, GRF = "#23926A", "#E8F6EF"   # learned/encoder (green)
GY, GYF = "#8A94A6", "#F3F4F6"
TX, TX2 = "#1F2937", "#4B5563"
SANS = "DejaVu Sans, Liberation Sans, Arial, sans-serif"
SERIF = "DejaVu Serif, STIX Two Text, Times New Roman, serif"
o = []
def add(s): o.append(s)
def rect(x,y,w,h,stroke,fill,rx=14,sw=2.5,dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def text(x,y,s,size=24,fill=TX,weight="normal",anchor="start",family=SANS,style="normal"):
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" font-weight="{weight}" font-style="{style}" text-anchor="{anchor}">{s}</text>')
def arrow(x1,y1,x2,y2,color=TX2,sw=2.5,dash=None,mid="ah"):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{mid})"/>')
def sub(base, s, size):  # italic base with subscript
    return f'{base}<tspan dy="7" font-size="{int(size*0.68)}">{s}</tspan><tspan dy="-7"></tspan>'

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>')
for mid,col in [("ah",TX2),("ahO",OR),("ahB",BL),("ahG",GR)]:
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
add('<radialGradient id="poolG" cx="50%" cy="0%" r="90%"><stop offset="0%" stop-color="#FFD25A"/><stop offset="55%" stop-color="#F29A2E"/><stop offset="100%" stop-color="#C8571B"/></radialGradient>')
add('<radialGradient id="heatG" cx="50%" cy="0%" r="75%"><stop offset="0%" stop-color="#9CC3F2"/><stop offset="100%" stop-color="#DCE9FA"/></radialGradient>')
add('<clipPath id="plateClip"><rect x="60" y="262" width="460" height="120"/></clipPath>')
add('<clipPath id="iconClip"><rect x="944" y="812" width="190" height="62"/></clipPath>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# ---------------- panel (a) ----------------
rect(20,20,540,600,"#CBD5E1","#FAFBFC",rx=18,sw=2)
text(44,64,"(a) Two-state reduction",30,TX,"bold")
# torch and arc
add(f'<path d="M262 104 L318 104 L306 170 L274 170 Z" fill="#64748B" stroke="#475569" stroke-width="2"/>')
add(f'<line x1="290" y1="170" x2="290" y2="196" stroke="#334155" stroke-width="5"/>')
add(f'<path d="M290 196 Q270 230 262 262 M290 196 Q310 230 318 262" fill="none" stroke="#5DA9F6" stroke-width="3"/>')
# plate, heated region, pool (cross-section)
rect(60,262,460,120,"#9CA3AF","#E5E7EB",rx=4,sw=2)
add('<g clip-path="url(#plateClip)">')
add('<path d="M120 262 A170 92 0 0 0 460 262 Z" fill="url(#heatG)"/>')
add('<path d="M240 262 A50 44 0 0 0 340 262 Z" fill="url(#poolG)" stroke="#B4521A" stroke-width="2"/>')
add('</g>')
# travel direction
arrow(352,238,432,238,TX2,2.5)
text(442,246,"v",24,TX2,"normal","start",SERIF,"italic")
# labels
add(f'<line x1="290" y1="300" x2="290" y2="424" stroke="{OR}" stroke-width="2"/>')
text(56,452,'pool geometry <tspan font-family="'+SERIF+'" font-style="italic">z</tspan>',24,OR,"bold")
add(f'<line x1="430" y1="300" x2="470" y2="424" stroke="{BL}" stroke-width="2"/>')
text(530,452,'stored heat <tspan font-family="'+SERIF+'" font-style="italic">δ</tspan>',24,BL,"bold","end")
text(334,122,'input <tspan font-family="'+SERIF+'" font-style="italic">u</tspan>:',24,TX2)
text(334,150,"current, travel speed",20,TX2)
# lumped balances
add(f'<line x1="44" y1="482" x2="536" y2="482" stroke="#E2E8F0" stroke-width="2"/>')
eq1 = (f'<tspan font-style="italic">C</tspan><tspan dy="7" font-size="17">p</tspan><tspan dy="-7"> </tspan>'
       f'<tspan font-style="italic">ż</tspan> = <tspan font-style="italic">P</tspan><tspan dy="7" font-size="17">p</tspan><tspan dy="-7">(</tspan><tspan font-style="italic">u</tspan>) − '
       f'<tspan font-style="italic">Q</tspan><tspan dy="7" font-size="17">p</tspan><tspan dy="-7">(</tspan><tspan font-style="italic">z</tspan>, <tspan font-style="italic">u</tspan>, <tspan font-style="italic">δ</tspan>)')
eq2 = (f'<tspan font-style="italic">C</tspan><tspan dy="7" font-size="17">n</tspan><tspan dy="-7"> </tspan>'
       f'<tspan font-style="italic">δ&#8288;</tspan> = <tspan font-style="italic">Q</tspan><tspan dy="7" font-size="17">c</tspan><tspan dy="-7">(</tspan><tspan font-style="italic">z</tspan>, <tspan font-style="italic">u</tspan>, <tspan font-style="italic">δ</tspan>) − '
       f'<tspan font-style="italic">K</tspan>(<tspan font-style="italic">u</tspan>)<tspan font-style="italic">δ</tspan>')
text(56,530,"pool",21,OR,"bold")
text(196,530,eq1,25,TX,"normal","start",SERIF)
text(56,580,"workpiece",21,BL,"bold")
text(196,580,eq2,25,TX,"normal","start",SERIF)

# arrow a -> b
arrow(560,330,586,330,TX2,3)

# ---------------- panel (b) ----------------
rect(590,20,1190,600,"#CBD5E1","#FAFBFC",rx=18,sw=2)
text(614,64,"(b) Physics-structured NODE",30,TX,"bold")
rect(1488,42,26,26,GR,GRF,rx=5,sw=2)
text(1524,63,"learned from data",22,TX2)
# input
rect(606,300,134,56,GY,GYF,rx=10,sw=2)
text(673,337,'<tspan font-family="'+SERIF+'" font-style="italic">u</tspan>(<tspan font-family="'+SERIF+'" font-style="italic">t</tspan>)',26,TX,"normal","middle")
arrow(740,328,778,328,TX2,2.5)
# NODE box
rect(780,130,700,410,"#64748B","#FFFFFF",rx=16,sw=2.5)
rect(806,156,440,196,"#CBD5E1","#F8FAFC",rx=12,sw=2)
text(826,190,"relaxation form, from (2)",22,TX2)
text(826,252,'<tspan font-style="italic">ż</tspan> = −Γ [ <tspan font-style="italic">z</tspan> − <tspan font-style="italic">z</tspan><tspan dy="8" font-size="19">eq</tspan><tspan dy="-8"> ]</tspan>',30,OR,"normal","start",SERIF)
text(826,320,'<tspan font-style="italic">δ&#8288;</tspan> = Λ [ <tspan font-style="italic">φ</tspan> − <tspan font-style="italic">δ</tspan> ]',30,BL,"normal","start",SERIF)
rect(806,384,440,130,GR,GRF,rx=12,sw=2)
text(826,420,"equilibria and rates",22,GR,"bold")
text(826,458,'<tspan font-family="'+SERIF+'" font-style="italic">z</tspan><tspan dy="7" font-size="16">eq</tspan><tspan dy="-7">(</tspan><tspan font-family="'+SERIF+'" font-style="italic">u</tspan>, <tspan font-family="'+SERIF+'" font-style="italic">δ</tspan>),  Γ(<tspan font-family="'+SERIF+'" font-style="italic">z</tspan>, <tspan font-family="'+SERIF+'" font-style="italic">u</tspan>, <tspan font-family="'+SERIF+'" font-style="italic">δ</tspan>)',24,TX)
text(826,494,'<tspan font-family="'+SERIF+'" font-style="italic">φ</tspan>(<tspan font-family="'+SERIF+'" font-style="italic">z</tspan>, <tspan font-family="'+SERIF+'" font-style="italic">u</tspan>),  Λ',24,TX)
arrow(1026,384,1026,354,GR,2.5,None,"ahG")
# integrator
rect(1292,226,150,150,"#475569","#F1F5F9",rx=10,sw=2.5)
text(1367,318,'∫ <tspan font-style="italic">dt</tspan>',40,TX,"normal","middle",SERIF)
arrow(1246,262,1290,262,TX2,2.5)
# states
rect(1530,198,220,74,OR,ORF,rx=12,sw=2.5)
text(1640,232,'<tspan font-family="'+SERIF+'" font-style="italic" font-weight="bold">z</tspan>(<tspan font-family="'+SERIF+'" font-style="italic">t</tspan>)',26,OR,"bold","middle")
text(1640,260,"pool state",19,OR,"normal","middle")
rect(1530,328,220,74,BL,BLF,rx=12,sw=2.5)
text(1640,362,'<tspan font-family="'+SERIF+'" font-style="italic" font-weight="bold">δ</tspan>(<tspan font-family="'+SERIF+'" font-style="italic">t</tspan>)',26,BL,"bold","middle")
text(1640,390,"hidden thermal state",19,BL,"normal","middle")
arrow(1442,250,1528,250,OR,3,None,"ahO")
arrow(1442,358,1528,358,BL,3,None,"ahB")
# state feedback
add(f'<polyline points="1750,235 1766,235 1766,104 1026,104" fill="none" stroke="{TX2}" stroke-width="2.5"/>')
add(f'<line x1="1750" y1="365" x2="1766" y2="365" stroke="{TX2}" stroke-width="2.5"/>')
add(f'<line x1="1766" y1="365" x2="1766" y2="235" stroke="{TX2}" stroke-width="2.5"/>')
arrow(1026,104,1026,128,TX2,2.5)
text(1300,96,"current state",20,TX2,"normal","middle")

# ---------------- panel (c) ----------------
rect(20,650,1760,350,"#CBD5E1","#FAFBFC",rx=18,sw=2)
text(44,694,"(c) Sources of the observed pool state",30,TX,"bold")
# images sub-box
rect(40,712,860,272,GY,"#FFFFFF",rx=14,sw=2)
text(62,750,"Process images",25,TX,"bold")
for k,(lab,cx,shape) in enumerate([("GTAW",104,(26,22)),("GMAW",208,(30,18)),("WAAM",312,(22,26))]):
    x0 = cx-44
    add(f'<rect x="{x0}" y="772" width="88" height="88" rx="6" fill="#1E293B"/>')
    add(f'<ellipse cx="{cx}" cy="818" rx="{shape[0]}" ry="{shape[1]}" fill="url(#poolG)"/>')
    text(cx,890,lab,20,TX2,"normal","middle")
arrow(364,816,420,816,TX2,2.5)
rect(424,780,170,74,GY,GYF,rx=10,sw=2)
text(509,812,"DINOv2",22,TX,"bold","middle")
text(509,840,"frozen",20,TX2,"normal","middle")
arrow(594,816,636,816,TX2,2.5)
rect(640,780,150,74,GR,GRF,rx=10,sw=2)
text(715,812,'head <tspan font-family="'+SERIF+'" font-style="italic">h</tspan>',22,TX,"bold","middle")
text(715,840,"trained",20,TX2,"normal","middle")
arrow(790,816,826,816,TX2,2.5)
text(832,824,'<tspan font-family="'+SERIF+'" font-style="italic" font-weight="bold">z</tspan>',26,OR,"bold")
text(62,948,'one network design for GTAW, GMAW, WAAM;  <tspan font-family="'+SERIF+'" font-style="italic">d</tspan> = 6',21,TX2)
# arrow from image state to pool block
arrow(842,796,842,542,OR,3,None,"ahO")
text(997,586,"observed pool state",20,OR,"normal","middle")
# simulation sub-box
rect(920,712,840,272,GY,"#FFFFFF",rx=14,sw=2)
text(1180,750,"Finite-volume simulation",25,TX,"bold")
# plate icon (cross-section) with heated region and dashed 5-15 mm band
rect(944,812,190,62,"#9CA3AF","#E5E7EB",rx=3,sw=2)
add('<g clip-path="url(#iconClip)">')
add('<path d="M964 812 A75 50 0 0 0 1114 812 Z" fill="url(#heatG)"/>')
add('<path d="M1017 812 A22 22 0 0 0 1061 812 Z" fill="url(#poolG)" stroke="#B4521A" stroke-width="1.5"/>')
add('</g>')
add(f'<line x1="1039" y1="770" x2="1039" y2="812" stroke="#475569" stroke-width="4"/>')
text(1039,904,"heat conduction",19,TX2,"normal","middle")
text(1039,928,"with phase change",19,TX2,"normal","middle")
# z from simulation
text(1222,800,'<tspan font-family="'+SERIF+'" font-style="italic" font-weight="bold">z</tspan>: 7 measures of the top-surface',22,OR)
text(1222,830,"liquidus contour (area, width, …)",22,OR)
arrow(1152,820,1152,542,OR,3,None,"ahO")
# withheld thermal reference
text(1222,884,'Δ<tspan font-family="'+SERIF+'" font-style="italic">T</tspan><tspan dy="7" font-size="15">w</tspan><tspan dy="-7">: temperature rise 5–15 mm</tspan>',22,BL)
text(1222,914,"from the arc, withheld",22,BL)
arrow(1690,884,1690,404,BL,3,"9,7","ahB")
text(1676,584,"compared",20,BL,"normal","end")
text(1676,608,'with <tspan font-family="'+SERIF+'" font-style="italic">δ</tspan>',20,BL,"normal","end")

add('</svg>')
open(OUT+'_raw.svg','w').write('\n'.join(o))
print('ok')

JS = r"""
() => {
  const svg = document.querySelector('svg');
  const NS = 'http://www.w3.org/2000/svg';
  for (const t of svg.querySelectorAll('text')) {
    const s = t.textContent;
    let idx = -1;
    // character indices in SVG DOM count UTF-16 code units
    for (let i = 0; i < s.length; i++) {
      if (s[i] === '⁠') {
        const ext = t.getExtentOfChar(i - 1);
        const fs = parseFloat(t.getAttribute('font-size'));
        const c = document.createElementNS(NS, 'circle');
        c.setAttribute('cx', (ext.x + ext.width * 0.66).toFixed(1));
        c.setAttribute('cy', (ext.y + ext.height * 0.015).toFixed(1));
        c.setAttribute('r', (fs * 0.075).toFixed(2));
        c.setAttribute('fill', t.getAttribute('fill'));
        svg.appendChild(c);
      }
    }
  }
  return svg.outerHTML;
}
"""
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1800,'height':1010}, device_scale_factor=2)
        svg = open(OUT+'_raw.svg').read()
        await pg.set_content(f'<html><body style="margin:0">{svg}</body></html>')
        out = await pg.evaluate(JS)
        open(OUT+'.svg','w').write('<?xml version="1.0" encoding="UTF-8"?>\n'+out)
        await pg.screenshot(path=OUT+'.png', clip={'x':0,'y':0,'width':1800,'height':1010})
        await pg.pdf(path=OUT+'.pdf', width='1800px', height='1010px', print_background=True, page_ranges='1')
        await b.close()
asyncio.run(main())
