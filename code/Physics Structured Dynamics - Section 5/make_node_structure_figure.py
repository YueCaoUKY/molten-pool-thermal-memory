"""Figure 2: block structure of the physics-structured NODE (Simulink style).

Writes Figure_2_NODE_structure.{svg,png,pdf} next to this script. Requires Python with Playwright and a
Chromium browser (python -m playwright install chromium). The dot over delta is drawn as a circle
placed by measuring the glyph in the browser, so the final SVG is the one written by the render step.
"""
import os, asyncio
from playwright.async_api import async_playwright
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figure_2_NODE_structure")
# Figure 2: block structure of the physics-structured NODE (Simulink-style)
W, H = 1800, 760
OR, ORF = "#D98A1C", "#FFF5E6"
BL, BLF = "#2F6FC4", "#EEF4FD"
GR, GRF = "#23926A", "#E8F6EF"
GY, GYF = "#64748B", "#F1F5F9"
TX, TX2 = "#1F2937", "#4B5563"
SANS = "DejaVu Sans, Liberation Sans, Arial, sans-serif"
SERIF = "DejaVu Serif, STIX Two Text, Times New Roman, serif"
o=[]
def add(s): o.append(s)
def rect(x,y,w,h,stroke,fill,rx=10,sw=2.5,dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def text(x,y,s,size=22,fill=TX,weight="normal",anchor="start",family=SANS,style="normal"):
    add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" font-weight="{weight}" font-style="{style}" text-anchor="{anchor}">{s}</text>')
def line(pts,color=TX2,sw=2.4,arrow=True,mid="ah",dash=None):
    d=f' stroke-dasharray="{dash}"' if dash else ''
    m=f' marker-end="url(#{mid})"' if arrow else ''
    add(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="{sw}"{d}{m}/>')
def dot(x,y,c=TX2): add(f'<circle cx="{x}" cy="{y}" r="5" fill="{c}"/>')
def I(s): return f'<tspan font-family="{SERIF}" font-style="italic">{s}</tspan>'
def sub(s,sz=15,dy=7): return f'<tspan dy="{dy}" font-size="{sz}">{s}</tspan><tspan dy="-{dy}">&#8203;</tspan>'
def circ(x,y,label,r=24):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{GYF}" stroke="{GY}" stroke-width="2.4"/>')
    text(x,y+9,label,26,TX,"normal","middle")
def tri(x,y,w,h,label,stroke,fill,lsize=24,lfam=SANS):
    add(f'<path d="M{x} {y-h/2} L{x} {y+h/2} L{x+w} {y} Z" fill="{fill}" stroke="{stroke}" stroke-width="2.4"/>')
    text(x+w*0.36,y+8,label,lsize,TX,"normal","middle",lfam)

add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>')
for mid,col in [("ah",TX2),("ahO",OR),("ahB",BL)]:
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
# bands
add(f'<rect x="272" y="66" width="800" height="306" rx="16" fill="#FFF8EE"/>')
add(f'<rect x="272" y="444" width="800" height="168" rx="16" fill="#F2F7FE"/>')
text(1100,80,'pool state equation',22,OR,"bold")
text(1100,114,I('ż')+' = −Γ('+I('z')+', '+I('u')+', '+I('δ')+') [ '+I('z')+' − '+I('z')+sub('eq')+'('+I('u')+', '+I('δ')+') ]',24,TX)
text(1140,454,'thermal state equation',22,BL,"bold")
text(1140,488,I('δ&#8288;')+' = Λ [ '+I('φ')+'('+I('z')+', '+I('u')+') − '+I('δ')+' ]',24,TX)

# input u and bus
rect(26,356,110,50,GY,GYF,rx=10)
text(81,390,I('u')+'('+I('t')+')',24,TX,"normal","middle")
line([(136,381),(190,381)],arrow=False)
line([(190,116),(190,492)],arrow=False); dot(190,381)
for y in (116,266,492):
    line([(190,y),(298,y)]); dot(190,y) if y not in (116,492) else None
text(160,370,I('u'),20,TX2,"normal","middle")

# MLP blocks
rect(300,84,230,100,GR,GRF)
text(415,120,"MLP",22,GR,"bold","middle"); text(415,158,I('z')+sub('eq',15),24,TX,"normal","middle")
rect(300,236,230,120,GR,GRF)
text(415,270,"MLP",22,GR,"bold","middle"); text(415,306,'Γ = diag('+I('γ')+sub('j',15)+')',22,TX,"normal","middle")
text(415,338,'1/'+I('γ')+sub('j',14)+' ≤ '+I('τ')+sub('0',14),20,TX2,"normal","middle")
rect(300,456,230,104,GR,GRF)
text(415,490,"MLP",22,GR,"bold","middle"); text(415,524,I('φ'),24,TX,"normal","middle")
text(415,550,'≥ 0 (softplus)',19,TX2,"normal","middle")

# pool row operations
circ(620,134,"Σ")
text(600,100,"+",22,TX,"bold","middle"); text(588,128,"−",24,TX,"bold","middle")
line([(530,134),(594,134)],OR,mid="ahO"); text(562,124,I('z')+sub('eq',13,6),18,OR,"normal","middle")
circ(740,134,"×")
line([(644,134),(714,134)]); text(679,114,I('z')+' − '+I('z')+sub('eq',13,6),16,TX2,"normal","middle")
line([(530,296),(740,296),(740,160)]); text(756,232,'Γ',22,TX2)
tri(796,134,62,56,'−1',GY,GYF,20)
line([(764,134),(794,134)])
rect(930,98,100,72,GY,GYF,rx=8)
text(980,144,'1/'+I('s'),26,TX,"normal","middle")
line([(858,134),(928,134)]); text(894,122,I('ż'),22,TX,"normal","middle",SERIF)
text(980,196,I('z')+'(0) = '+I('z')+sub('1',14),19,TX2,"normal","middle")
# z output and feedback
line([(1030,134),(1076,134)],OR,arrow=False); dot(1076,134,OR)
line([(1076,134),(1546,134)],OR,mid="ahO")
line([(1076,134),(1076,46),(620,46),(620,108)],OR,mid="ahO"); dot(620,46,OR)
line([(620,46),(250,46),(250,526)],OR,arrow=False)
line([(250,296),(298,296)],OR,mid="ahO"); dot(250,296,OR)
line([(250,526),(298,526)],OR,mid="ahO")
rect(1548,104,210,60,OR,ORF,rx=10)
text(1653,142,I('ẑ')+'('+I('t')+')  pool state',22,OR,"bold","middle")

# thermal row operations
circ(620,508,"Σ")
text(586,500,"+",22,TX,"bold","middle"); text(642,552,"−",24,TX,"bold","middle")
line([(530,508),(594,508)],BL,mid="ahB"); text(562,498,I('φ'),20,BL,"normal","middle")
line([(644,508),(690,508)])
tri(692,508,96,74,'Λ',GR,GRF,26,SERIF)
text(740,578,'1/'+I('λ')+sub('k',14)+' ∈ ['+I('τ')+sub('k−1',13)+', '+I('τ')+sub('k',13)+']',19,TX2,"normal","middle")
rect(930,472,100,72,GY,GYF,rx=8)
text(980,518,'1/'+I('s'),26,TX,"normal","middle")
line([(788,508),(928,508)])
text(864,494,I('δ&#8288;'),22,TX,"normal","middle")
text(980,570,I('δ')+'(0) = 0',19,TX2,"normal","middle")
# delta output and feedback
line([(1030,508),(1104,508)],BL,arrow=False); dot(1104,508,BL)
line([(1104,508),(1546,508)],BL,mid="ahB")
line([(1104,508),(1104,632),(620,632),(620,534)],BL,mid="ahB"); dot(620,632,BL)
line([(620,632),(220,632),(220,152)],BL,arrow=False)
line([(220,326),(298,326)],BL,mid="ahB"); dot(220,326,BL)
line([(220,152),(298,152)],BL,mid="ahB")
rect(1548,478,210,60,BL,BLF,rx=10)
text(1653,516,I('δ')+'('+I('t')+')  hidden',22,BL,"bold","middle")

# legend
rect(30,690,28,28,GR,GRF,rx=6,sw=2)
text(70,712,"learned from data: pool equilibrium, pool rates, thermal targets, thermal rates",20,TX2)
add(f'<circle cx="1044" cy="704" r="14" fill="{GYF}" stroke="{GY}" stroke-width="2.4"/>')
text(1070,712,"fixed by the structure: sum, product, gain, integrator",20,TX2)
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
        pg = await b.new_page(viewport={'width':1800,'height':760}, device_scale_factor=2)
        svg = open(OUT+'_raw.svg').read()
        await pg.set_content(f'<html><body style="margin:0">{svg}</body></html>')
        out = await pg.evaluate(JS)
        open(OUT+'.svg','w').write('<?xml version="1.0" encoding="UTF-8"?>\n'+out)
        await pg.screenshot(path=OUT+'.png', clip={'x':0,'y':0,'width':1800,'height':760})
        await pg.pdf(path=OUT+'.pdf', width='1800px', height='760px', print_background=True, page_ranges='1')
        await b.close()
asyncio.run(main())
