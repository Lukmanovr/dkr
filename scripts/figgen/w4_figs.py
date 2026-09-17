"""Generate Week 4 figures from checked toy arithmetic; no external packages."""
from pathlib import Path
import math
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/figures"
TARGETS = set(sys.argv[1:]) or set(re.findall(
    r"fig-w4-[a-z-]+(?=\.html)",
    (ROOT / "lectures/04-knowledge-graphs.qmd").read_text(encoding="utf-8")))


def svg(body, label, height=180):
    return (f'<svg viewBox="0 0 760 {height}" role="img" aria-label="{label}" '
            'xmlns="http://www.w3.org/2000/svg" font-family="Source Sans 3, sans-serif">'
            '<defs><marker id="arrow-w4" markerWidth="7" markerHeight="7" refX="6" '
            'refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" '
            'fill="var(--dkr-muted, #62626d)"/></marker></defs>' + body + '</svg>')


def text(x, y, value, size=28):
    return (f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" '
            f'fill="var(--dkr-text, #242429)">{value}</text>')


def node(x, y, name, color):
    return (f'<circle cx="{x}" cy="{y}" r="23" fill="{color}"/>'
            f'<text x="{x}" y="{y+9}" text-anchor="middle" font-size="27" '
            f'font-weight="700" fill="white">{name}</text>')


def line(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="7,7"' if dashed else ''
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="var(--dkr-muted, #62626d)" stroke-width="3" '
            f'marker-end="url(#arrow-w4)"{dash}/>')


def card(title, content):
    return f'<section class="w4-card"><h4>{title}</h4>{content}</section>'


def write(name, drawing, content, caption):
    if name not in TARGETS:
        return
    # Give marker IDs document-wide uniqueness when all figures share a page.
    drawing = drawing.replace('arrow-w4', name + '-arrow')
    html = ('```{=html}\n<link rel="stylesheet" href="../assets/d3/w4-learning.css">\n'
            f'<figure class="dkr-fig w4-learning">{drawing}{content}'
            f'<figcaption class="fig-caption">{caption}</figcaption></figure>\n```\n')
    (OUT / f'{name}.html').write_text(html, encoding='utf-8')


body = (line(125, 170, 355, 170) + line(405, 170, 635, 170)
        + '<path d="M100,144 Q380,-10 660,144" fill="none" stroke="var(--dkr-blue, #0f8377)" '
          'stroke-width="3" stroke-dasharray="8,7" marker-end="url(#arrow-w4)"/>'
        + node(100, 170, 'A', '#d9a62e') + node(380, 170, 'B', '#cf4a30')
        + node(660, 170, 'C', '#199473')
        + text(100, 225, 'IU') + text(380, 225, 'Innopolis')
        + text(660, 225, 'Tatarstan') + text(240, 145, 'located_in', 25)
        + text(520, 145, 'located_in', 25) + text(380, 42, 'Proposed edge: located_in ?', 28))
write('fig-w4-hero', svg(body, 'Two stored containment edges; a dashed proposed edge asks whether IU is located in Tatarstan',250),
      '<p class="w4-visual-question"><strong>The question:</strong> should Tatarstan rank highly for (IU, located_in, ?)?</p>',
      'Solid arrows are stored facts. The dashed arrow is the missing candidate fact. '
      'A rule can justify an inference; an embedding model can suggest a ranked answer.')

transe = -math.dist((1+2, 1+1), (3, 1))
distmult = sum(a*b*c for a, b, c in zip((2, 1), (1, 3), (1, 2)))
h, r, t = 1+1j, 1j, -1+1j
cf, cb = (h*r*t.conjugate()).real, (t*r*h.conjugate()).real
rf, rb = -abs(h*r-t), -abs(t*r-h)
assert transe == -1 and distmult == 8 and (cf, cb) == (2, -2)
assert rf == 0 and math.isclose(rb, -math.sqrt(8))
body = (text(180, 44, 'Predict a point') + text(580, 44, 'Measure alignment')
        + line(80, 95, 280, 95) + node(60, 95, 'h', '#d9a62e')
        + node(310, 95, 't', '#199473')
        + text(580, 102, 'multiply → sum', 30)
        + text(180, 155, 'TransE · TransR · RotatE', 26)
        + text(580, 155, 'DistMult · ComplEx', 26))
cards = (card('TransE · real vectors', f'<p>Add, subtract, measure, negate.</p><p>Miss by 1 → score <strong>{transe:g}</strong>.</p>')
         + card('TransR · relation-specific projection', '<p>Project both endpoints, then translate.</p><p>M = [1, 0] gives tail scores <strong>0, 0, −3</strong>.</p>')
         + card('DistMult · real vectors', f'<p>Multiply coordinates, then add.</p><p>Forward <strong>{distmult}</strong>; reverse <strong>{distmult}</strong>.</p>')
         + card('ComplEx · complex vectors', f'<p>Conjugate tail; multiply; sum; real part.</p><p>Forward <strong>{cf:g}</strong>; reverse <strong>{cb:g}</strong>.</p>')
         + card('RotatE · complex entities + angles', f'<p>Rotate head; measure miss; negate.</p><p>Forward <strong>{abs(rf):g}</strong>; reverse <strong>{rb:.3f}</strong>.</p>'))
write('fig-w4-scorers', svg(body, 'Distance scorers predict a point; bilinear scorers measure alignment'),
      '<div class="w4-grid">' + cards + '</div>',
      'Five recipes, one interface: one real score per triple, with larger better. '
      'The values repeat the worked examples; compare rankings within a model, not score magnitudes across models.')

body = (node(110, 80, 'A', '#d9a62e') + node(380, 80, 'B', '#cf4a30')
        + node(650, 80, 'C', '#199473') + line(140, 80, 345, 80)
        + line(410, 80, 615, 80) + text(245, 45, 'r₁') + text(515, 45, 'r₂')
        + text(380, 150, 'Two steps → one combined transformation', 27))
cards = (card('Translation', '<p>r₃ = r₁ + r₂</p><p>Add displacement vectors.</p>')
         + card('Rotation', '<p>r₃ = r₁ ∘ r₂</p><p>Multiply unit rotations; add angles.</p>'))
write('fig-w4-patterns', svg(body, 'A to B by relation one, B to C by relation two'),
      '<div class="w4-grid">' + cards + '</div>',
      'Both TransE and RotatE compose exact transformations. This conditional algebra '
      'does not guarantee logical closure of every high-scoring triple after training.')

entities = {'Tatarstan': 4.0, 'Russia': 4.3, 'Volga': 6.0, 'Kazan': 1.0, 'IU': .4}
scores = {name: -abs(6.0-1.8-value) for name, value in entities.items()}
ordered = sorted(scores, key=scores.get, reverse=True)
raw = ordered.index('Tatarstan')+1
filtered = [name for name in ordered if name != 'Russia'].index('Tatarstan')+1
assert (raw, filtered) == (2, 1) and 'Volga' in ordered
body = (text(180, 45, 'Raw rank') + text(580, 45, 'Filtered rank')
        + text(180, 115, str(raw), 56) + text(580, 115, str(filtered), 56)
        + line(300, 90, 460, 90) + text(380, 155, 'Keep target · exclude another known truth', 26))
rows = ''.join(f'<tr><td>{name}</td><td>{scores[name]:.1f}</td><td>'
               + ('Target: keep' if name == 'Tatarstan' else 'Known true: exclude' if name == 'Russia' else 'Keep')
               + '</td></tr>' for name in ordered)
write('fig-w4-eval', svg(body, 'Tatarstan rank improves from two to one after excluding Russia'),
      '<table><thead><tr><th>Candidate tail</th><th>Score</th><th>Filter action</th></tr></thead><tbody>'
      + rows + '</tbody></table>',
      'Query: (Volga, flows_through, ?), target: Tatarstan. Scores are calculated from '
      'a hand-set one-dimensional TransE model. Russia is another known true answer; '
      'removing it changes the rank without changing any score. All five toy entities are candidates.')
# A complex coordinate is two real coordinates, drawn with equal axis scales.
scale, origin_x, origin_y = 100, 380, 230
point_h = (origin_x + h.real*scale, origin_y - h.imag*scale)
point_t = (origin_x + t.real*scale, origin_y - t.imag*scale)
radius = abs(h)*scale
assert math.isclose(math.dist(point_h, (origin_x, origin_y)), radius)
assert abs(h*r-t) == 0
body = (line(175,230,590,230) + line(380,340,380,25)
        + text(645,239,'real',27) + text(440,38,'imaginary',27)
        + f'<path d="M{point_h[0]},{point_h[1]} A{radius},{radius} 0 0 0 {point_t[0]},{point_t[1]}" '
          'fill="none" stroke="var(--dkr-blue, #0f8377)" stroke-width="4" marker-end="url(#arrow-w4)"/>'
        + line(origin_x,origin_y,*point_h) + line(origin_x,origin_y,*point_t)
        + node(*point_h,'h','#d9a62e') + node(*point_t,'t','#199473')
        + text(590,135,'(1, 1)',28) + text(180,135,'(−1, 1)',28)
        + text(380,190,'90°',30) + text(350,265,'0',27))
write('fig-w4-rotation', svg(body, 'Multiplying 1+i by i rotates (1,1) counterclockwise to (-1,1) about the origin',365),
      '<div class="w4-grid">' + card('Pair of real coordinates','<p>(a, b) → (−b, a)</p>')
      + card('Complex multiplication','<p>(a + bi)i = −b + ai</p>') + '</div>',
      'A quarter-turn preserves distance from the origin. Head and tail are one complex '
      'coordinate each, drawn as pairs of real coordinates. The relation r = i specifies '
      'the turn. We will use this operation in both ComplEx and RotatE, with different scoring rules.')

# Introduce the conceptual pipeline before the companion explains tensor IDs.
body = ''
for x, title, value in [(130, 'A fact', 'head → tail'), (380, 'Vectors', 'h, r, t'), (630, 'Comparison', 'one score')]:
    body += (f'<rect x="{x-105}" y="25" width="210" height="120" rx="10" '
             'fill="var(--dkr-paper, #f8f7f4)" stroke="var(--dkr-border, #d4d4d4)" stroke-width="2"/>'
             + text(x, 68, title) + text(x, 116, value, 27))
body += line(240,85,267,85) + line(490,85,517,85)
write('fig-w4-lookup', svg(body, 'A triple retrieves head, relation, and tail representations; their comparison produces one score'),
      '<div class="w4-grid">' + card('Entities become points', '<p>IU → <strong>(1, 1)</strong></p><p>Innopolis → <strong>(3, 2)</strong></p><p>Reuse each entity’s vector wherever it appears.</p>')
      + card('A relation defines the comparison', '<p>For a translation model:</p><p>located_in → <strong>(2, 1)</strong></p><p>Move the head, then compare it with the tail.</p>') + '</div>',
      'The encoder retrieves representations; the decoder compares them. Different models '
      'change the comparison while keeping the same idea of learned entity vectors.')

body = ''
patterns = [('Symmetry', 'r', 'r'), ('Direction', 'r', None), ('Inverse', 'r₁', 'r₂')]
for j, (name, forward, reverse) in enumerate(patterns):
    y = 75 + j*145
    body += text(120,y+10,name,28) + node(330,y,'a','#d9a62e') + node(650,y,'b','#cf4a30')
    body += line(360,y-10,615,y-10) + text(490,y-25,forward)
    if reverse:
        body += line(620,y+16,365,y+16) + text(490,y+55,reverse)
    else:
        body += text(490,y+55,'no reverse',27)
y=510
body += text(120,y+10,'Compose',28)
for x,n,c in [(280,'a','#d9a62e'),(465,'b','#cf4a30'),(650,'c','#199473')]: body+=node(x,y,n,c)
body += line(307,y,430,y)+line(492,y,615,y)+text(370,y-25,'r₁')+text(555,y-25,'r₂')
body += line(305,y+58,625,y+58,True)+text(465,y+100,'r₃')
y=710
body += text(120,y+10,'One to many',28)+node(330,y,'h','#d9a62e')+node(650,y-40,'t₁','#cf4a30')+node(650,y+40,'t₂','#199473')
body += line(360,y-8,615,y-35)+line(360,y+8,615,y+35)+text(490,y-42,'r')+text(490,y+64,'r')
write('fig-w4-atlas',svg(body,'Five relation patterns: symmetry, direction, inverse relations, composition and one-to-many',805),
      '<p><strong>Read the arrows first:</strong> roommates · parent and child · advisor and advisee · two steps imply a third · one advisor has several students.</p>',
      'These are graph constraints, before choosing any vectors. Solid arrows show facts in the example; '
      'the dashed composition arrow is the consequence of a declared rule. The same letter r means the same relation.')

assert 0+2==2 and 2+2!=0
body=(text(110,83,'Forward')+text(110,213,'Reverse')
      +node(245,75,'a','#d9a62e')+node(435,75,'b','#cf4a30')
      +line(275,75,400,75)+text(340,45,'+2')
      +node(245,205,'a','#d9a62e')+node(435,205,'b','#cf4a30')
      +line(465,205,600,205)+text(535,175,'+2')
      +text(635,215,'×',43)+line(610,260,275,260,True)+text(435,305,'Misses a by 4'))
write('fig-w4-collapse',svg(body,'Translation by plus two fits a to b but sends b farther right, not back to a',335),
      '<div class="w4-grid">'+card('One direction fits','<p>a = 0, r = 2, b = 2.</p><p>Forward score: −|0 + 2 − 2| = <strong>0</strong>.</p>')
      +card('The reverse fails','<p>Keep the same relation r = 2.</p><p>Reverse score: −|2 + 2 − 0| = <strong>−4</strong>.</p>')+'</div>',
      'A translation has the same displacement wherever it starts. Demanding zero error in both directions '
      'forces r = 0 and a = b. This exact-fit argument explains a geometric limitation, not every approximate ranking.')
# ComplEx is a real alignment after a complex transformation.
# Every panel uses 62 pixels per real/imaginary unit.
body = ''
for cx, title in [(190, 'Forward: aligned'), (570, 'Reverse: opposed')]:
    body += text(cx, 35, title, 28) + line(cx-135,180,cx+135,180) + line(cx,290,cx,65)
    body += text(cx+112,218,'real',26) + text(cx+60,83,'imag.',26)
cx, cy, s = 190, 180, 62
body += (f'<line x1="{cx}" y1="{cy}" x2="{cx-s}" y2="{cy-s}" stroke="#199473" stroke-width="10"/>'
         + line(cx,cy,cx-s,cy-s,True) + text(125,100,'hr = t',28))
cx=570
body += line(cx,cy,cx-s,cy+s)+line(cx,cy,cx+s,cy-s)
body += node(cx-s,cy+s,'tr','#cf4a30')+node(cx+s,cy-s,'h','#d9a62e')
body += text(190,330,'score +2',32)+text(570,330,'score −2',32)
assert (h*r).real*t.real+(h*r).imag*t.imag == 2
assert (t*r).real*h.real+(t*r).imag*h.imag == -2
write('fig-w4-complex',svg(body,'Forward transformed head and tail align; in reverse the transformed tail points opposite the original head',355),
      '<div class="w4-grid">'+card('Forward comparison','<p>hr = (−1, 1), t = (−1, 1)</p><p>(−1)(−1) + (1)(1) = <strong>2</strong></p>')
      +card('Reverse comparison','<p>tr = (−1, −1), h = (1, 1)</p><p>(−1)(1) + (−1)(1) = <strong>−2</strong></p>')+'</div>',
      'Use h = 1 + i, r = i, t = −1 + i. The real part of z times the conjugate of w '
      'is the dot product of their two real coordinate pairs. Aligned vectors reward this example; opposed vectors penalize it.')

# The learning cycle names actions; the margin widget supplies the arithmetic.
body = ''
for x,y,title,detail in [(195,75,'1 · Observe','a stored triple'),(565,75,'2 · Sample','replace one entity'),
                          (565,265,'3 · Compare','scores → loss'),(195,265,'4 · Adjust','shared vectors')]:
    body += (f'<rect x="{x-150}" y="{y-55}" width="300" height="110" rx="10" '
             'fill="var(--dkr-paper, #f8f7f4)" stroke="var(--dkr-border, #d4d4d4)" stroke-width="2"/>'
             +text(x,y-6,title,30)+text(x,y+34,detail,26))
body += line(350,75,407,75)+line(565,137,565,202)+line(410,265,350,265)+line(195,205,195,137)
write('fig-w4-training',svg(body,'Observe a fact, sample a comparison, calculate a loss, adjust shared vectors, and repeat',340),
      '<p class="w4-visual-question"><strong>One comparison:</strong> IU → Innopolis versus IU → Kazan, using the same relation.</p>',
      'The scorer is the model’s geometric rule. The loss says how the two scores should differ. '
      'Gradients adjust the shared parameters, so an update changes other triples that use them too.')
# Foundations: preserve semantic types before introducing geometry.
body = node(125,190,'P','#d9a62e') + text(125,245,'Paper')
for y,n,label,rel,ly,col in [(70,'A','Author','written_by',70,'#199473'),
                           (190,'V','Venue','published_at',165,'#cf4a30'),
                           (310,'Y','Year','published_in_year',300,'#7855a4')]:
    body += line(154,190+(y-190)*.055,620,y+(190-y)*.055)
    body += node(650,y,n,col)+text(650,y+50,label)+text(390,ly,rel,28)
write('fig-w4-typed',svg(body,'A paper connects to an author, venue, and year through three different relation types',390),
      '<p class="w4-visual-question"><strong>Same neighborhood, different questions:</strong> who wrote it, where was it published, and when?</p>',
      'Node labels describe kinds of entities; edge labels describe kinds of facts. '
      'This schematic models the year as a node. Removing relation labels loses information even when node types remain.')

body=text(440,35,'Candidate tails')+text(90,235,'Heads')
known={(0,1),(1,2),(2,3)}
for j,n in enumerate('ABCD'): body+=text(300+90*j,85,n)
for i,n in enumerate('ABCD'):
    body+=text(215,142+80*i,n)
    for j in range(4):
        x,y=270+90*j,105+80*i
        fill='var(--dkr-blue, #0f8377)' if (i,j) in known else 'var(--dkr-paper, #f8f7f4)'
        body+=f'<rect x="{x}" y="{y}" width="60" height="60" rx="6" fill="{fill}" stroke="var(--dkr-border, #d4d4d4)"/>'
        cell=text(x+30,y+40,'1' if (i,j) in known else '?',30)
        if (i,j) in known: cell=cell.replace('var(--dkr-text, #242429)','white')
        body+=cell
write('fig-w4-incomplete',svg(body,'One relation as a head-by-tail matrix: three observed facts and thirteen unknown entries',430),
      '<p><strong>One relation, four entities, sixteen possible ordered pairs.</strong> '
      'Three cells are observed. Query (A, r, ?) asks us to compare the unknown cells across row A.</p>',
      '1 means a stored fact; ? means unknown, not false. A separate matrix is needed for each relation. '
      'Benchmark masking can hide a recorded truth among the question marks.')

body=''
for offset,a,b in [(0,'a','b'),(250,'c','d'),(500,'e','f')]:
    body+=node(65+offset,185,a,'#d9a62e')+node(215+offset,110,b,'#199473')
    body+=line(91+offset,172,187+offset,124)+text(140+offset,65,'r = (2, 1)',26)
write('fig-w4-parallel',svg(body,'Three equal parallel relation displacements connect different pairs of entities',230),
      '<div class="w4-grid">'+card('Different starting points','<p>a = (0, 0), c = (0, 2), e = (3, 0)</p>')
      +card('The same displacement','<p>b = (2, 1), d = (2, 3), f = (5, 1)</p>')+'</div>',
      'Subtract the head from the tail in each pair: the difference is always (2,1). '
      'The three mini-diagrams have separate origins and equal scales. Sharing r does not force the heads to coincide.')
for a,b in [((0,0),(2,1)),((0,2),(2,3)),((3,0),(5,1))]: assert tuple(y-x for x,y in zip(a,b))==(2,1)

body=text(380,32,'Undo one step')+node(120,110,'h','#d9a62e')+node(640,110,'t','#199473')
body+=line(150,94,608,94)+line(610,128,151,128)+text(380,76,'r = (2, 1)')+text(380,177,'r inverse = (−2, −1)')
body+=text(380,247,'Combine two steps')
body+=node(120,430,'a','#d9a62e')+node(380,315,'b','#cf4a30')+node(640,430,'c','#199473')
body+=line(150,416,350,329)+line(410,329,610,416)+line(150,438,610,438)
body+=text(205,348,'r₁')+text(555,348,'r₂')+text(380,492,'r₃ = r₁ + r₂')
write('fig-w4-path',svg(body,'Inverse relations negate a displacement; a two-step path adds displacement vectors',525),
      '<p><strong>Composition example:</strong> a = (0,0), b = (2,1), c = (3,3). '
      'The two shifts (2,1) and (1,2) combine into (3,3).</p>',
      'Top: the reverse uses a different relation vector. Bottom: the direct relation is constrained '
      'to equal the sum. Arrow geometry is schematic; the stated coordinates define the exact calculation.')

body=text(110,42,'Kernel')+text(260,42,'Exact tails')+text(605,42,'Output')
body+=line(60,230,335,230)+line(110,360,110,80)+line(250,360,250,80)
body+=node(250,160,'t₁','#cf4a30')+node(250,300,'t₂','#199473')
body+=text(285,125,'(2,1)',26)+text(285,350,'(2,−1)',26)
body+=line(360,230,470,230)+text(415,195,'M')+line(505,230,705,230)+node(620,230,'2','#199473')
body+=text(110,405,'(0,v)')+text(260,405,'(2,v)')+text(620,290,'Mt₁ = Mt₂',28)
write('fig-w4-kernel',svg(body,'The map M equals [1,0]. Vertical shifts vanish; the line x=2 maps to one output value 2',440),
      '<p><strong>M = [1, 0].</strong> Kernel shifts are (0,v). Exact tails for a projected target of 2 are (2,v). '
      'The two sets are parallel, but only the kernel passes through the origin.</p>',
      'Points on the same vertical line share an output. A second relation using [0,1] would keep the vertical difference instead.')

A=(0,1,7); B=(2,1,4); C=(5,8,4)
assert (B[0]-A[0],B[1]-A[1])==(2,0)
assert (C[0]-B[0],C[2]-B[2])==(3,0)
assert (C[0]-A[0],0)==(5,0)
body=node(120,160,'a','#d9a62e')+node(380,80,'b','#cf4a30')+node(640,160,'c','#199473')
body+=line(148,151,350,90)+line(410,90,612,151)+line(151,171,609,171)
body+=text(245,72,'r₁')+text(515,72,'r₂')+text(380,220,'r₃: keep only the shared constraint',26)
body+=text(120,270,'(0,1,7)')+text(380,30,'(2,1,4)')+text(640,270,'(5,8,4)')
write('fig-w4-kernel-compose',svg(body,'Three entities fit two projected translations; the composed map preserves only the first coordinate',300),
      '<table><thead><tr><th>Map</th><th>Kept coordinates</th><th>Shift</th></tr></thead><tbody>'
      '<tr><td>M₁</td><td>(x, y)</td><td>(2, 0)</td></tr><tr><td>M₂</td><td>(x, z)</td><td>(3, 0)</td></tr>'
      '<tr><td>M₃</td><td>(x, 0)</td><td>(5, 0)</td></tr></tbody></table>',
      'The first step can change z freely, the second can change y freely. The composed constraint '
      'keeps x and discards both freedoms. All three maps have shape 2×3; M₃ uses a zero second row.')

body=''
for cx,title,vals in [(200,'Full matrix',[1,2,0,3]),(560,'Diagonal matrix',[1,0,0,3])]:
    body+=text(cx,40,title)
    for i,val in enumerate(vals):
        x=cx-100+(i%2)*105;y=80+(i//2)*100
        fill='var(--dkr-paper, #f8f7f4)' if i in [1,2] else 'var(--dkr-border, #d4d4d4)'
        body+=f'<rect x="{x}" y="{y}" width="95" height="85" rx="8" fill="{fill}" stroke="var(--dkr-muted, #62626d)"/>'
        body+=text(x+47.5,y+56,str(val),40)
    body+=text(cx,327,'Forward 16' if cx==200 else 'Forward 8')
    body+=text(cx,373,'Reverse 10' if cx==200 else 'Reverse 8')
assert 2*(1*1+2*2)+1*(0*1+3*2)==16
assert 1*(1*2+2*1)+2*(0*2+3*1)==10
write('fig-w4-bilinear-matrix',svg(body,'A full two-by-two bilinear matrix distinguishes reversal; its diagonal restriction gives equal scores',405),
      '<p><strong>Use h = (2,1), t = (1,2), and score hᵀAt.</strong> '
      'The off-diagonal 2 couples the first head feature to the second tail feature. DistMult removes cross-coordinate terms.</p>',
      'A general relation matrix has d² entries; a diagonal one has d free entries. '
      'The cheaper parameterization imposes symmetry. Full matrices can still be symmetric; direction is permitted, not forced.')

body=text(190,35,'Distance: one circle',28)+text(565,35,'Alignment: one line',28)
body+=line(55,300,340,300)+line(75,345,75,90)
body+='<circle cx="195" cy="240" r="60" fill="none" stroke="var(--dkr-blue, #0f8377)" stroke-width="3" stroke-dasharray="7,5"/>'
body+=node(195,240,'q','#d9a62e')+node(195,180,'t₂','#199473')+node(195,300,'t₁','#cf4a30')
body+=text(275,180,'−1')+text(275,350,'−1')
body+=line(445,240,725,240)+line(470,350,470,90)+line(590,345,590,95)
body+='<line x1="470" y1="240" x2="580" y2="240" stroke="var(--dkr-blue, #0f8377)" stroke-width="4"/>'
body+='<path d="M578,232 L590,240 L578,248 Z" fill="var(--dkr-blue, #0f8377)"/>'+text(530,220,'q')
body+=node(590,180,'t₁','#cf4a30')+node(590,300,'t₂','#199473')+text(670,180,'4')+text(670,310,'4')
write('fig-w4-levelsets',svg(body,'Two distinct tails share a nonzero distance on a circle; two distinct tails share a dot product on a line',380),
      '<div class="w4-grid">'+card('TransE: q = h + r = (2,1)','<p>t₁ = (2,0), t₂ = (2,2).</p><p>Both distances are 1; both scores are −1. Zero distance would leave only q.</p>')
      +card('DistMult: q = h ∘ r = (2,0)','<p>t₁ = (2,1), t₂ = (2,−1).</p><p>Both scores are 2×2 = 4. The vertical coordinate does not affect this score.</p>')+'</div>',
      'A level set collects equal-score candidates. Distance contours shrink to one point at zero error; '
      'a fixed dot-product contour can contain an entire line of distinct tails.')

body=line(155,240,650,240)+line(380,385,380,70)+text(700,250,'real',27)+text(440,75,'imag.',27)
body+=node(510,175,'z','#cf4a30')+node(510,305,'z̄','#199473')+line(510,200,510,277,True)
body+=text(600,163,'2 + i')+text(600,352,'2 − i')+text(330,276,'0')
write('fig-w4-conjugate',svg(body,'Conjugation reflects 2+i to 2-i across the real axis, keeping the real coordinate',410),
      '<p><strong>Conjugation:</strong> (a,b) → (a,−b). '
      '<strong>Multiplication:</strong> (a+bi)(c+di) = (ac−bd) + (ad+bc)i.</p>',
      'Conjugation is a reflection, while multiplication by i is a quarter-turn. '
      'They are different operations. ComplEx applies the conjugate to the tail so reversal changes which entity is reflected.')

# Historical measurements reproduced from Stanford PDF page 62, not a local run.
bench=[('TransE',.294,.226),('DistMult',.241,.430),('ComplEx',.247,.440),('RotatE',.338,.476)]
body=''
for top,title,column in [(0,'FB15k-237 · reported MRR',1),(385,'WN18RR · reported MRR',2)]:
    body+=text(380,top+35,title,29)
    for i,row in enumerate(bench):
        y=top+95+i*60; end=250+row[column]/.5*400
        body+=text(120,y+10,row[0])+text(710,y+10,f'{row[column]:.3f}',28)
        body+=f'<line x1="250" y1="{y}" x2="{end}" y2="{y}" stroke="var(--dkr-blue, #0f8377)" stroke-width="12"/>'
        body+=f'<circle cx="{end}" cy="{y}" r="8" fill="var(--dkr-accent, #d9603b)"/>'
    body+=f'<line x1="250" y1="{top+320}" x2="650" y2="{top+320}" stroke="var(--dkr-muted, #62626d)" stroke-width="2"/>'
    for v in [0,.25,.5]: body+=text(250+v/.5*400,top+355,str(v),26)
assert bench[0][1]>bench[1][1] and bench[0][2]<bench[1][2]
write('fig-w4-benchmark',svg(body,'Historical MRR comparison: TransE outranks DistMult on FB15k-237 but their order reverses on WN18RR',765),
      '<p><strong>Predict the lesson:</strong> why does the TransE versus DistMult ordering change between datasets?</p>',
      'Both axes span 0–0.5. '
      'These are reported benchmark results, not measurements from our toy walkthrough and not a current leaderboard.')
print(f'Generated {len(TARGETS)} Week 4 figures; toy arithmetic and historical comparison values checked.')
