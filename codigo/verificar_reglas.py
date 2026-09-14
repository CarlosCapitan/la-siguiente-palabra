import re, sys, unicodedata
PROHIBIDAS = ["softmax","vector","matriz","matrices","gradiente","token","embedding",
    "query","key","value","parametro","hiperparametro","logaritm","dimension",
    "producto escalar","funcion de perdida","backprop","tensor","capa oculta","neurona artificial"]
GRIEGO = re.compile(r'[Ͱ-Ͽ]')
MATE   = re.compile(r'[=∑∏√∫±×·⋅≈≤≥^]|\b\d+\s*[*/]\s*\d+')
CODIGO = re.compile(r'^\s*(```|import |def |>>> )')

def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s.lower())
                   if unicodedata.category(c) != 'Mn')

path = sys.argv[1]
lineas = open(path, encoding='utf-8').read().split('\n')
fallos = []
for i, l in enumerate(lineas, 1):
    if l.strip().startswith('*[') or l.strip().startswith('[NO EJECUTADO'):
        continue
    n = norm(l)
    for p in PROHIBIDAS:
        if p in n:
            fallos.append((i, 'jerga', p, l.strip()[:70]))
    if GRIEGO.search(l): fallos.append((i,'griego','',l.strip()[:70]))
    if MATE.search(l):   fallos.append((i,'notacion','',l.strip()[:70]))
    if CODIGO.match(l):  fallos.append((i,'codigo','',l.strip()[:70]))

if fallos:
    for f in fallos: print(f"  linea {f[0]:>4}  {f[1]:<9} {f[2]:<16} {f[3]}")
    print(f"\nFALLA: {len(fallos)} incidencia(s)")
else:
    print("PASA: cero notacion, cero jerga sin traducir")
