import re, sys, unicodedata
# Jerga que no entra en el cuerpo por ningún motivo. Los términos que el libro SÍ se
# apropia —red neuronal, neurona, capa, pesos— no se vigilan aquí: se declaran en
# notas/VOCABULARIO.md y los vigila verificar_vocabulario.py, que comprueba algo más
# exigente que su ausencia: que se bauticen antes de usarse (reglas 5 bis y 5 ter).
PROHIBIDAS = ["softmax","vector","matriz","matrices","gradiente","token","embedding",
    "query","key","value","parametro","hiperparametro","logaritm","dimension",
    "producto escalar","funcion de perdida","backprop","tensor","capa oculta"]
GRIEGO = re.compile(r'[Ͱ-Ͽ]')
MATE   = re.compile(r'[=∑∏√∫±×·⋅≈≤≥^]|\b\d+\s*[*/]\s*\d+')
CODIGO = re.compile(r'^\s*(```|import |def |>>> )')
# Envoltorio de imprenta para los trozos que la fuente del libro no dibuja (chino, tailandés,
# árabe). Es formato, no notación: se quita antes de buscar.
MARCADO = re.compile(r'`\\[a-záéíóúñ]+\{(.*?)\}`\{=latex\}')

def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s.lower())
                   if unicodedata.category(c) != 'Mn')

path = sys.argv[1]
lineas = open(path, encoding='utf-8').read().split('\n')

# El capítulo 13 es el mapa de los sótanos: existe precisamente para imprimir los nombres
# técnicos que el resto del libro evita (regla 5 bis). Se le levanta la prohibición de jerga
# —y solo ésa—, en voz alta para que la excepción no pase inadvertida. Las demás siguen: ni
# fórmulas, ni griego, ni código.
ES_EL_MAPA = 'cap13' in path
if ES_EL_MAPA:
    print("NOTA: es el mapa de los sótanos; se permite la jerga y NO se permite nada más.")
fallos = []
for i, l in enumerate(lineas, 1):
    if l.strip().startswith('*[') or l.strip().startswith('[NO EJECUTADO'):
        continue
    l = MARCADO.sub(r'\1', l)
    n = norm(l)
    if not ES_EL_MAPA:
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
