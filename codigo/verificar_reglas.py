#!/usr/bin/env python3
"""Vigila la regla 1 del libro: ni notación, ni jerga sin traducir, ni código en el cuerpo.

    verificar_reglas.py capitulo.md
    verificar_reglas.py --selftest

Es el verificador que sostiene la promesa del prólogo —«en este libro no hay ni una sola
fórmula»—, así que se comprueba a sí mismo como cualquier otra medición del libro."""
import re, sys, unicodedata

# Jerga que no entra en el cuerpo por ningún motivo. Los términos que el libro SÍ se
# apropia —red neuronal, neurona, capa, pesos— no se vigilan aquí: se declaran en
# notas/VOCABULARIO.md y los vigila verificar_vocabulario.py, que comprueba algo más
# exigente que su ausencia: que se bauticen antes de usarse (reglas 5 bis y 5 ter).
PROHIBIDAS = ["softmax","vector","matriz","matrices","gradiente","token","embedding",
    "query","key","value","parametro","hiperparametro","logaritm","dimension",
    "producto escalar","funcion de perdida","backprop","tensor","capa oculta",
    "n-grama","bigrama","trigrama"]
GRIEGO = re.compile(r'[Ͱ-Ͽ]')
MATE   = re.compile(r'[=∑∏√∫±×·⋅≈≤≥^]|\b\d+\s*[*/]\s*\d+')
CODIGO = re.compile(r'^\s*(```|import |def |>>> )')
# Envoltorio de imprenta para los trozos que la fuente del libro no dibuja (chino, tailandés,
# árabe). Es formato, no notación: se quita antes de buscar.
MARCADO = re.compile(r'`\\[a-záéíóúñ]+\{(.*?)\}`\{=latex\}')


def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s.lower())
                   if unicodedata.category(c) != 'Mn')


def revisar(lineas, es_el_mapa=False):
    """Devuelve la lista de incidencias: (número de línea, clase, término, trozo)."""
    fallos = []
    for i, l in enumerate(lineas, 1):
        if l.strip().startswith('*[') or l.strip().startswith('[NO EJECUTADO'):
            continue
        l = MARCADO.sub(r'\1', l)
        n = norm(l)
        if not es_el_mapa:
            for p in PROHIBIDAS:
                if p in n:
                    fallos.append((i, 'jerga', p, l.strip()[:70]))
        if GRIEGO.search(l): fallos.append((i, 'griego', '', l.strip()[:70]))
        if MATE.search(l):   fallos.append((i, 'notacion', '', l.strip()[:70]))
        if CODIGO.match(l):  fallos.append((i, 'codigo', '', l.strip()[:70]))
    return fallos


def clases(fallos):
    return sorted({f[1] for f in fallos})


def selftest():
    """Tres pruebas sobre lo único que este verificador decide: si una línea del libro lleva
    notación, jerga o código."""
    fallos = []

    # 1. TEST NULO — prosa limpia del libro, con cifras y con rayas dentro. Si aquí saltara
    #    algo, el verificador estaría marcando texto bueno y acabaríamos desactivándolo.
    limpio = ["En 1948, un ingeniero de la compañía telefónica Bell publicó un artículo.",
              "Aquel ZX81 tenía un kilobyte: mil veinticuatro bytes, ni uno más.",
              "Donde el suelo quede cerca —son cuatro sitios— te lo diré.",
              "La máquina acierta el 90,4 % de las veces, frente al 78,1 % de antes.",
              "    python ngrama.py --selftest"]   # el NOMBRE del programa no es jerga
    sale = revisar(limpio)
    print(f"[1] test nulo         prosa limpia: {len(sale)} incidencias"
          + (f" ({clases(sale)})" if sale else ""))
    if sale:
        fallos.append(f"test nulo: marca prosa buena: {sale[0]}")

    # 2. SEÑAL IMPLANTADA — una línea mala de cada clase, de una en una. Tiene que cazar las
    #    cuatro, y cada una por su nombre, no cuatro veces la misma.
    implantadas = {'jerga':    "El modelo guarda cada palabra como un vector de números.",
                   'griego':   "Se ajusta con un paso de tamaño α en cada vuelta.",
                   'notacion': "El resultado es 3 × 4 ≈ 12, redondeando.",
                   'codigo':   "import numpy as np"}
    implantadas['jerga2'] = "Eso es lo que hace un modelo de n-gramas, ni más ni menos."
    implantadas['jerga'], implantadas['jerga2'] = implantadas['jerga'], implantadas['jerga2']
    cazadas = {}
    for clase, linea in implantadas.items():
        esperada = 'jerga' if clase.startswith('jerga') else clase
        cazadas[clase] = esperada in clases(revisar([linea]))
    print("[2] señal implantada  " + "; ".join(
        f"{c}: {'la caza' if v else 'SE LE ESCAPA'}" for c, v in cazadas.items()))
    if not all(cazadas.values()):
        fallos.append("señal implantada: se le escapa "
                      + ", ".join(c for c, v in cazadas.items() if not v))

    # 3. INVARIANTE DEL DOMINIO — dos cosas que tienen que seguir siendo verdad pase lo que
    #    pase. Primera: en el mapa de los sótanos se levanta la prohibición de jerga Y NADA
    #    MÁS; una fórmula ahí sigue siendo una fórmula. Segunda: el envoltorio de imprenta
    #    que dibuja el chino es formato, no notación, y no puede contar como incidencia.
    en_el_mapa = revisar([implantadas['jerga'], implantadas['notacion']], es_el_mapa=True)
    solo_notacion = clases(en_el_mapa) == ['notacion']
    envuelto = revisar(["La máquina escribió `\\textcjk{下}`{=latex} y siguió."])
    print(f"[3] invariante        en el mapa queda {clases(en_el_mapa)} "
          f"({'bien: solo la notación' if solo_notacion else 'MAL'}); "
          f"envoltorio de imprenta: {len(envuelto)} incidencias")
    if not solo_notacion or envuelto:
        fallos.append("invariante: el mapa levanta solo la jerga, y el envoltorio no es notación")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


if '--selftest' in sys.argv:
    sys.exit(selftest())

path = sys.argv[1]
lineas = open(path, encoding='utf-8').read().split('\n')

# El mapa de los sótanos (hoy el capítulo 14) existe precisamente para imprimir los nombres
# técnicos que el resto del libro evita (regla 5 bis). Se le levanta la prohibición de jerga
# —y solo ésa—, en voz alta para que la excepción no pase inadvertida. Las demás siguen: ni
# fórmulas, ni griego, ni código.
ES_EL_MAPA = 'cap14' in path
if ES_EL_MAPA:
    print("NOTA: es el mapa de los sótanos; se permite la jerga y NO se permite nada más.")

fallos = revisar(lineas, ES_EL_MAPA)
if fallos:
    for f in fallos:
        print(f"  linea {f[0]:>4}  {f[1]:<9} {f[2]:<16} {f[3]}")
    print(f"\nFALLA: {len(fallos)} incidencia(s)")
    sys.exit(1)
print("PASA: cero notacion, cero jerga sin traducir")
