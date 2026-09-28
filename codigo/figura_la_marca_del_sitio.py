#!/usr/bin/env python3
"""
Capítulo 9 — la marca del sitio, dibujada (L24).

«A los números de cada palabra se les suma un patrón que codifica en qué posición estaba.» En la
máquina del Quijote del capítulo 8 se ve con una letra: la «e» de «quiero» (sitio 4) y la de
«acordarme» (sitio 16) tienen la misma lista de números; a cada una se le suma la marca de su sitio,
y lo que entra en la máquina son dos listas distintas. Cada barra es uno de los seis primeros
números de la lista: hacia arriba si es positivo, hacia abajo si es negativo, con su valor escrito.

Los números NO se escriben aquí: se leen del apartado 3 de `datos/salidas/la_e_de_acordarme.txt`.
En gris: el libro se imprime en negro.

Uso:
    python figura_la_marca_del_sitio.py --selftest
    python figura_la_marca_del_sitio.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/la_e_de_acordarme.txt"
DESTINO = "../figuras/la_marca_del_sitio.png"
ALTO = 4.7                   # pulgadas
ALTO_FILA = 8.8              # unidades del lienzo por fila de barras
ESCALA = 3.0                 # unidades del lienzo por cada 1 de valor

# ==========================================================

import argparse
import re
import sys
from pathlib import Path

from matplotlib.patches import Rectangle

from infografia import COLOR, GRIS, Lienzo

AQUI = Path(__file__).resolve().parent


def leer(ruta):
    """Del apartado 3: dos bloques de tres filas (lista, marca, lo que entra), con sus números."""
    texto = Path(ruta).read_text(encoding="utf-8")
    assert "3. LA MISMA «e» EN DOS SITIOS" in texto, f"se esperaba el apartado 3 en {ruta}"
    tres = texto.split("3. LA MISMA «e» EN DOS SITIOS", 1)[1].split("\n4.", 1)[0]
    filas = []
    for m in re.finditer(r"^  (la lista de la «e»|más la marca del sitio \d+|da: lo que entra)"
                         r"((?:\s+-?\d+,\d+)+)$", tres, re.M):
        filas.append((m.group(1), [float(v.replace(",", ".")) for v in m.group(2).split()]))
    assert len(filas) == 6, f"se esperaban seis filas; hay {len(filas)}"
    bloques = [filas[:3], filas[3:]]
    for b in bloques:
        for a, m_, s in zip(*[v for _, v in b]):
            assert abs(a + m_ - s) <= 0.011, f"la suma no cuadra: {a} + {m_} no es {s}"
    return bloques


def dibujar(bloques, paleta, ruta):
    L = Lienzo("La misma letra, en dos sitios",
               "La «e» de «quiero» (sitio 4) y la de «acordarme» (sitio 16): la misma lista,\n"
               "más la marca de su sitio. Los seis primeros de sus 32 números.", paleta, alto=ALTO)
    p, ax = L.p, L.ax
    n_barras = 0
    for n, bloque in enumerate(bloques, 1):
        sitio = re.search(r"\d+", bloque[1][0]).group(0)
        x0, ytop, ancho = L.panel(n, f"La «e» en el sitio {sitio}", 3 * ALTO_FILA + 10.4)
        for i, (nombre, vals) in enumerate(bloque):
            y0 = ytop + 1.0 - (i + 0.5) * ALTO_FILA
            rot = {0: "la lista de la «e»", 1: f"más la marca del sitio {sitio}", 2: "da: lo que entra"}[i]
            L.texto(x0, y0, rot, tam=7.2,
                    negrita=(i == 2))
            ax.plot([x0 + 30, x0 + ancho], [y0, y0], color=p.marco, linewidth=0.6)
            paso = (ancho - 32) / len(vals)
            for j, v in enumerate(vals):
                x = x0 + 32 + j * paso
                alto = v * ESCALA
                ax.add_patch(Rectangle((x, min(y0, y0 + alto)), paso * 0.55, abs(alto),
                                       facecolor=p.acento if i == 2 else p.contra if i == 1 else p.neutro,
                                       edgecolor=p.tinta, linewidth=0.4))
                n_barras += 1
                L.texto(x + paso * 0.275, y0 + alto + (1.3 if v >= 0 else -1.3),
                        f"{v:.2f}".replace(".", ","), ha="center", tam=6.2)
    L.pie("Cada barra, un número: hacia arriba si es positivo, hacia abajo si es negativo.\n"
          "La lista de la «e» es la misma en los dos sitios; lo que entra, no.")
    L.guardar(ruta)
    return n_barras


def selftest():
    fallos = []
    bloques = leer(AQUI / SALIDA)
    import os, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("nada\n")
    try:
        leer(fh.name)
        fallos.append("test nulo: leyó listas de un texto que no las tiene")
    except (AssertionError, AttributeError):
        pass
    finally:
        os.unlink(fh.name)
    print("[1] test nulo         un texto sin el apartado 3 no da figura")
    # 2. SEÑAL — la lista de la «e» es la misma en los dos sitios y lo que entra es distinto.
    misma = bloques[0][0][1] == bloques[1][0][1]
    distinto = bloques[0][2][1] != bloques[1][2][1]
    print(f"[2] señal             misma lista: {'sí' if misma else 'NO'}; lo que entra, distinto: "
          f"{'sí' if distinto else 'NO'}")
    if not (misma and distinto):
        fallos.append("señal: la lista no es la misma, o lo que entra no cambia con el sitio")
    # 3. INVARIANTE — cada «lo que entra» es la suma de las dos filas de encima (lo comprueba leer),
    #    y se dibujan las 36 barras en las dos paletas.
    for pal in (COLOR, GRIS):
        n = dibujar(bloques, pal, "/dev/null")
    print(f"[3] invariante        las sumas cuadran; barras dibujadas: {n}")
    if n != 36:
        fallos.append("invariante: no se dibujan las 36 barras")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    dibujar(leer(AQUI / SALIDA), GRIS, str(AQUI / DESTINO))
    print(f"escrita {DESTINO}")


if __name__ == "__main__":
    main()
