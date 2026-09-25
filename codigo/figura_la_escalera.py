#!/usr/bin/env python3
"""
Capítulo 1 — la escalera, presentada antes de enseñarla.

La sección «La escalera» del capítulo suelta el primer bloque de texto generado tras una sola
línea de presentación, y el lector no sabe qué está mirando hasta varios párrafos después. Esta
figura es la presentación: el mismo texto, la misma máquina, una sola perilla, y lo que escupe
en cada posición.

Las muestras no se escriben aquí: se leen de `datos/salidas/ngrama.txt`, que es lo que imprimió
la máquina. Si esa salida cambia, la figura cambia.

Uso:
    python figura_la_escalera.py --selftest
    python figura_la_escalera.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/ngrama.txt"
DESTINO = "../figuras/la_escalera.png"

# Los peldaños que se enseñan, con el rótulo de la perilla. Son los de letras: el capítulo pasa
# a palabras después, y mezclarlos aquí sería enseñar dos experimentos como si fueran uno.
PELDANOS = [("azar puro", "nada"), ("1 letra de contexto", "1 letra"),
            ("2 letras de contexto", "2 letras"), ("3 letras de contexto", "3 letras"),
            ("5 letras de contexto", "5 letras")]
LETRAS_MUESTRA = 150        # cuánto de cada muestra cabe en la caja sin apretarla

# ==========================================================

import argparse, re, sys, textwrap
from infografia import Lienzo, GRIS, COLOR


def leer(ruta):
    t = open(ruta, encoding="utf-8").read()
    m = re.match(r"Corpus: ([\d.]+) caracteres, ([\d.]+) palabras y ([\d.]+) signos; ([\d.]+) palabras distintas",
                 t)
    assert m, "no encuentro la línea del corpus en la salida"
    d = {"caracteres": m.group(1), "palabras": m.group(2), "distintas": m.group(4), "muestras": {}}
    for rotulo, _ in PELDANOS:
        m = re.search(r"--- LETRAS, " + re.escape(rotulo) + r" ---\n(.+)", t)
        assert m, f"no encuentro la muestra «{rotulo}» en la salida"
        d["muestras"][rotulo] = m.group(1).strip()
    return d


def dibujar(d, paleta, ruta):
    L = Lienzo("La escalera: una sola perilla",
               "Una máquina que cuenta cuántas veces va cada letra detrás de cada trozo, y luego\n"
               "escribe echando a suertes con esos recuentos. Lo único que se le cambia es cuántas\n"
               "letras mira hacia atrás. El texto contado es siempre el mismo.",
               paleta, alto=6.24)
    p = paleta
    L.texto(4, L.y - 1.0, f"El Quijote entero: {d['caracteres']} letras, {d['palabras']} palabras, "
                          f"{d['distintas']} distintas.", tam=7.2, color=p.suave)
    L.y -= 5.0

    for n, (rotulo, perilla) in enumerate(PELDANOS, start=1):
        x, y, w = L.panel(n, f"La perilla en «{perilla}»", 16.6)
        texto = d["muestras"][rotulo][:LETRAS_MUESTRA]
        for i, linea in enumerate(textwrap.wrap(texto, 62)[:3]):
            L.texto(x, y - i * 3.2, linea, tam=6.6, mono=True)
        if n < len(PELDANOS):
            L.flecha()

    L.pie("Las cinco muestras salen de datos/salidas/ngrama.txt, la salida del programa del\n"
          "capítulo. Están recortadas para que quepan; no se ha cambiado ninguna letra.")
    L.guardar(ruta)
    return ruta


def selftest():
    fallos, d = [], leer(SALIDA)
    # 1. TEST NULO — una salida sin las marcas de peldaño tiene que reventar, no dibujar una
    #    escalera con menos escalones y sin avisar.
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write("Corpus: 1 caracteres, 1 palabras y 1 signos; 1 palabras distintas.\n"); tmp.close()
    try:
        leer(tmp.name); revienta = False
    except AssertionError:
        revienta = True
    os.unlink(tmp.name)
    print(f"[1] test nulo         con una salida sin peldaños, "
          f"{'revienta' if revienta else 'NO revienta'}")
    if not revienta:
        fallos.append("test nulo: leer() no revienta con una salida sin peldaños")

    # 2. SEÑAL IMPLANTADA — cada muestra que se dibuja está, carácter a carácter, en la salida.
    crudo = open(SALIDA, encoding="utf-8").read()
    fuera = [r for r in d["muestras"] if d["muestras"][r][:LETRAS_MUESTRA] not in crudo]
    print(f"[2] señal implantada  {len(d['muestras']) - len(fuera)} de {len(d['muestras'])} "
          f"muestras aparecen literales en la salida")
    if fuera:
        fallos.append(f"señal implantada: estas muestras no están literales: {fuera}")

    # 3. INVARIANTE DEL DOMINIO — las cinco muestras son distintas entre sí. Si dos salieran
    #    iguales, la perilla no estaría haciendo nada y la figura estaría mintiendo.
    distintas = len({v[:LETRAS_MUESTRA] for v in d["muestras"].values()})
    print(f"[3] invariante        {distintas} muestras distintas de {len(PELDANOS)} peldaños")
    if distintas != len(PELDANOS):
        fallos.append(f"invariante: solo {distintas} muestras distintas")

    print()
    if fallos:
        for f in fallos: print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    d = leer(SALIDA)
    print("Escrito", dibujar(d, GRIS, DESTINO))


if __name__ == "__main__":
    main()
