#!/usr/bin/env python3
"""
Capítulo 1 — lo poco que hay dentro de las casillas llenas, y lo que costaría llenarlas.

La figura anterior enseña cuántas casillas están vacías. Ésta enseña las dos cosas que el
capítulo dice a continuación y que también son números grandes que se leen y se pasan de largo:
que de las pocas casillas con algo dentro, casi cuatro de cada cinco tienen una sola pareja; y
que mirar una palabra más hacia atrás multiplica por treinta y seis mil el texto que haría falta.

Todos los números salen de `datos/salidas/casillas_vacias.txt`. Los dos que no están escritos
allí —las casillas vistas más de una vez y el cociente entre los dos montones de Quijotes— los
calcula esta figura a partir de los que sí están, y el pie enseña las dos cuentas para que el
lector pueda rehacerlas.

Uso:
    python figura_lo_poco_que_hay.py --selftest
    python figura_lo_poco_que_hay.py
"""

# ======================= CONSTANTES =======================

SALIDA = "../datos/salidas/casillas_vacias.txt"
DESTINO = "../figuras/lo_poco_que_hay.png"

# ==========================================================

import argparse, re, sys
from infografia import Lienzo, GRIS, COLOR, TRAMAS


def numero(s):
    return float(s.replace(".", "").replace(",", ".")) if "," in s else int(s.replace(".", ""))


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def buscar(texto, patron, que):
    m = re.search(patron, texto)
    assert m, f"no encuentro {que} en la salida"
    return m.group(1)


def leer(ruta):
    t = open(ruta, encoding="utf-8").read()
    corte = t.find("SI SE MIRARAN TRES PALABRAS HACIA ATRÁS")
    assert corte > 0, "no encuentro la sección de las tres palabras en la salida"
    dos, tres = t[:corte], t[corte:]
    d = {
        "ocupadas":   buscar(dos, r"casillas con algo dentro \(parejas vistas\)\s+([\d.]+)",
                             "las casillas ocupadas"),
        "una_vez":    buscar(dos, r"parejas vistas una sola vez en todo el libro\s+([\d.]+)",
                             "las parejas vistas una sola vez"),
        "pc_una_vez": buscar(dos, r"las de una sola vez\s+([\d,]+) %",
                             "el porcentaje de las de una sola vez"),
        "quijotes2":  buscar(dos, r"palabra en cada casilla\s+([\d.]+)",
                             "los Quijotes de dos palabras"),
        "quijotes3":  buscar(tres, r"palabra en cada casilla\s+([\d.]+)",
                             "los Quijotes de tres palabras"),
    }
    d["repetidas"] = numero(d["ocupadas"]) - numero(d["una_vez"])
    d["pc_repetidas"] = 100 * d["repetidas"] / numero(d["ocupadas"])
    d["veces_mas"] = numero(d["quijotes3"]) / numero(d["quijotes2"])
    return d


def dibujar(d, paleta, ruta):
    from matplotlib.patches import Rectangle
    p = paleta
    L = Lienzo("Y lo poco que hay, casi no se repite",
               "De las casillas que sí tienen algo dentro, la mayoría lo tiene una sola vez\n"
               "en todo el libro. Y con una palabra más de memoria, la tabla se dispara.",
               paleta, alto=4.35)

    # --- 1. de las llenas, casi todas con una sola pareja ------------------------------------
    x, y, w = L.panel(1, "Vistas una vez, y nunca más", 31.0)
    L.texto(x, y + 0.4, f"Las {d['ocupadas']} casillas con algo dentro, repartidas según cuántas "
                        f"veces", tam=7.2, color=p.suave)
    L.texto(x, y - 2.8, "aparece esa pareja de palabras en el libro entero:", tam=7.2,
            color=p.suave)

    frac = numero(d["una_vez"]) / numero(d["ocupadas"])
    yb, hb = y - 15.6, 7.0
    L.ax.add_patch(Rectangle((x, yb), w * frac, hb, facecolor=p.acento, edgecolor=p.tinta,
                             linewidth=0.7))
    L.ax.add_patch(Rectangle((x + w * frac, yb), w * (1 - frac), hb, facecolor=p.papel,
                             hatch=TRAMAS["rayas"][0], edgecolor=p.tinta, linewidth=0.7))
    L.texto(x + w * frac / 2, yb + hb / 2, f"{d['pc_una_vez']} %", tam=9, color="white",
            ha="center", negrita=True)
    # El 21,2 % va encima de su trozo: escrito dentro, las rayas se lo comen.
    L.texto(x + w, yb + hb + 2.6, f"{d['pc_repetidas']:.1f} %".replace(".", ","), tam=8,
            ha="right", negrita=True)
    L.texto(x, yb - 4.2, f"{d['una_vez']} vistas una sola vez", tam=7.2, color=p.suave)
    L.texto(x + w, yb - 4.2, f"{miles(d['repetidas'])} vistas más de una vez", tam=7.2,
            color=p.suave, ha="right")

    # --- 2. y esto mirando solo dos palabras --------------------------------------------------
    x, y, w = L.panel(2, "Y esto mirando solo dos palabras", 34.0)
    L.texto(x, y + 0.4, "Cuántos Quijotes de texto harían falta para escribir una sola palabra",
            tam=7.2, color=p.suave)
    L.texto(x, y - 2.8, "en cada casilla de la tabla:", tam=7.2, color=p.suave)
    L.ficha(x, y - 9.6, f"mirando 2 palabras atrás:  {d['quijotes2']} Quijotes", w, alto=5.6,
            relleno=p.fondo, tam=8.0)
    L.ficha(x, y - 17.0, f"mirando 3 palabras atrás:  {d['quijotes3']} Quijotes", w, alto=5.6,
            relleno=p.acento, tinta="white", negrita=True, tam=8.0)
    L.texto(x + w / 2, y - 23.4,
            f"una palabra más de memoria cuesta {miles(round(d['veces_mas']))} veces más texto",
            tam=7.4, ha="center", color=p.suave)

    L.pie("Los números salen de datos/salidas/casillas_vacias.txt, medidos sobre el Quijote. Los\n"
          f"dos calculados aquí: {miles(d['repetidas'])} = {d['ocupadas']} − {d['una_vez']}, "
          f"y {miles(round(d['veces_mas']))} = {d['quijotes3']} / {d['quijotes2']}.")
    L.guardar(ruta)
    return ruta


def selftest():
    fallos, d = [], leer(SALIDA)
    crudo = open(SALIDA, encoding="utf-8").read()

    # 1. TEST NULO — sin la sección de las tres palabras, «palabra en cada casilla» solo casa una
    #    vez y la figura pintaría el número de dos palabras en las dos fichas: el panel diría que
    #    mirar una palabra más no cuesta nada, que es lo contrario de lo que pasa.
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(crudo[:crudo.find("SI SE MIRARAN TRES PALABRAS HACIA ATRÁS")]); tmp.close()
    try:
        leer(tmp.name); revienta = False
    except AssertionError:
        revienta = True
    os.unlink(tmp.name)
    print(f"[1] test nulo         sin la sección de tres palabras, "
          f"{'revienta' if revienta else 'NO revienta'}")
    if not revienta:
        fallos.append("test nulo: leer() no revienta sin la sección de tres palabras")

    # 2. SEÑAL IMPLANTADA — cada número leído está, escrito igual, en la salida.
    leidos = [d[k] for k in ("ocupadas", "una_vez", "pc_una_vez", "quijotes2", "quijotes3")]
    fuera = [v for v in leidos if v not in crudo]
    print(f"[2] señal implantada  {len(leidos) - len(fuera)} de {len(leidos)} números leídos "
          f"aparecen literales en la salida")
    if fuera:
        fallos.append(f"señal implantada: estos números no están literales: {fuera}")

    # 3. INVARIANTE DEL DOMINIO — los dos montones suman el total, los dos porcentajes suman cien
    #    y mirar tres palabras cuesta más que mirar dos. Si alguna de las tres fallara, la figura
    #    estaría diciendo algo que no puede ser.
    suma = d["repetidas"] + numero(d["una_vez"]) == numero(d["ocupadas"])
    cien = abs(numero(d["pc_una_vez"]) + d["pc_repetidas"] - 100) < 0.1
    crece = numero(d["quijotes3"]) > numero(d["quijotes2"])
    print(f"[3] invariante        {miles(d['repetidas'])} + {d['una_vez']} = {d['ocupadas']}: "
          f"{'sí' if suma else 'NO'}; los dos porcentajes suman cien: {'sí' if cien else 'NO'}; "
          f"tres palabras cuestan más que dos: {'sí' if crece else 'NO'}")
    if not suma:
        fallos.append("invariante: los dos montones no suman el total")
    if not cien:
        fallos.append("invariante: los dos porcentajes no suman cien")
    if not crece:
        fallos.append("invariante: tres palabras salen más baratas que dos")

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
    print("Escrito", dibujar(leer(SALIDA), GRIS, DESTINO))


if __name__ == "__main__":
    main()
