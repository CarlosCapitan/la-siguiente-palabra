#!/usr/bin/env python3
"""
Capítulo 8 — la papelera por dentro: ¿qué aporta a la mezcla el contenido de «El»? (L24)

El capítulo dice que más de la mitad del reparto del adjetivo va a la primera palabra, «El», y que
esa palabra «no aporta nada»: es una papelera. Pero si la mezcla se lleva el contenido de cada
palabra en proporción al reparto, más de la mitad de lo que recibe el adjetivo debería ser el
contenido de «El». Las dos cosas solo casan si el contenido de «El» es pequeño. Este programa lo
mide en el mismo modelo, con las mismas frases y la misma palabra que pregunta que
`reparto_una_a_una.py`:

  - para cada una de las 336 miradas, el contenido de cada palabra (los 64 números que entrega) y
    su porción del reparto;
  - el «tamaño» de un contenido: de media, cuánto se aleja de cero cada uno de sus 64 números;
  - la parte de la mezcla que viene de cada palabra: el tamaño de su contenido multiplicado por su
    porción, entre lo mismo sumado para todas las palabras.

El contenido se saca de dentro del modelo con un gancho a la salida de la tabla que lo fabrica, y el
selftest comprueba que la mezcla rehecha con esos contenidos y ese reparto es la que el modelo le
pasa a la capa siguiente.

Uso:
    python la_papelera_por_dentro.py --selftest
    python la_papelera_por_dentro.py > ../datos/salidas/la_papelera_por_dentro.txt
"""

# ======================= CONSTANTES =======================

from reparto_atencion import MODELO, FRASES, FRACCION_CAPAS_FINALES, SEMILLA, cargar_modelo

TOL_MEZCLA = 1e-4        # diferencia máxima entre la mezcla rehecha y la del modelo
TOL_NULO = 0.10          # en el test nulo, la parte de la mezcla tiene que ser la porción ±10 puntos

# ==========================================================

import argparse
import random
import sys

import numpy as np
import torch

from formato import coma, comprobar_ancho

ANCHO = 64


def palabras_de(tok, frase):
    """Las subunidades del troceador, agrupadas en palabras como en reparto_una_a_una.py."""
    ids = tok(frase, return_tensors="pt")["input_ids"][0]
    piezas = [tok.decode([i]) for i in ids]
    palabras, de_palabra = [], []
    for pieza in piezas:
        if pieza.startswith(" ") or not palabras:
            palabras.append(pieza.strip())
        else:
            palabras[-1] += pieza
        de_palabra.append(len(palabras) - 1)
    return piezas, palabras, de_palabra


def por_dentro(tok, modelo, frase, cambiar_el=False, antes_del_punto=False):
    """Para cada capa y mirada: la porción de cada palabra y el contenido que entrega (sumando sus
    subunidades). Con cambiar_el=True, el contenido de «El» se sustituye por la media de los de las
    demás palabras de esa mirada (solo para el test nulo)."""
    cod = tok(frase, return_tensors="pt")
    piezas, palabras, de_palabra = palabras_de(tok, frase)
    q = max(k for k, w in enumerate(de_palabra) if w == len(palabras) - 1)
    if antes_del_punto:
        assert piezas[q].strip() == ".", f"se esperaba que la frase acabara en un punto; acaba en {piezas[q]!r}"
        q -= 1
    capas = modelo.model.layers
    cfg = modelo.config
    H, G = cfg.num_attention_heads, cfg.num_key_value_heads
    D = cfg.hidden_size // H
    valores, mezclas = {}, {}
    ganchos = []
    for c, capa in enumerate(capas):
        ganchos.append(capa.self_attn.v_proj.register_forward_hook(
            lambda m, i, o, c=c: valores.__setitem__(c, o[0].detach().double().numpy())))
        ganchos.append(capa.self_attn.o_proj.register_forward_pre_hook(
            lambda m, i, c=c: mezclas.__setitem__(c, i[0][0].detach().double().numpy())))
    with torch.no_grad():
        salida = modelo(**cod, output_attentions=True)
    for g in ganchos:
        g.remove()
    n = len(palabras)
    porcion = np.zeros((len(capas), H, n))
    aporte = np.zeros((len(capas), H, n, D))      # porción × contenido, por palabra
    contenido = np.zeros((len(capas), H, n, D))
    peor = 0.0
    for c in range(len(capas)):
        a = salida.attentions[c][0, :, q, :].double().numpy()      # (H, piezas)
        v = valores[c].reshape(len(piezas), G, D)
        for h in range(H):
            vh = v[:, h // (H // G), :].copy()
            if cambiar_el:
                otras = [k for k, w in enumerate(de_palabra) if w != 0]
                for k, w in enumerate(de_palabra):
                    if w == 0:
                        vh[k] = vh[otras].mean(0)
            rehecha = a[h] @ vh
            peor = max(peor, float(np.abs(rehecha - mezclas[c][q, h * D:(h + 1) * D]).max()))
            for k, w in enumerate(de_palabra):
                porcion[c, h, w] += a[h, k]
                aporte[c, h, w] += a[h, k] * vh[k]
                contenido[c, h, w] += vh[k] / de_palabra.count(w)
    return piezas, palabras, porcion, aporte, contenido, (peor if not cambiar_el else None)


def resumen(porcion, aporte, contenido, desde=0):
    """Promedios sobre las miradas de las capas desde `desde`: porción de «El» y de las demás,
    tamaño del contenido de «El» y de las demás, y parte de la mezcla que viene de «El»."""
    P, A, C = porcion[desde:], aporte[desde:], contenido[desde:]
    tam_aporte = np.abs(A).mean(-1)                       # (capas, H, palabras)
    parte = tam_aporte / tam_aporte.sum(-1, keepdims=True)
    tam_cont = np.abs(C).mean(-1)
    n = P.shape[-1]
    otras = list(range(1, n))
    return dict(porcion_el=P[..., 0].mean(), porcion_otras=P[..., otras].sum(-1).mean(),
                tam_el=tam_cont[..., 0].mean(), tam_otras=tam_cont[..., otras].mean(),
                parte_el=parte[..., 0].mean(), parte_otras=parte[..., otras].sum(-1).mean(),
                miradas=P.shape[0] * P.shape[1])


def selftest(tok, modelo):
    fallos = []
    frase = FRASES["alto"]
    # 1. TEST NULO — si el contenido de «El» fuera como el de las demás palabras, su parte de la
    #    mezcla tendría que parecerse a su porción del reparto: la medida no encuentra «pequeño» por
    #    construcción.
    _, _, P, A, C, _ = por_dentro(tok, modelo, frase, cambiar_el=True)
    r = resumen(P, A, C)
    dif = abs(r["parte_el"] - r["porcion_el"])
    print(f"[1] test nulo         con el contenido de «El» igual al de las demás: porción "
          f"{coma(100 * r['porcion_el'])} %, parte de la mezcla {coma(100 * r['parte_el'])} %")
    if dif > TOL_NULO:
        fallos.append("test nulo: con un contenido normal, la parte de la mezcla no se parece a la porción")
    # 2. SEÑAL — la mezcla rehecha con los contenidos del gancho y el reparto es la que el modelo le
    #    pasa a la tabla siguiente, en las 336 miradas.
    _, _, P, A, C, peor = por_dentro(tok, modelo, frase)
    print(f"[2] señal             mezcla rehecha contra la del modelo: diferencia {peor:.1e}")
    if peor > TOL_MEZCLA:
        fallos.append(f"señal: la mezcla rehecha no es la del modelo ({peor:.1e})")
    # 3. INVARIANTE — en cada mirada las porciones suman uno y las partes de la mezcla también.
    suma = float(np.abs(P.sum(-1) - 1).max())
    t = np.abs(A).mean(-1)
    suma_p = float(np.abs((t / t.sum(-1, keepdims=True)).sum(-1) - 1).max())
    print(f"[3] invariante        porciones y partes suman uno: error {suma:.1e} y {suma_p:.1e}")
    if suma > 1e-4 or suma_p > 1e-9:
        fallos.append("invariante: un reparto no suma uno")
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
    random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
    tok, modelo = cargar_modelo(MODELO)
    print("--- selftest ---")
    codigo = selftest(tok, modelo)
    if codigo or args.selftest:
        return codigo
    capas = modelo.config.num_hidden_layers
    desde = capas - max(1, int(round(capas * FRACCION_CAPAS_FINALES)))
    L = ["", "####### capítulo 8: la papelera por dentro #######",
         f"modelo: {MODELO}, el mismo de reparto_una_a_una.py", ""]
    piezas, palabras, *_ = por_dentro(tok, modelo, FRASES["alto"])
    import textwrap
    L += ["La frase, en los trozos en que la parte el modelo:"]
    L += ["  " + l for l in textwrap.wrap(" | ".join(p.strip() or "_" for p in piezas), 58)]
    L += [
          "Pregunta el último trozo, como en reparto_una_a_una.py.", ""]
    L += ["1. LO QUE PONE «El» EN LA MEZCLA", "",
          "  «porción»: la parte del reparto que se lleva, de cada cien.",
          "  «tamaño»: de media, cuánto se aleja de cero cada uno de los",
          "  64 números de su contenido.",
          "  «parte de la mezcla»: su contenido por su porción, frente a lo",
          "  mismo de todas las palabras juntas, de cada cien.", ""]
    for adj, frase in FRASES.items():
        _, _, P, A, C, _ = por_dentro(tok, modelo, frase)
        for nombre, d in (("todas", 0), ("últimas", desde)):
            r = resumen(P, A, C, d)
            L += [f"  «{adj}», {nombre} ({r['miradas']} miradas)",
                  f"    {'':<22}{'porción':>10}{'tamaño':>10}{'parte de':>12}",
                  f"    {'':<22}{'':>10}{'':>10}{'la mezcla':>12}",
                  f"    {'«El»':<22}{coma(100 * r['porcion_el']) + ' %':>10}{coma(r['tam_el'], 3):>10}"
                  f"{coma(100 * r['parte_el']) + ' %':>12}",
                  f"    {'las demás palabras':<22}{coma(100 * r['porcion_otras']) + ' %':>10}"
                  f"{coma(r['tam_otras'], 3):>10}{coma(100 * r['parte_otras']) + ' %':>12}", ""]
    L += ["  «todas»: las 336 miradas; «últimas»: las de las 8 últimas",
          "  capas. El tamaño de «las demás» es la media de una palabra;",
          "  su porción y su parte, la suma de todas.", ""]
    L += ["2. EL REPARTO DEL FINAL DE LA FRASE, PALABRA A PALABRA",
          "   (lo mismo que reparto_una_a_una.py, con la última columna",
          "   rotulada por lo que es)", "",
          f"  {'frase':<8}{'miradas':<9}{'«El»':>10}{'«vaso»':>10}{'«cajón»':>10}{'el final':>10}",
          f"  {'-----':<8}{'-------':<9}{'----':>10}{'------':>10}{'-------':>10}{'--------':>10}"]
    for adj, frase in FRASES.items():
        _, pal, P, A, C, _ = por_dentro(tok, modelo, frase)
        iv = [i for i, w in enumerate(pal) if w.strip(".,").lower() == "vaso"][0]
        ic = [i for i, w in enumerate(pal) if w.strip(".,").lower() == "cajón"][0]
        for nombre, d in (("todas", 0), ("últimas", desde)):
            Q = P[d:]
            L.append(f"  {adj:<8}{nombre:<9}" + "".join(f"{coma(100 * Q[..., i].mean()) + ' %':>10}"
                                                      for i in (0, iv, ic, len(pal) - 1)))
    L += ["", "  Lo que va a los trozos de una palabra se suma: «v» y «aso»",
          "  cuentan como «vaso». «el final»: los dos últimos trozos",
          "  juntos, el adjetivo y el punto: lo que se queda el final",
          "  de la frase. «El» es la primera palabra: la",
          "  papelera. «todas»: las 336 miradas; «últimas»: las 112 de",
          "  las 8 últimas capas.", ""]
    L += ["3. SI PREGUNTA EL ADJETIVO Y NO EL PUNTO DE DETRÁS",
          "   (reparto_una_a_una.py pregunta desde el último trozo de la",
          "   frase, que es el punto; aquí, desde el trozo de antes)", "",
          f"  {'adjetivo':<10}{'pregunta':<12}{'«El»':>10}{'«vaso»':>10}{'«cajón»':>10}",
          f"  {'--------':<10}{'--------':<12}{'----':>10}{'------':>10}{'-------':>10}"]
    for adj, frase in FRASES.items():
        for nombre, antes in (("el punto", False), ("el adjetivo", True)):
            pz, pal, P, A, C, _ = por_dentro(tok, modelo, frase, antes_del_punto=antes)
            iv = [i for i, w in enumerate(pal) if w.strip(".,").lower() == "vaso"][0]
            ic = [i for i, w in enumerate(pal) if w.strip(".,").lower() == "cajón"][0]
            L.append(f"  {adj:<10}{nombre:<12}{coma(100 * P[..., 0].mean()) + ' %':>10}"
                     f"{coma(100 * P[..., iv].mean()) + ' %':>10}{coma(100 * P[..., ic].mean()) + ' %':>10}")
    L += ["", "  Media de las 336 miradas, de cada cien.", ""]
    for l in comprobar_ancho([l.rstrip() for l in L], ANCHO):
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
