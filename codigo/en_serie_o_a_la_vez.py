#!/usr/bin/env python3
"""
Capítulo 8 — por qué lo decisivo fue quitar la lectura en orden.

Compara dos maneras de procesar una frase con el mismo tamaño de modelo:
  - EN SERIE: una red recurrente, que no puede empezar la palabra N sin haber terminado la
    N-1. Por muchos procesadores que haya, esperan.
  - A LA VEZ: un bloque de atención, donde todas las posiciones se calculan a la vez.

Mide el tiempo de una pasada de entrenamiento (adelante y atrás) para longitudes crecientes,
y cuenta los PASOS EN SERIE, que no dependen de la máquina.

Uso:
    python en_serie_o_a_la_vez.py
    python en_serie_o_a_la_vez.py --selftest
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260914
ANCHO = 256                # números por posición, igual en las dos máquinas
CABEZAS = 8
LOTE = 16
LONGITUDES = [64, 128, 256, 512, 1024]
REPETICIONES = 3
CALENTAMIENTO = 1

SALIDA_CSV = "en_serie_o_a_la_vez.csv"

# ==========================================================

import argparse
import csv
import os
import sys
import time

from formato import miles, coma
import numpy as np
import torch
import torch.nn as nn


def dispositivo():
    """El procesador que se usa. Con la variable de entorno FORZAR_CPU=1 se obliga a usar
    el procesador normal aunque haya tarjeta gráfica.

    Hace falta porque el capítulo compara las DOS cosas en la MISMA máquina: primero en el
    procesador y luego en la tarjeta. Si la medición del procesador se hace en otro
    ordenador, la frase «el mismo programa en el mismo portátil» deja de ser verdad."""
    if os.environ.get("FORZAR_CPU") == "1":
        return torch.device("cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DISPOSITIVO = dispositivo()


class EnSerie(nn.Module):
    """Recurrente: la posición N necesita el resultado de la N-1. PASOS EN SERIE = longitud."""

    def __init__(self, ancho=ANCHO):
        super().__init__()
        self.nucleo = nn.LSTM(ancho, ancho, batch_first=True)
        self.salida = nn.Linear(ancho, ancho)

    def forward(self, x):
        h, _ = self.nucleo(x)
        return self.salida(h)

    @staticmethod
    def pasos_en_serie(longitud):
        return longitud


class ALaVez(nn.Module):
    """Atención: todas las posiciones se comparan a la vez. PASOS EN SERIE = 1.

    Lleva la red densa que acompaña siempre a la atención en un bloque real. Sin ella, esta
    máquina tiene la MITAD de números ajustables que la recurrente y la comparación sale
    sesgada a su favor: el test nulo lo detectó (4,28x de ventaja con una sola posición,
    donde no hay nada que paralelizar). Con la red densa, los tamaños quedan parejos."""

    def __init__(self, ancho=ANCHO, cabezas=CABEZAS):
        super().__init__()
        self.atencion = nn.MultiheadAttention(ancho, cabezas, batch_first=True)
        self.densa = nn.Sequential(nn.Linear(ancho, 2 * ancho), nn.GELU(),
                                   nn.Linear(2 * ancho, ancho))
        self.salida = nn.Linear(ancho, ancho)

    def forward(self, x):
        h, _ = self.atencion(x, x, x, need_weights=False)
        return self.salida(self.densa(h))

    @staticmethod
    def pasos_en_serie(longitud):
        return 1


def cronometrar(modelo, longitud, repeticiones=REPETICIONES):
    torch.manual_seed(SEMILLA)
    x = torch.randn(LOTE, longitud, ANCHO, device=DISPOSITIVO)
    objetivo = torch.randn(LOTE, longitud, ANCHO, device=DISPOSITIVO)
    perdida = nn.MSELoss()

    def una_pasada():
        modelo.zero_grad()
        perdida(modelo(x), objetivo).backward()
        if DISPOSITIVO.type == "mps":
            torch.mps.synchronize()
        elif DISPOSITIVO.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(CALENTAMIENTO):
        una_pasada()
    t = time.perf_counter()
    for _ in range(repeticiones):
        una_pasada()
    return (time.perf_counter() - t) / repeticiones


def selftest():
    fallos = []
    serie = EnSerie().to(DISPOSITIVO)
    a_la_vez = ALaVez().to(DISPOSITIVO)

    # 1. TEST NULO — con UNA SOLA posición no hay nada que paralelizar, así que cualquier
    #    diferencia de tiempo ahí es sobrecarga fija del montaje, no paralelismo. La prueba
    #    es: si a longitud larga la máquina «a la vez» sale ganando, esa ventaja tiene que
    #    ser MAYOR que la sobrecarga medida con una sola posición. Si no, lo que se está
    #    midiendo es el montaje.
    t_s1 = cronometrar(serie, 1, 5)
    t_v1 = cronometrar(a_la_vez, 1, 5)
    ventaja1 = t_s1 / t_v1
    t_sL = cronometrar(serie, 512, 2)
    t_vL = cronometrar(a_la_vez, 512, 2)
    ventajaL = t_sL / t_vL
    print(f"[1] test nulo         ventaja con 1 posición (pura sobrecarga) {ventaja1:.2f}x; "
          f"con 512 posiciones {ventajaL:.2f}x")
    if ventajaL > 1.0 and ventajaL <= ventaja1:
        fallos.append(
            f"test nulo: la ventaja a longitud 512 ({ventajaL:.2f}x) no supera la sobrecarga "
            f"fija medida con una sola posición ({ventaja1:.2f}x); el resultado lo explica el "
            "montaje, no el paralelismo")

    # 2. SEÑAL IMPLANTADA — la máquina en serie tiene que tardar aproximadamente el doble al
    #    doblar la longitud: su trabajo crece con el número de pasos.
    t_a = cronometrar(serie, 128, 3)
    t_b = cronometrar(serie, 256, 3)
    razon = t_b / t_a
    print(f"[2] señal implantada  al doblar la longitud, la de en serie tarda {razon:.2f}x "
          f"(se espera cerca de 2)")
    if not 1.3 <= razon <= 3.2:
        fallos.append(f"señal implantada: al doblar la longitud la razón es {razon:.2f}, "
                      "fuera del intervalo esperado de 1,3 a 3,2")

    # 3. INVARIANTE DEL DOMINIO — el recuento de pasos en serie no depende de la máquina.
    ok = all(EnSerie.pasos_en_serie(n) == n and ALaVez.pasos_en_serie(n) == 1
             for n in LONGITUDES)
    print(f"[3] invariante        pasos en serie: en serie = longitud, a la vez = 1: "
          f"{'sí' if ok else 'NO'}")
    if not ok:
        fallos.append("invariante: el recuento de pasos en serie no es el esperado")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    global ANCHO, LOTE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    # El tamaño de la máquina de juguete se puede cambiar SIN tocar las constantes: las
    # de arriba siguen siendo las de la medición del libro, y una ejecución con otro
    # tamaño lo dice en su propia cabecera. Hace falta porque la pregunta abierta es si
    # la ventaja de la tarjeta gráfica depende del tamaño de lo que se le manda hacer.
    ap.add_argument("--ancho", type=int, default=ANCHO,
                    help=f"números por posición (por omisión {ANCHO})")
    ap.add_argument("--lote", type=int, default=LOTE,
                    help=f"cuántos textos a la vez (por omisión {LOTE})")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    ANCHO, LOTE = args.ancho, args.lote
    assert ANCHO % CABEZAS == 0, \
        f"Se esperaba un ancho múltiplo de {CABEZAS} cabezas; se pidió {ANCHO}"
    assert LOTE >= 1, f"Se esperaba un lote de 1 o más; se pidió {LOTE}"

    print(f"Procesador usado: {DISPOSITIVO}")
    serie = EnSerie(ancho=ANCHO).to(DISPOSITIVO)
    a_la_vez = ALaVez(ancho=ANCHO, cabezas=CABEZAS).to(DISPOSITIVO)
    n_serie = sum(p.numel() for p in serie.parameters())
    n_vez = sum(p.numel() for p in a_la_vez.parameters())
    print(f"{ANCHO} números por posición, lotes de {LOTE}.")
    print(f"Números ajustables: en serie {miles(n_serie)} | a la vez {miles(n_vez)} "
          f"({n_vez/n_serie:.2f} veces).\n")
    filas = []

    print("--- PASOS QUE HAY QUE DAR UNO DETRÁS DE OTRO (no depende de la máquina) ---")
    # Los números van con el separador de miles del castellano, y TODAS las columnas
    # igual: antes la longitud salía «1024» y los pasos «1,024» en la misma fila.
    print(f"{'longitud':>10}{'en serie':>12}{'a la vez':>12}")
    for n in LONGITUDES:
        print(f"{miles(n):>10}{miles(EnSerie.pasos_en_serie(n)):>12}"
              f"{miles(ALaVez.pasos_en_serie(n)):>12}")

    print("\n--- TIEMPO DE UNA PASADA DE ENTRENAMIENTO ---")
    print(f"{'longitud':>10}{'en serie (s)':>15}{'a la vez (s)':>15}{'ventaja':>10}")
    for n in LONGITUDES:
        t_s = cronometrar(serie, n)
        t_v = cronometrar(a_la_vez, n)
        print(f"{miles(n):>10}{coma(t_s, 4):>15}{coma(t_v, 4):>15}"
              f"{coma(t_s / t_v, 1):>9}x")
        filas.append([str(DISPOSITIVO), n, f"{t_s:.6f}", f"{t_v:.6f}", f"{t_s/t_v:.2f}"])

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(
            [["procesador", "longitud", "en_serie_s", "a_la_vez_s", "ventaja"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
