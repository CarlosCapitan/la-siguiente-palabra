#!/usr/bin/env python3
"""
Capítulo 6 — la memoria que se olvidaba.

Dos mediciones:
  1. Recordar a distancia: una red recurrente tiene que devolver un símbolo que vio N pasos
     antes. Se mide el acierto según la distancia, con memoria simple y con memoria con
     compuertas.
  2. Escribir en español: una red recurrente entrenada sobre texto real, para comparar con
     el modelo de contar del capítulo 1.

Uso:
    python memoria_recurrente.py
    python memoria_recurrente.py --selftest
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260914

# --- 1. recordar a distancia ---
DISTANCIAS = [10, 20, 40, 80, 160]
SIMBOLOS = 8                  # símbolos distintos que hay que recordar
RELLENO = 4                   # símbolos de paja que se intercalan
OCULTO_MEMORIA = 64
PASOS_MEMORIA = 4000        # con 1500 pasos el resultado lo decidia la suerte del
TASA_MEMORIA = 3e-3         # entrenamiento y no la memoria: salia no monotono con la
SEMILLAS_MEMORIA = 2        # distancia. Mas pasos, tasa menor y varias semillas.
LOTE_MEMORIA = 128
EJEMPLOS_PRUEBA = 2000

# --- 2. escribir en español ---
CORPUS = "../datos/corpus_es"
MAX_CARACTERES = 4_000_000
LONGITUD = 40                 # PALABRAS de contexto por ejemplo. Se trabaja con palabras,
VOCABULARIO_MAXIMO = 20_000   # no con letras, porque es lo comparable con el capitulo 1.
OCULTO_TEXTO = 512
CAPAS_TEXTO = 2
LOTE_TEXTO = 64
TASA_TEXTO = 2e-3
MINUTOS_MAXIMO = 9.0          # presupuesto de entrenamiento; el guion informa de lo que hizo
LARGO_MUESTRA = 45
TEMPERATURA = 0.8

UMBRAL_SELFTEST = 0.90

# ==========================================================

import argparse
import glob
import math
import os
import re
import sys
import time
import unicodedata

import numpy as np
import torch
import torch.nn as nn

from formato import coma, comprobar_ancho, miles


def dispositivo():
    """Usa la GPU del Mac (Metal) si está disponible; si no, el procesador. Los TIEMPOS cambian
    mucho, y por eso el libro solo cita tiempos medidos aquí. Los aciertos no: con la misma
    semilla salen iguales en el Mac y en un contenedor Linux, comprobado. Lo único que baila es
    la última cifra de la suma del invariante, que es ruido de coma flotante."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DISPOSITIVO = dispositivo()


def fijar_semilla(s=SEMILLA):
    torch.manual_seed(s)
    np.random.seed(s)


# --------------------------- 1. recordar a distancia ---------------------------

class Recordadora(nn.Module):
    """Lee una secuencia y, al final, tiene que decir cuál era el primer símbolo."""

    def __init__(self, vocabulario, oculto, con_compuertas):
        super().__init__()
        self.emb = nn.Embedding(vocabulario, 32)
        self.nucleo = (nn.LSTM if con_compuertas else nn.RNN)(32, oculto, batch_first=True)
        if con_compuertas:
            # La compuerta de olvido arranca abierta (sesgo a 1). Es practica estandar desde
            # Gers et al. (2000) y NO es afinar hasta que salga lo que uno quiere: sin esto
            # la red empieza olvidandolo todo y no llega nunca a aprender a recordar. Medido:
            # a distancia 20 pasa de 0,142 (azar) a 1,000 solo con este cambio.
            for nombre, par in self.nucleo.named_parameters():
                if "bias" in nombre:
                    n = par.shape[0] // 4
                    par.data[n:2 * n].fill_(1.0)
        self.salida = nn.Linear(oculto, SIMBOLOS)

    def forward(self, x):
        h, _ = self.nucleo(self.emb(x))
        return self.salida(h[:, -1, :])


def lote_memoria(distancia, n, rng):
    """[símbolo a recordar] + paja de longitud `distancia` -> hay que devolver el símbolo."""
    objetivo = rng.integers(0, SIMBOLOS, size=n)
    paja = rng.integers(SIMBOLOS, SIMBOLOS + RELLENO, size=(n, distancia))
    x = np.concatenate([objetivo[:, None], paja], axis=1)
    return torch.tensor(x, dtype=torch.long), torch.tensor(objetivo, dtype=torch.long)


def medir_memoria(distancia, con_compuertas, semilla=SEMILLA, pasos=PASOS_MEMORIA):
    fijar_semilla(semilla)
    rng = np.random.default_rng(semilla)
    modelo = Recordadora(SIMBOLOS + RELLENO, OCULTO_MEMORIA, con_compuertas)
    opt = torch.optim.Adam(modelo.parameters(), lr=TASA_MEMORIA)
    perdida = nn.CrossEntropyLoss()
    modelo.train()
    for _ in range(pasos):
        x, y = lote_memoria(distancia, LOTE_MEMORIA, rng)
        opt.zero_grad()
        l = perdida(modelo(x), y)
        l.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
    modelo.eval()
    with torch.no_grad():
        x, y = lote_memoria(distancia, EJEMPLOS_PRUEBA, rng)
        acierto = (modelo(x).argmax(1) == y).float().mean().item()
    return acierto


def medir_memoria_repetida(distancia, con_compuertas):
    """Varias semillas: una sola es ruido de entrenamiento, no una medida de memoria."""
    v = [medir_memoria(distancia, con_compuertas, SEMILLA + k) for k in range(SEMILLAS_MEMORIA)]
    return float(np.mean(v)), float(np.min(v)), float(np.max(v))


# --------------------------- 2. escribir en español ---------------------------

ALFABETO = "abcdefghijklmnñopqrstuvwxyzáéíóúü ,.;:¿?¡!\n"


def cargar_texto():
    ficheros = sorted(glob.glob(os.path.join(CORPUS, "*.txt")))
    assert ficheros, f"Se esperaban libros en «{CORPUS}»; no se encontró ninguno"
    trozos, total = [], 0
    for ruta in ficheros:
        with open(ruta, encoding="utf-8", errors="replace") as fh:
            t = fh.read()
        if "*** START OF" in t:
            t = t.split("*** START OF", 1)[1].split("\n", 1)[-1]
        if "*** END OF" in t:
            t = t.split("*** END OF", 1)[0]
        t = unicodedata.normalize("NFC", t.lower())
        t = "".join(c if c in ALFABETO else " " for c in t)
        t = re.sub(r" {2,}", " ", t)
        trozos.append(t)
        total += len(t)
        if total >= MAX_CARACTERES:
            break
    texto = "".join(trozos)[:MAX_CARACTERES]
    assert len(texto) >= 500_000, \
        f"Se esperaban al menos 500.000 caracteres; se encontraron {miles(len(texto))}"
    return texto


def vocabulario_y_datos(texto):
    from collections import Counter
    palabras = texto.split()
    cuenta = Counter(palabras)
    vocab = ["<rara>"] + [p for p, _ in cuenta.most_common(VOCABULARIO_MAXIMO - 1)]
    indice = {p: i for i, p in enumerate(vocab)}
    datos = torch.tensor([indice.get(p, 0) for p in palabras], dtype=torch.long)
    cobertura = sum(c for p, c in cuenta.items() if p in indice) / len(palabras)
    return vocab, indice, datos, cobertura


class RedTexto(nn.Module):
    def __init__(self, vocabulario):
        super().__init__()
        self.emb = nn.Embedding(vocabulario, 256)
        self.nucleo = nn.LSTM(256, OCULTO_TEXTO, num_layers=CAPAS_TEXTO, batch_first=True)
        self.salida = nn.Linear(OCULTO_TEXTO, vocabulario)

    def forward(self, x, estado=None):
        h, estado = self.nucleo(self.emb(x), estado)
        return self.salida(h), estado


def entrenar_texto(datos, vocabulario, minutos):
    fijar_semilla()
    modelo = RedTexto(vocabulario).to(DISPOSITIVO)
    opt = torch.optim.Adam(modelo.parameters(), lr=TASA_TEXTO)
    perdida = nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEMILLA)
    t0, pasos = time.time(), 0
    modelo.train()
    while time.time() - t0 < minutos * 60:
        i = rng.integers(0, len(datos) - LONGITUD - 1, size=LOTE_TEXTO)
        x = torch.stack([datos[j:j + LONGITUD] for j in i]).to(DISPOSITIVO)
        y = torch.stack([datos[j + 1:j + LONGITUD + 1] for j in i]).to(DISPOSITIVO)
        opt.zero_grad()
        pred, _ = modelo(x)
        l = perdida(pred.reshape(-1, vocabulario), y.reshape(-1))
        l.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        pasos += 1
    return modelo, pasos, float(l.item())


def generar(modelo, indice, vocab, arranque, largo=LARGO_MUESTRA):
    modelo.eval()
    salida, estado = arranque.split(), None
    x = torch.tensor([[indice.get(p, 0) for p in salida]], dtype=torch.long).to(DISPOSITIVO)
    salida = list(salida)
    with torch.no_grad():
        for _ in range(largo):
            logits, estado = modelo(x, estado)
            p = torch.softmax(logits[0, -1] / TEMPERATURA, dim=-1)
            p[0] = 0.0
            siguiente = int(torch.multinomial(p / p.sum(), 1))
            salida.append(vocab[siguiente])
            x = torch.tensor([[siguiente]], dtype=torch.long).to(DISPOSITIVO)
    return " ".join(salida)


# --------------------------- pruebas ---------------------------

def selftest():
    fallos = []
    fijar_semilla()
    rng = np.random.default_rng(SEMILLA)

    # 1. TEST NULO — el símbolo a recordar se sustituye por paja, así que la respuesta no
    #    está en la secuencia. El acierto debe caer al azar (1 entre 8 = 0,125).
    modelo = Recordadora(SIMBOLOS + RELLENO, OCULTO_MEMORIA, True)
    opt = torch.optim.Adam(modelo.parameters(), lr=TASA_MEMORIA)
    perdida = nn.CrossEntropyLoss()
    for _ in range(600):
        x, y = lote_memoria(10, LOTE_MEMORIA, rng)
        x[:, 0] = SIMBOLOS                       # borra la pista
        opt.zero_grad(); perdida(modelo(x), y).backward(); opt.step()
    with torch.no_grad():
        x, y = lote_memoria(10, EJEMPLOS_PRUEBA, rng); x[:, 0] = SIMBOLOS
        a_nulo = (modelo(x).argmax(1) == y).float().mean().item()
    print(f"[1] test nulo         sin pista en la secuencia: acierto {coma(a_nulo, 3)} "
          f"(azar = {coma(1 / SIMBOLOS, 3)})")
    if a_nulo > 0.25:
        fallos.append(f"test nulo: {coma(a_nulo, 3)} de acierto sin pista; algo filtra la respuesta")

    # 2. SEÑAL IMPLANTADA — a distancia 1 la tarea es trivial y tiene que resolverse.
    a_facil = medir_memoria(1, True, pasos=800)
    print(f"[2] señal implantada  a distancia 1: acierto {coma(a_facil, 3)}")
    if a_facil < UMBRAL_SELFTEST:
        fallos.append(f"señal implantada: se esperaba >= {coma(UMBRAL_SELFTEST, 2)} "
                      f"a distancia 1; se obtuvo {coma(a_facil, 3)}")

    # 3. INVARIANTE DEL DOMINIO — lo que sale del generador de texto son probabilidades:
    #    no negativas y sumando uno.
    texto = cargar_texto()[:400_000]
    vocab, indice, datos, _ = vocabulario_y_datos(texto)
    red = RedTexto(len(vocab))
    with torch.no_grad():
        logits, _ = red(datos[:64].unsqueeze(0))
        p = torch.softmax(logits[0], dim=-1)
    suma = float(p.sum(dim=-1).min()), float(p.sum(dim=-1).max())
    # .2e es notación científica en inglés (punto decimal); se castellaniza solo el punto
    # que es decimal, no el resto de la cadena (la "e" del exponente no se toca).
    minimo_es = f"{p.min():.2e}".replace(".", ",")
    print(f"[3] invariante        probabilidades por paso: mínimo {minimo_es}, suma entre "
          f"{coma(suma[0], 6)} y {coma(suma[1], 6)}")
    if p.min() < 0 or not all(abs(s - 1) < 1e-4 for s in suma):
        fallos.append(f"invariante: probabilidades fuera de rango o que no suman uno: {suma}")

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
    ap.add_argument("--minutos", type=float, default=MINUTOS_MAXIMO)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    print("--- 1. RECORDAR A DISTANCIA ---")
    print("Acierto al devolver un símbolo visto N pasos antes (azar = 0,125)\n")
    print(f"Media de {SEMILLAS_MEMORIA} entrenamientos; entre paréntesis, el peor y el mejor.")
    print(f"{'distancia':>10}{'memoria simple':>26}{'con compuertas':>26}")
    for d in DISTANCIAS:
        ms, mns, mxs = medir_memoria_repetida(d, False)
        mc, mnc, mxc = medir_memoria_repetida(d, True)
        simple = f"{coma(ms, 3)} ({coma(mns, 3)}-{coma(mxs, 3)})"
        puertas = f"{coma(mc, 3)} ({coma(mnc, 3)}-{coma(mxc, 3)})"
        print(f"{d:>10}{simple:>26}{puertas:>26}")
    print()
    # La clave de las columnas, debajo de la tabla y impresa por el programa (regla 9): un
    # rótulo que dice lo que mide no cabe en la cabecera, así que el rótulo va corto arriba
    # y la clave entera aquí abajo, en líneas que el libro copia tal cual (regla 6).
    for l in comprobar_ancho([
            "«distancia»: cuántos símbolos de paja hay en medio.",
            f"«acierto»: 1 es acertar siempre; {coma(1 / SIMBOLOS, 3)} es puro azar."]):
        print(l)

    print("\n--- 2. ESCRIBIR EN ESPAÑOL ---")
    texto = cargar_texto()
    vocab, indice, datos, cobertura = vocabulario_y_datos(texto)
    print(f"{miles(len(datos))} palabras, {miles(len(vocab))} distintas en el vocabulario "
          f"({coma(cobertura * 100)} % del texto cubierto).")
    modelo, pasos, ultima = entrenar_texto(datos, len(vocab), args.minutos)
    print(f"entrenada {args.minutos:.0f} minutos en {DISPOSITIVO}: {miles(pasos)} pasos.\n")
    for arranque in ("el caballero", "no sabía", "cuando llegó"):
        print(f"[arranque: «{arranque}»]")
        print(generar(modelo, indice, vocab, arranque))
        print()


if __name__ == "__main__":
    main()
