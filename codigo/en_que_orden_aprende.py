#!/usr/bin/env python3
"""
Capítulo 9 — en qué orden aprende un transformer entrenado desde cero.

Entrena un transformer minúsculo, letra a letra, sobre libros en español, y guarda lo que
escribe en cuatro momentos del entrenamiento. Mide también lo mal que lo hace a lo largo del
tiempo y el acierto al principio y al final.

El capítulo tiene un presupuesto de números acotado (ver REGLAS.md del repositorio privado):
cuatro muestras de texto, una curva, dos porcentajes y una comparación de tamaño. Este guion
produce exactamente eso y nada más.

Uso:
    python en_que_orden_aprende.py
    python en_que_orden_aprende.py --selftest
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260914
CORPUS = "../datos/corpus_es"
MAX_CARACTERES = 8_000_000

CONTEXTO = 128             # letras que puede mirar hacia atrás
ANCHO = 192                # números por letra
CAPAS = 4
CABEZAS = 4
LOTE = 64
TASA = 3e-4

MINUTOS_TOTAL = 25.0
INSTANTANEAS_MIN = [0.5, 2.0, 8.0, 25.0]   # cuándo pedirle que escriba
ARRANQUE = "el "
LARGO_MUESTRA = 200
TEMPERATURA = 0.8
EJEMPLOS_ACIERTO = 20_000

SALIDA_CURVA = "../datos/en_que_orden_aprende_curva.csv"
SALIDA_FIGURA = "../figuras/en_que_orden_aprende.png"
MODELO_GRANDE = 500_000_000_000   # orden de magnitud de un modelo grande de hoy

# ==========================================================

import argparse
import csv
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
import torch.nn.functional as F

ALFABETO = "abcdefghijklmnñopqrstuvwxyzáéíóúü ,.;:¿?¡!\n"


def dispositivo():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DISPOSITIVO = dispositivo()


def fijar_semilla(s=SEMILLA):
    torch.manual_seed(s)
    np.random.seed(s)


def cargar_texto(maximo=MAX_CARACTERES):
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
        trozos.append(re.sub(r" {2,}", " ", t))
        total += len(trozos[-1])
        if total >= maximo:
            break
    texto = "".join(trozos)[:maximo]
    assert len(texto) >= 200_000, \
        f"Se esperaban al menos 200.000 caracteres; se encontraron {len(texto):,}"
    return texto


class Bloque(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(ANCHO)
        self.at = nn.MultiheadAttention(ANCHO, CABEZAS, batch_first=True)
        self.n2 = nn.LayerNorm(ANCHO)
        self.densa = nn.Sequential(nn.Linear(ANCHO, 4 * ANCHO), nn.GELU(),
                                   nn.Linear(4 * ANCHO, ANCHO))

    def forward(self, x, mascara):
        h = self.n1(x)
        a, _ = self.at(h, h, h, attn_mask=mascara, need_weights=False)
        x = x + a
        return x + self.densa(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, vocabulario):
        super().__init__()
        self.emb = nn.Embedding(vocabulario, ANCHO)
        self.pos = nn.Embedding(CONTEXTO, ANCHO)
        self.bloques = nn.ModuleList([Bloque() for _ in range(CAPAS)])
        self.norma = nn.LayerNorm(ANCHO)
        self.salida = nn.Linear(ANCHO, vocabulario)

    def forward(self, idx):
        n = idx.shape[1]
        mascara = torch.triu(torch.ones(n, n, device=idx.device, dtype=torch.bool), 1)
        x = self.emb(idx) + self.pos(torch.arange(n, device=idx.device))
        for b in self.bloques:
            x = b(x, mascara)
        return self.salida(self.norma(x))


def lote(datos, rng):
    i = rng.integers(0, len(datos) - CONTEXTO - 1, size=LOTE)
    x = torch.stack([datos[j:j + CONTEXTO] for j in i]).to(DISPOSITIVO)
    y = torch.stack([datos[j + 1:j + CONTEXTO + 1] for j in i]).to(DISPOSITIVO)
    return x, y


def escribir(modelo, indice, letras, largo=LARGO_MUESTRA, arranque=ARRANQUE, semilla=SEMILLA):
    modelo.eval()
    g = torch.Generator(device="cpu").manual_seed(semilla)
    idx = torch.tensor([[indice[c] for c in arranque]], dtype=torch.long, device=DISPOSITIVO)
    salida = list(arranque)
    with torch.no_grad():
        for _ in range(largo):
            logits = modelo(idx[:, -CONTEXTO:])[0, -1]
            p = torch.softmax(logits.float() / TEMPERATURA, dim=-1).cpu()
            siguiente = int(torch.multinomial(p, 1, generator=g))
            salida.append(letras[siguiente])
            idx = torch.cat([idx, torch.tensor([[siguiente]], device=DISPOSITIVO)], dim=1)
    modelo.train()
    return "".join(salida).replace("\n", " ")


def acierto(modelo, datos, rng, n=EJEMPLOS_ACIERTO):
    modelo.eval()
    aciertos = vistos = 0
    with torch.no_grad():
        while vistos < n:
            x, y = lote(datos, rng)
            pred = modelo(x).argmax(-1)
            aciertos += int((pred == y).sum())
            vistos += y.numel()
    modelo.train()
    return aciertos / vistos


def entrenar(datos, vocabulario, indice, letras, minutos, instantaneas):
    fijar_semilla()
    rng = np.random.default_rng(SEMILLA)
    modelo = Transformer(vocabulario).to(DISPOSITIVO)
    opt = torch.optim.AdamW(modelo.parameters(), lr=TASA)
    curva, muestras = [], []
    pendientes = list(instantaneas)
    a_inicial = acierto(modelo, datos, np.random.default_rng(1))
    t0, pasos = time.time(), 0
    modelo.train()
    while True:
        transcurrido = (time.time() - t0) / 60
        if pendientes and transcurrido >= pendientes[0]:
            m = pendientes.pop(0)
            muestras.append((m, pasos, escribir(modelo, indice, letras)))
        if transcurrido >= minutos:
            break
        x, y = lote(datos, rng)
        opt.zero_grad()
        p = modelo(x)
        perdida = F.cross_entropy(p.reshape(-1, vocabulario), y.reshape(-1))
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        pasos += 1
        if pasos % 20 == 0:
            curva.append((transcurrido, float(perdida.item())))
    a_final = acierto(modelo, datos, np.random.default_rng(1))
    return modelo, pasos, curva, muestras, a_inicial, a_final


def figura(curva):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(sin matplotlib: no se dibuja la figura)")
        return
    os.makedirs(os.path.dirname(SALIDA_FIGURA), exist_ok=True)
    t = [c[0] for c in curva]
    v = [c[1] for c in curva]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(t, v, linewidth=1.1, color="#333333")
    ax.set_xlabel("minutos de entrenamiento")
    ax.set_ylabel("lo mal que lo hace")
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(SALIDA_FIGURA, dpi=200)
    print(f"Figura escrita en {SALIDA_FIGURA}")


def selftest():
    fallos = []
    texto = cargar_texto(600_000)
    letras = sorted(set(texto))
    indice = {c: i for i, c in enumerate(letras)}
    datos = torch.tensor([indice[c] for c in texto], dtype=torch.long)
    vocab = set(texto.split())

    # 1. TEST NULO — con el texto barajado letra a letra no hay estructura que aprender: el
    #    acierto tiene que quedarse muy por debajo del que se alcanza con texto real.
    barajado = list(texto[:400_000])
    np.random.default_rng(SEMILLA).shuffle(barajado)
    d_nulo = torch.tensor([indice[c] for c in barajado], dtype=torch.long)
    _, _, _, _, _, a_nulo = entrenar(d_nulo, len(letras), indice, letras, 1.0, [])
    _, _, _, _, _, a_real = entrenar(datos, len(letras), indice, letras, 1.0, [])
    print(f"[1] test nulo         acierto: texto real {a_real:.3f}  texto barajado {a_nulo:.3f}")
    if a_nulo >= a_real:
        fallos.append(f"test nulo: el texto barajado alcanza {a_nulo:.3f}, no menos que el real "
                      f"({a_real:.3f}); el montaje no distingue estructura")

    # 2. SEÑAL IMPLANTADA — una cadena rara repetida muchas veces debe acabar apareciendo.
    marca = "qxqxqx"
    implantado = texto[:300_000] + (" " + marca) * 3000
    letras2 = sorted(set(implantado))
    indice2 = {c: i for i, c in enumerate(letras2)}
    d2 = torch.tensor([indice2[c] for c in implantado], dtype=torch.long)
    m2, _, _, _, _, _ = entrenar(d2, len(letras2), indice2, letras2, 2.0, [])
    escrito = escribir(m2, indice2, letras2, 600, arranque="qxqx")
    print(f"[2] señal implantada  «{marca}» aparece {escrito.count(marca)} veces en 600 letras")
    if marca not in escrito:
        fallos.append(f"señal implantada: «{marca}» no aparece en lo que escribe")

    # 3. INVARIANTE DEL DOMINIO — la salida por letra son probabilidades: suman uno.
    modelo = Transformer(len(letras)).to(DISPOSITIVO)
    with torch.no_grad():
        p = torch.softmax(modelo(datos[:CONTEXTO].unsqueeze(0).to(DISPOSITIVO))[0].float(), -1)
    sumas = p.sum(-1)
    print(f"[3] invariante        {p.shape[1]} letras posibles; las probabilidades suman entre "
          f"{float(sumas.min()):.6f} y {float(sumas.max()):.6f}")
    if not torch.allclose(sumas, torch.ones_like(sumas), atol=1e-4):
        fallos.append("invariante: las probabilidades por letra no suman uno")

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
    ap.add_argument("--minutos", type=float, default=MINUTOS_TOTAL)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    texto = cargar_texto()
    letras = sorted(set(texto))
    indice = {c: i for i, c in enumerate(letras)}
    datos = torch.tensor([indice[c] for c in texto], dtype=torch.long)
    print(f"Procesador usado: {DISPOSITIVO}")
    print(f"{len(texto):,} letras de libros en español, {len(letras)} símbolos distintos.\n")

    instantaneas = [m for m in INSTANTANEAS_MIN if m <= args.minutos]
    modelo, pasos, curva, muestras, a_ini, a_fin = entrenar(
        datos, len(letras), indice, letras, args.minutos, instantaneas)

    total = sum(p.numel() for p in modelo.parameters())
    print("--- TAMAÑO ---")
    print(f"números ajustables: {total:,}")
    print(f"un modelo grande de hoy tiene del orden de {MODELO_GRANDE:,}, "
          f"unas {MODELO_GRANDE/total:,.0f} veces más")
    print(f"pasos de entrenamiento en {args.minutos:.0f} minutos: {pasos:,}\n")

    print("--- LO QUE ESCRIBE, EN CUATRO MOMENTOS ---")
    for minutos, paso, muestra in muestras:
        etiqueta = f"{int(minutos*60)} segundos" if minutos < 1 else f"{minutos:g} minutos"
        print(f"\n[a los {etiqueta} — {paso:,} pasos]")
        print(muestra)

    print("\n--- ACIERTO AL ADIVINAR LA SIGUIENTE LETRA ---")
    print(f"al empezar: {a_ini*100:.0f} %")
    print(f"al acabar:  {a_fin*100:.0f} %")

    os.makedirs(os.path.dirname(SALIDA_CURVA), exist_ok=True)
    with open(SALIDA_CURVA, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["minutos", "lo_mal_que_lo_hace"]] +
                                 [[f"{a:.4f}", f"{b:.4f}"] for a, b in curva])
    print(f"\nCurva escrita en {SALIDA_CURVA}")
    figura(curva)


if __name__ == "__main__":
    main()
