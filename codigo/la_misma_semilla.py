#!/usr/bin/env python3
"""
Capítulo 6 — ¿por qué la misma semilla dio modelos distintos? (Orden 24.4, 19 oct 2026).

El capítulo 6 dice que las dos tiradas de una misma semilla salieron distintas (las huellas de
`escribir_muestras.py` lo confirman: b77c7fc3137d y 6b0014b74c89 con la semilla 20260914), y
`memoria_recurrente.py` afirma que «el modelo en el paso N es SIEMPRE el mismo, en cualquier máquina».
Las dos cosas no pueden ser ciertas a la vez. Este programa busca de dónde viene la diferencia.

Reutiliza, importándolos, los datos de `memoria_recurrente.py` (`cargar_texto`,
`vocabulario_y_datos`), su red `RedTexto`, la huella de los pesos `huella_pesos` de
`escribir_muestras.py` y el bucle de `entrenar_texto` (repetido aquí paso a paso, línea por línea,
para sacar los pesos en los pasos 0, 1, 10, 100 y 300; el selftest comprueba que acaba en los mismos
pesos que `entrenar_texto`). Cada tirada es un proceso aparte (subprocess), con la semilla 20260914 y
300 pasos, y escribe: la huella del vocabulario (la lista en su orden), la huella de los datos ya
convertidos a números, la huella de los lotes (los índices que sortea el entrenamiento) y la huella
de los pesos en los pasos 0, 1, 10, 100 y 300. Cuatro montajes, dos tiradas cada uno:

  1. procesador;
  2. procesador con `torch.use_deterministic_algorithms(True)`;
  3. tarjeta (mps);
  4. tarjeta con `PYTHONHASHSEED=0` en los dos procesos.

(En los montajes 1 a 3 `PYTHONHASHSEED` no está fijado: se quita del entorno de los procesos.)
Tabla: por montaje y paso, si las dos tiradas coinciden bit a bit y la diferencia máxima entre sus
pesos; y en qué paso se separan por primera vez (entre los pasos que se miran).

=====================================================================================
PREDICCIÓN DE CLAUDE, escrita antes de medir (Orden 24, 19 oct 2026, copiada tal cual):

  El paso 0 coincide en todo (0,9); en el procesador las dos tiradas coinciden hasta el paso 300
  (0,8); en la tarjeta se separan desde los primeros pasos (0,75).
  Qué cambia el texto: si el procesador coincide y la tarjeta no, la causa es la tarjeta y el cap. 6
  lo explica ahí mismo, en vez de mandar al 9; si ya difieren el vocabulario, los datos o el paso 0,
  es un fallo del programa (no era «todo idéntico») y hay que arreglarlo; si todo coincide en los
  dos, 300 pasos no bastan para verlo y lo dices.
=====================================================================================

Uso:
    python la_misma_semilla.py --selftest
    python la_misma_semilla.py > ../datos/salidas/la_misma_semilla.txt
"""

# ======================= CONSTANTES =======================

SEMILLA = 20260914
PASOS = 300
PASOS_VISTOS = [0, 1, 10, 100, 300]
MONTAJES = [
    # (nombre, dispositivo, algoritmos deterministas, PYTHONHASHSEED)
    ("procesador", "cpu", False, None),
    ("procesador, algoritmos deterministas", "cpu", True, None),
    ("tarjeta (mps)", "mps", False, None),
    ("tarjeta, PYTHONHASHSEED=0", "mps", False, "0"),
]
TIRADAS = 2
SELFTEST_PASOS = 5                  # pasos del selftest
SELFTEST_CARACTERES = 400_000       # texto del selftest, como en memoria_recurrente.py
ANCHO_SALIDA = 62                   # líneas de no más de 62 caracteres, fuera de las tablas

# ==========================================================

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import memoria_recurrente as mr
from escribir_muestras import huella_pesos
from formato import coma, comprobar_ancho, miles, tabla_editorial

AQUI = Path(__file__).resolve().parent


def sha(datos):
    return hashlib.sha256(datos).hexdigest()[:12]


def huella_del_vocabulario(vocab):
    """La lista, en su orden: una palabra por línea."""
    return sha("\n".join(vocab).encode("utf-8"))


def huella_de_los_datos(datos):
    """Los datos ya convertidos a números (el tensor de enteros que entrena la red)."""
    return sha(datos.numpy().astype(np.int64).tobytes())


def estado_de(modelo):
    """Los pesos, copiados a CPU en float32."""
    return {k: v.detach().to(torch.float32).cpu().clone() for k, v in modelo.state_dict().items()}


def diferencia_maxima(a, b):
    """La mayor diferencia entre dos pesos cualesquiera (en valor absoluto) de dos estados."""
    assert a.keys() == b.keys(), "Se esperaban los mismos tensores en los dos estados"
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def entrenar_con_instantaneas(datos, nvocab, dispositivo, pasos_vistos=PASOS_VISTOS, semilla=SEMILLA):
    """El bucle de `memoria_recurrente.entrenar_texto`, línea por línea, sacando en cada paso de
    `pasos_vistos` la huella y una copia de los pesos, y la huella de los lotes (los índices que se
    han sorteado hasta ese paso). Devuelve {paso: (huella de pesos, huella de lotes, pesos)}."""
    mr.fijar_semilla(semilla)
    modelo = mr.RedTexto(nvocab).to(dispositivo)
    opt = torch.optim.Adam(modelo.parameters(), lr=mr.TASA_TEXTO)
    perdida = nn.CrossEntropyLoss()
    rng = np.random.default_rng(semilla)
    lotes = hashlib.sha256()
    salida = {}
    if 0 in pasos_vistos:
        salida[0] = (huella_pesos(modelo), lotes.hexdigest()[:12], estado_de(modelo))
    modelo.train()
    for paso in range(1, max(pasos_vistos) + 1):
        i = rng.integers(0, len(datos) - mr.LONGITUD - 1, size=mr.LOTE_TEXTO)
        lotes.update(i.astype(np.int64).tobytes())
        x = torch.stack([datos[j:j + mr.LONGITUD] for j in i]).to(dispositivo)
        y = torch.stack([datos[j + 1:j + mr.LONGITUD + 1] for j in i]).to(dispositivo)
        opt.zero_grad()
        pred, _ = modelo(x)
        l = perdida(pred.reshape(-1, nvocab), y.reshape(-1))
        l.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step()
        if paso in pasos_vistos:
            salida[paso] = (huella_pesos(modelo), lotes.hexdigest()[:12], estado_de(modelo))
    return salida


# ---------------------------------------------------------------- una tirada (en su proceso)

def una_tirada(ruta, dispositivo, deterministas):
    """Lo que hace cada proceso aparte: carga, convierte, entrena PASOS pasos y escribe en `ruta` las
    huellas y los pesos de los pasos vistos."""
    if deterministas:
        torch.use_deterministic_algorithms(True)
    t0 = time.time()
    texto = mr.cargar_texto()
    vocab, _, datos, _ = mr.vocabulario_y_datos(texto)
    vistas = entrenar_con_instantaneas(datos, len(vocab), torch.device(dispositivo))
    torch.save(dict(
        vocabulario=huella_del_vocabulario(vocab), datos=huella_de_los_datos(datos),
        palabras=len(datos), vocab=len(vocab), pytorch=str(torch.__version__),
        hashseed=os.environ.get("PYTHONHASHSEED", "sin fijar"), segundos=time.time() - t0,
        pasos={p: (h, hl) for p, (h, hl, _) in vistas.items()},
        pesos={p: e for p, (_, _, e) in vistas.items()}), ruta)


def lanzar(ruta, dispositivo, deterministas, hashseed):
    """Un proceso nuevo, sin PYTHONHASHSEED salvo que el montaje lo pida."""
    entorno = dict(os.environ)
    entorno.pop("PYTHONHASHSEED", None)
    if hashseed is not None:
        entorno["PYTHONHASHSEED"] = hashseed
    orden = [sys.executable, str(AQUI / Path(__file__).name), "--tirada", str(ruta),
             "--dispositivo", dispositivo]
    if deterministas:
        orden.append("--deterministas")
    r = subprocess.run(orden, env=entorno, cwd=AQUI, capture_output=True, text=True)
    assert r.returncode == 0, f"La tirada ha fallado ({r.returncode}):\n{r.stderr[-1500:]}"
    return torch.load(ruta)


# ---------------------------------------------------------------- impresión

def comparar(a, b):
    """De dos tiradas: por paso, (¿coinciden los pesos bit a bit?, ¿los lotes?, diferencia máxima); y los
    pasos en que se separan por primera vez los pesos."""
    por_paso = {}
    for p in PASOS_VISTOS:
        por_paso[p] = (a["pasos"][p][0] == b["pasos"][p][0], a["pasos"][p][1] == b["pasos"][p][1],
                       diferencia_maxima(a["pesos"][p], b["pesos"][p]))
    primero = next((p for p in PASOS_VISTOS if not por_paso[p][0]), None)
    return por_paso, primero


def cientifica(x):
    if x == 0:
        return "0"
    return f"{x:.2e}".replace(".", ",")


def informe(resultados, emitir):
    emitir(["--- LAS DOS TIRADAS DE CADA MONTAJE ---", ""])
    filas_a = []
    for nombre, (a, b) in resultados.items():
        filas_a.append([nombre, "sí" if a["vocabulario"] == b["vocabulario"] else "NO",
                        "sí" if a["datos"] == b["datos"] else "NO",
                        "sí" if all(a["pasos"][p][1] == b["pasos"][p][1] for p in PASOS_VISTOS) else "NO",
                        a["hashseed"] + " / " + b["hashseed"]])
    emitir(tabla_editorial(
        "Lo que entra: ¿coinciden las dos tiradas?",
        ["montaje", "vocabulario", "datos", "lotes", "PYTHONHASHSEED"], filas_a, "icccc",
        ["Vocabulario: la lista de palabras, en su orden. Datos: el texto ya convertido a números.",
         "Lotes: los índices que sortea el entrenamiento, en los 300 pasos."]))
    emitir([""])
    filas, separa = [], {}
    for nombre, (a, b) in resultados.items():
        por_paso, primero = comparar(a, b)
        separa[nombre] = primero
        for p in PASOS_VISTOS:
            igual, _, dif = por_paso[p]
            filas.append([nombre, str(p), "sí" if igual else "no", cientifica(dif)])
    emitir(tabla_editorial(
        "Los pesos de las dos tiradas, por montaje y paso",
        ["montaje", "paso", "¿coinciden bit a bit?", "diferencia máxima"], filas, "idcd",
        ["Misma semilla (20260914), 300 pasos, un proceso aparte por tirada.",
         "Diferencia máxima: lo que más se separan dos pesos de las dos tiradas."]))
    emitir([""])
    emitir(tabla_editorial(
        "En qué paso se separan por primera vez",
        ["montaje", "primer paso en que difieren"],
        [[n, "no se separan hasta el 300" if p is None else str(p)] for n, p in separa.items()], "id",
        [f"Entre los pasos que se miran: {', '.join(map(str, PASOS_VISTOS))}."]))
    emitir([""])
    emitir(tabla_editorial(
        "Las huellas de los pesos, tirada a tirada",
        ["montaje", "paso", "tirada 1", "tirada 2"],
        [[n, str(p), a["pasos"][p][0], b["pasos"][p][0]] for n, (a, b) in resultados.items()
         for p in PASOS_VISTOS], "idii",
        ["Doce primeras cifras hexadecimales del sha256 de los pesos, en float32 y en orden de clave."]))
    emitir([""])
    return separa


# ---------------------------------------------------------------- selftest

def selftest():
    fallos = []
    texto = mr.cargar_texto()[:SELFTEST_CARACTERES]
    vocab, _, datos, _ = mr.vocabulario_y_datos(texto)
    cpu = torch.device("cpu")
    pasos = [0, 1, SELFTEST_PASOS]

    # 1. TEST NULO — dos semillas distintas dan huellas distintas ya en el paso 0 (si no, la huella no
    #    distingue nada), y las dos huellas del mismo modelo, calculadas dos veces, son iguales.
    a = entrenar_con_instantaneas(datos, len(vocab), cpu, pasos, SEMILLA)
    b = entrenar_con_instantaneas(datos, len(vocab), cpu, pasos, SEMILLA + 1)
    print(f"[1] test nulo         semillas distintas, huellas del paso 0: {a[0][0]} y {b[0][0]}; "
          f"y del paso {SELFTEST_PASOS}: {a[SELFTEST_PASOS][0]} y {b[SELFTEST_PASOS][0]}")
    if a[0][0] == b[0][0] or a[SELFTEST_PASOS][0] == b[SELFTEST_PASOS][0]:
        fallos.append("test nulo: dos semillas distintas dan la misma huella")

    # 2. SEÑAL IMPLANTADA — el bucle de aquí acaba en los mismos pesos, bit a bit, que
    #    `memoria_recurrente.entrenar_texto` (en el procesador); y un peso cambiado a mano se ve.
    mr.DISPOSITIVO = cpu
    modelo, _, _, _ = mr.entrenar_texto(datos, len(vocab), SELFTEST_PASOS, SEMILLA)
    suya = huella_pesos(modelo)
    mia = a[SELFTEST_PASOS][0]
    tocado = {k: v.clone() for k, v in a[SELFTEST_PASOS][2].items()}
    primera = next(iter(tocado))
    tocado[primera].view(-1)[0] += 1.0
    dif = diferencia_maxima(a[SELFTEST_PASOS][2], tocado)
    print(f"[2] señal implantada  mi bucle y entrenar_texto, a los {SELFTEST_PASOS} pasos: {mia} y {suya}; un "
          f"peso cambiado en 1 se ve: diferencia {coma(dif, 1)}")
    if mia != suya or abs(dif - 1.0) > 1e-6:
        fallos.append("señal implantada: el bucle repetido no es el de entrenar_texto, o la diferencia no se ve")

    # 3. INVARIANTE — en el procesador, el mismo entrenamiento dos veces en el mismo proceso da los
    #    mismos pesos bit a bit, los mismos lotes y diferencia máxima 0; y los hashes del vocabulario y de
    #    los datos son los mismos dos veces.
    c = entrenar_con_instantaneas(datos, len(vocab), cpu, pasos, SEMILLA)
    iguales = all(a[p][0] == c[p][0] and a[p][1] == c[p][1] for p in pasos)
    dif0 = max(diferencia_maxima(a[p][2], c[p][2]) for p in pasos)
    hv = huella_del_vocabulario(vocab) == huella_del_vocabulario(mr.vocabulario_y_datos(texto)[0])
    hd = huella_de_los_datos(datos) == huella_de_los_datos(mr.vocabulario_y_datos(texto)[2])
    print(f"[3] invariante        dos veces lo mismo en el procesador: pesos y lotes iguales: "
          f"{'sí' if iguales else 'NO'}; diferencia máxima {cientifica(dif0)}; vocabulario y datos iguales: "
          f"{'sí' if hv and hd else 'NO'}")
    if not iguales or dif0 != 0.0 or not (hv and hd):
        fallos.append("invariante: el procesador no repite lo mismo dos veces en el mismo proceso")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tirada", help="(interno) fichero donde escribir una tirada")
    ap.add_argument("--dispositivo", choices=("cpu", "mps"), help="(interno) con --tirada")
    ap.add_argument("--deterministas", action="store_true", help="(interno) con --tirada")
    args = ap.parse_args()
    if args.tirada:
        una_tirada(args.tirada, args.dispositivo, args.deterministas)
        return 0
    if args.selftest:
        return selftest()
    assert torch.backends.mps.is_available(), "Se esperaba la tarjeta (mps) en este portátil"

    def emitir(lineas):
        dentro = getattr(emitir, "dentro", False)
        for l in lineas:
            if l.startswith("::: "):
                dentro = True
            if not dentro:
                comprobar_ancho([l], ANCHO_SALIDA)
            if l == ":::":
                dentro = False
            print(l)
        emitir.dentro = dentro

    emitir(["######## la misma semilla (Orden 24) ########",
            f"fecha: {date.today().isoformat()}",
            f"máquina: {platform.system()} {platform.machine()}, Apple M4 Max",
            f"python {platform.python_version()}, torch {torch.__version__}",
            f"semilla {SEMILLA}, {PASOS} pasos, {TIRADAS} tiradas por montaje,",
            "cada tirada en un proceso aparte.",
            "Los montajes 3 y 4 entrenan en la tarjeta (mps).", ""])
    resultados = {}
    with tempfile.TemporaryDirectory() as carpeta:
        for k, (nombre, dispositivo, deterministas, hashseed) in enumerate(MONTAJES, 1):
            tiradas = [lanzar(Path(carpeta) / f"m{k}_t{t}.pt", dispositivo, deterministas, hashseed)
                       for t in range(1, TIRADAS + 1)]
            resultados[nombre] = tuple(tiradas)
    otros = {(t["vocabulario"], t["datos"]) for pareja in resultados.values() for t in pareja}
    informe(resultados, emitir)
    emitir(tabla_editorial(
        "El texto y el vocabulario, en las ocho tiradas", ["", "valor"],
        [["huellas distintas de vocabulario y datos", str(len(otros))],
         ["palabras del texto", miles(next(iter(resultados.values()))[0]["palabras"])],
         ["palabras en el vocabulario", miles(next(iter(resultados.values()))[0]["vocab"])]], "id",
        ["Una sola huella: las ocho tiradas leyeron lo mismo."] if len(otros) == 1 else
        ["Más de una huella: las tiradas no leyeron lo mismo."]))
    emitir([""])
    return 0


if __name__ == "__main__":
    sys.exit(main())
