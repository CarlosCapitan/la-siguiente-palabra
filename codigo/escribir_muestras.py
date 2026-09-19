#!/usr/bin/env python3
"""
Capítulo 6 — las muestras de la red, separadas de medirlas.

Entrena la red recurrente de `memoria_recurrente.py` con una semilla dada y genera las
muestras de texto que luego mide `lo_habia_visto_ya.py`. Separar esto de la medida es lo que
permite cambiar la medida (cambiar el umbral, añadir un generador) sin repetir dos horas de
entrenamiento por cada cambio.

Una tirada = un entrenamiento completo con una semilla, más su generación. Se piden siempre
`--semilla` y `--tirada`: una tirada sin etiquetar no sirve para comparar nada.

Uso:
    python escribir_muestras.py --semilla 20260914 --tirada 1
"""

# ======================= CONSTANTES =======================

LARGOS = (45, 200)            # palabras por muestra
VENTANAS = 200                # muestras con ventana de 45
VENTANAS_LARGAS = 50          # muestras con ventana de 200
SEMILLA_MUESTREO = 20260919   # fija el muestreo APARTE del entrenamiento
DIR_SALIDA = "../datos/salidas/muestras"

# ==========================================================

import argparse
import hashlib
import os

import torch

import lo_habia_visto_ya as lhvy
import memoria_recurrente as mr


def huella_pesos(modelo):
    """sha256 sobre los tensores de `state_dict()`, en float32 y en orden de clave, primeros
    12 hexadecimales. Convierte «la GPU no es determinista» en un dato impreso: si dos
    tiradas de la misma semilla dan huellas distintas, dos entrenamientos idénticos sobre el
    papel han producido dos modelos distintos de verdad."""
    h = hashlib.sha256()
    estado = modelo.state_dict()
    for clave in sorted(estado.keys()):
        t = estado[clave].detach().to(torch.float32).cpu().contiguous()
        h.update(clave.encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--semilla", type=int, required=True,
                    help="semilla del entrenamiento (obligatoria: sin ella la tirada no es "
                         "comparable con nada)")
    ap.add_argument("--tirada", type=int, required=True, choices=(1, 2),
                    help="1 o 2: qué repetición de esta semilla es")
    args = ap.parse_args()

    os.makedirs(DIR_SALIDA, exist_ok=True)

    texto = mr.cargar_texto()
    vocab, indice, datos, cobertura = mr.vocabulario_y_datos(texto)
    print(f"{len(datos)} palabras, {len(vocab)} en el vocabulario "
          f"({cobertura * 100:.1f} % del texto cubierto).")

    print(f"entrenando semilla {args.semilla}, tirada {args.tirada}, "
          f"{mr.PASOS_TEXTO} pasos, dispositivo {mr.DISPOSITIVO} ...")
    modelo, pasos, ultima, segundos = mr.entrenar_texto(datos, len(vocab), mr.PASOS_TEXTO,
                                                         semilla=args.semilla)
    print(f"entrenada: {pasos} pasos, {segundos:.0f} s en esta máquina, "
          f"pérdida final {ultima:.3f}.")

    # A partir de aquí, todo en CPU: torch.multinomial en Metal no garantiza el mismo
    # resultado con la misma semilla (medido: dos generaciones idénticas en MPS difieren),
    # y en CPU sí. Así el muestreo no depende de la GPU ni de lo que el entrenamiento haya
    # consumido del generador de azar, las seis tiradas se muestrean exactamente igual, y la
    # huella de los pesos queda como lo único que puede distinguir dos tiradas.
    modelo = modelo.to("cpu")
    mr.DISPOSITIVO = torch.device("cpu")
    mr.fijar_semilla(SEMILLA_MUESTREO)

    huella = huella_pesos(modelo)
    print(f"huella de los pesos: {huella}")

    vocab_conjunto = set(vocab)
    puntos = lhvy.arranques(datos_palabras(texto), vocab_conjunto, n=VENTANAS)

    muestras = []  # (largo, arranque_texto, muestra)
    for i, tri in enumerate(puntos):
        arranque_texto = " ".join(tri)
        for largo in LARGOS:
            if largo == LARGOS[1] and i >= VENTANAS_LARGAS:
                continue  # la ventana de 200 solo se genera para las primeras VENTANAS_LARGAS
            # generar() añade `largo - len(tri)` palabras a las `len(tri)` del arranque, para
            # que la muestra completa (arranque + generado) mida exactamente `largo` palabras.
            muestra = mr.generar(modelo, indice, vocab, arranque_texto, largo=largo - len(tri))
            muestras.append((largo, arranque_texto, muestra))

    ruta = os.path.join(DIR_SALIDA, f"red_s{args.semilla}_t{args.tirada}.txt")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(f"semilla: {args.semilla}\n")
        fh.write(f"tirada: {args.tirada}\n")
        fh.write(f"pasos: {pasos}\n")
        fh.write(f"dispositivo: {mr.DISPOSITIVO}\n")
        fh.write(f"pytorch: {torch.__version__}\n")
        fh.write(f"segundos: {segundos:.1f}\n")
        fh.write(f"huella de pesos: {huella}\n")
        for largo, arranque_texto, muestra in muestras:
            fh.write(f"{largo}\t{arranque_texto}\t{muestra}\n")
    print(f"escrito: {ruta} ({len(muestras)} muestras)")


def datos_palabras(texto):
    """La lista de palabras del corpus, tal como la ve `arranques()`. Se aísla en una función
    para no repetir `texto.split()` en cada sitio que lo necesita."""
    return texto.split()


if __name__ == "__main__":
    main()
