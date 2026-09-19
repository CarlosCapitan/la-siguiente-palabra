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
    python escribir_muestras.py --selftest
"""

# ======================= CONSTANTES =======================

LARGOS = (45, 200)            # palabras por muestra
VENTANAS = 200                # muestras con ventana de 45
VENTANAS_LARGAS = 50          # muestras con ventana de 200
SEMILLA_MUESTREO = 20260919   # fija el muestreo APARTE del entrenamiento
DIR_SALIDA = "../datos/salidas/muestras"

# ==========================================================

import argparse
import copy
import hashlib
import os
import sys

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


def _modelo_minusculo():
    """Un modelo diminuto en CPU, solo para probar `huella_pesos()`: no hace falta el
    vocabulario ni el corpus para comprobar que una función que resume tensores lo hace bien."""
    return torch.nn.Linear(4, 3)


def selftest():
    fallos = []

    # 1. TEST NULO — dos modelos con pesos distintos tienen que dar huellas distintas.
    torch.manual_seed(0)
    modelo_a = _modelo_minusculo()
    torch.manual_seed(1)
    modelo_b = _modelo_minusculo()
    huella_a, huella_b = huella_pesos(modelo_a), huella_pesos(modelo_b)
    print(f"[1] test nulo         pesos distintos: huellas {huella_a} y {huella_b}")
    if huella_a == huella_b:
        fallos.append("test nulo: dos modelos con pesos distintos dan la misma huella")

    # 2. SEÑAL IMPLANTADA — cambiar UN solo valor tiene que cambiar la huella.
    modelo_c = copy.deepcopy(modelo_a)
    with torch.no_grad():
        clave = next(iter(modelo_c.state_dict().keys()))
        modelo_c.state_dict()[clave].view(-1)[0] += 1.0
    huella_c = huella_pesos(modelo_c)
    print(f"[2] señal implantada  un valor cambiado en «{clave}»: huella {huella_c} "
          f"(antes {huella_a})")
    if huella_c == huella_a:
        fallos.append("señal implantada: cambiar un solo valor no cambia la huella")

    # 3. INVARIANTE DEL DOMINIO — la huella depende de los pesos, no de cómo se recorren ni
    #    de en qué dispositivo viven. `huella_pesos()` ordena por clave y pasa todo a CPU en
    #    float32 antes de sumar: esto comprueba que ese cuidado sirve para algo.
    estado_al_reves = dict(reversed(list(modelo_a.state_dict().items())))
    modelo_d = _modelo_minusculo()
    modelo_d.load_state_dict(estado_al_reves)
    huella_d = huella_pesos(modelo_d)
    print(f"[3a] invariante       mismos pesos, diccionario al revés: huella {huella_d} "
          f"(se espera {huella_a})")
    if huella_d != huella_a:
        fallos.append("invariante: el orden del diccionario de pesos cambia la huella")

    if torch.backends.mps.is_available():
        modelo_e = copy.deepcopy(modelo_a).to("mps")
        huella_e = huella_pesos(modelo_e)
        print(f"[3b] invariante       mismos pesos, en mps: huella {huella_e} "
              f"(se espera {huella_a})")
        if huella_e != huella_a:
            fallos.append("invariante: la huella cambia según el dispositivo (mps)")
    else:
        print("[3b] invariante       sin GPU (Metal) en esta máquina: prueba omitida")

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
    ap.add_argument("--semilla", type=int,
                    help="semilla del entrenamiento (obligatoria salvo con --selftest: sin "
                         "ella la tirada no es comparable con nada)")
    ap.add_argument("--tirada", type=int, choices=(1, 2),
                    help="1 o 2: qué repetición de esta semilla es (obligatoria salvo con "
                         "--selftest)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if args.semilla is None or args.tirada is None:
        ap.error("--semilla y --tirada son obligatorios sin --selftest")

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
    dispositivo_entrenamiento = str(mr.DISPOSITIVO)   # se guarda ANTES de cambiarlo
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
        # Las seis tiradas del 19 de septiembre de 2026 escribieron aquí «dispositivo: cpu»
        # porque esta línea leía el global DESPUÉS de pasarlo a CPU para generar: el
        # entrenamiento fue en la GPU (Metal), como delatan los ~1.300 s por tirada (en CPU
        # son 11,7 h medidas), pero la cabecera decía otra cosa. Fallo 4.6 en la propia
        # cabecera. Desde ahora se escriben los dos dispositivos, cada uno con su nombre.
        fh.write(f"dispositivo de entrenamiento: {dispositivo_entrenamiento}\n")
        fh.write(f"dispositivo de generacion: {mr.DISPOSITIVO}\n")
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
