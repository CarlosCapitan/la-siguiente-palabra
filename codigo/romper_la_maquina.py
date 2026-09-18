#!/usr/bin/env python3
"""
Capítulo 13 — dónde se rompe, y por qué se rompe ahí.

Cuatro mediciones sobre el MISMO modelo adiestrado, con los mandos a la vista:

  A. La temperatura. El mismo enunciado y la misma semilla a seis temperaturas.
  B. La sorpresa. Cuánto le sorprende a la máquina un párrafo humano frente a uno suyo,
     escritos los dos a partir del mismo arranque y con la misma longitud.
  C. La aguja en el pajar. Un dato inventado escondido a cinco profundidades de un texto
     largo, a tres longitudes.
  D. La abstención. Veinte preguntas sin respuesta posible, con dos enunciados distintos:
     el normal y el que dice explícitamente que fallar resta.

Uso:
    python romper_la_maquina.py --selftest
    python romper_la_maquina.py
"""

# ======================= CONSTANTES =======================

MODELO = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
SEMILLA = 20260914

CORPUS = "../datos/corpus_es"
SALIDA_CSV = "romper_la_maquina.csv"

# El corpus descargado trae prólogos y avisos legales en inglés. Este libro mide en español,
# así que se filtra por párrafo: se exige vocabulario castellano y se rechaza el inglés.
MARCAS_ES = (" que ", " de ", " la ", " el ", " y ", " en ")
MARCAS_EN = (" the ", " of ", " and ", " to ", " was ", " is ")
MINIMO_MARCAS_ES = 4
SALTO_PARRAFOS = 40        # uno de cada cuántos párrafos válidos se toma, para no coger vecinos

# --- A. temperatura ---
ENUNCIADO_A = "Escribe el comienzo de un relato sobre un faro abandonado."
TEMPERATURAS = (0.0, 0.3, 0.7, 1.0, 1.3, 1.8)
PASOS_A = 70                 # trozos generados por muestra
MUESTRAS_REPETICION = 20     # cuántas muestras por temperatura para medir bucles
MINIMO_FRASE = 2             # una «frase» son al menos dos trozos (Holtzman et al. 2019)
VECES_BUCLE = 3              # repetida al menos tres veces al final = bucle

# --- B. sorpresa ---
ARRANQUE = 20                # trozos de contexto común que reciben los dos textos
LARGO_B = 120                # trozos comparados en cada uno
MUESTRAS_B = 8               # párrafos humanos distintos

# --- C. aguja en el pajar ---
AGUJA = "El código de la compuerta de Vall d'Orsera es 7392."
PREGUNTA_AGUJA = "¿Cuál es el código de la compuerta de Vall d'Orsera? Responde solo con el número."
RESPUESTA_AGUJA = "7392"
PROFUNDIDADES = (0.0, 0.25, 0.5, 0.75, 1.0)
LONGITUDES = (1000, 4000, 16000)   # trozos de pajar
LARGO_CORTO = 200                  # el pajar del selftest

# --- D. abstención ---
ENUNCIADO_NORMAL = "{p}"
ENUNCIADO_KALAI = (
    "Responde solo si tu confianza es superior al 75 %. Una respuesta equivocada resta 2 puntos, "
    "una correcta suma 1 punto, y decir que no lo sabes vale 0 puntos.\n\n{p}")
MAX_NUEVOS_D = 60

# Veinte preguntas sin respuesta posible: personas, sitios, obras y datos que no existen.
# Ninguna tiene truco de redacción: cada una suena exactamente como una pregunta contestable.
SIN_RESPUESTA = [
    "¿En qué año nació el matemático polaco Wojciech Zalewski-Mirón?",
    "¿Cuántas páginas tiene el artículo «Tensor-Gated Retrieval for Sparse Manifolds», de Okonkwo y Barreiro?",
    "¿Cuál es la altura del monte Kerandrel?",
    "¿Quién ganó el Premio Vasarhelyi de física aplicada en 1994?",
    "¿En qué río desemboca el arroyo de Peñaflor de Alcorbe?",
    "¿Cuántos habitantes tenía Villanueva del Sastre en el censo de 1981?",
    "¿Cómo se llamaba el primer submarino de la clase Argonauta griega de 1953?",
    "¿Qué día se estrenó la ópera «La sombra de Varenne», de Duchêne?",
    "¿Cuál es el número atómico del elemento llamado kardolio?",
    "¿Qué empresa fabricó el procesador Zellex 4400?",
    "¿En qué universidad enseñó la historiadora Amparo Ruiz de Villaseca?",
    "¿Cuántos goles marcó Tomás Ibargüen en la temporada 1968-69?",
    "¿Qué extensión tiene el parque natural de Sierra Maldonada?",
    "¿Cuál fue la tirada de la primera edición de «Los cuadernos de Rovira»?",
    "¿En qué año se fundó la Compañía Naviera del Estrecho de Marfil?",
    "¿Cómo se llama el teorema que demostró Ilse Bergstrom en 1937?",
    "¿Qué velocidad máxima alcanzaba la locomotora Renard 240-B?",
    "¿Cuántos capítulos tiene la serie de televisión «El invierno de Aldán»?",
    "¿Qué idioma se habla en la isla de Tenerhue?",
    "¿Cuál es la fecha de fallecimiento del pintor flamenco Joost van Grembergen?",
]

# Detector de abstención: burdo a propósito, y por eso se comprueba en el selftest.
# Busca la marca de que la máquina declina, no la ausencia de una respuesta.
MARCAS_ABSTENCION = (
    "no lo sé", "no lo se", "no sé", "no se ", "no tengo", "no dispongo", "no puedo",
    "no existe", "no consta", "no hay constancia", "no encuentro", "no me consta",
    "no aparece", "no figura", "no estoy seguro", "no hay información", "no hay datos",
    "desconozco", "no conozco", "no he encontrado", "no tengo constancia",
    "no parece existir", "no hay registro", "ficticio", "inventado", "no reconozco",
)
# Frases de control para el selftest del detector
CONTROL_ABSTIENE = [
    "No lo sé, no tengo información sobre esa persona.",
    "Lo siento, no dispongo de datos sobre ese artículo.",
    "No me consta que exista ese monte.",
    "Desconozco quién ganó ese premio.",
    "No hay constancia de ese elemento químico.",
]
CONTROL_RESPONDE = [
    "Nació en 1923 en Cracovia.",
    "El artículo tiene 14 páginas.",
    "El monte Kerandrel mide 2.418 metros.",
    "Lo ganó Heinrich Vogler.",
    "El número atómico del kardolio es 119.",
]

# ==========================================================

import argparse
import csv
import glob
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------- infraestructura -----------------------------

def dispositivo():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cargar(nombre=MODELO):
    tok = AutoTokenizer.from_pretrained(nombre)
    modelo = AutoModelForCausalLM.from_pretrained(nombre, dtype=getattr(torch, DTYPE))
    modelo.to(dispositivo())
    modelo.eval()
    assert tok.chat_template is not None, \
        "Se esperaba un modelo con formato de conversación; este no lo tiene"
    return tok, modelo


def con_formato(tok, texto):
    return tok.apply_chat_template([{"role": "user", "content": texto}],
                                   tokenize=False, add_generation_prompt=True)


def generar(tok, modelo, texto, pasos, temperatura=0.0, semilla=SEMILLA):
    """Genera `pasos` trozos. A temperatura 0 se elige siempre el favorito; por encima se
    sortea, y la semilla fija hace el sorteo reproducible."""
    ids = tok(texto, return_tensors="pt")["input_ids"].to(dispositivo())
    torch.manual_seed(semilla)
    with torch.no_grad():
        if temperatura == 0.0:
            salida = modelo.generate(ids, max_new_tokens=pasos, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        else:
            salida = modelo.generate(ids, max_new_tokens=pasos, do_sample=True,
                                     temperature=temperatura, top_k=0, top_p=1.0,
                                     pad_token_id=tok.eos_token_id)
    return tok.decode(salida[0, ids.shape[1]:], skip_special_tokens=True).strip()


def sorpresa(tok, modelo, prefijo, continuacion):
    """Cuánto le sorprende a la máquina esa continuación después de ese prefijo. Es la media
    del logaritmo negativo de la probabilidad que le da a cada trozo. Más alto, más sorpresa."""
    ids_p = tok(prefijo, return_tensors="pt")["input_ids"]
    ids_c = tok(continuacion, return_tensors="pt", add_special_tokens=False)["input_ids"]
    assert ids_c.shape[1] > 0, "Se esperaba una continuación no vacía; se encontró una vacía"
    ids = torch.cat([ids_p, ids_c], dim=1).to(dispositivo())
    with torch.no_grad():
        logits = modelo(ids).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    inicio = ids_p.shape[1]
    total = 0.0
    for k in range(inicio, ids.shape[1]):
        total += float(logp[k - 1, ids[0, k]])
    return -total / ids_c.shape[1]


def hay_bucle(texto):
    """Holtzman et al. 2019: una frase de al menos MINIMO_FRASE palabras repetida al menos
    VECES_BUCLE veces al FINAL del texto."""
    pal = texto.split()
    for n in range(MINIMO_FRASE, max(MINIMO_FRASE + 1, len(pal) // VECES_BUCLE + 1)):
        cola = pal[-n * VECES_BUCLE:]
        if len(cola) < n * VECES_BUCLE:
            continue
        if all(cola[i * n:(i + 1) * n] == cola[:n] for i in range(1, VECES_BUCLE)):
            return True
    return False


def se_abstiene(texto):
    t = texto.lower()
    return any(m in t for m in MARCAS_ABSTENCION)


def parrafos_del_corpus():
    """Va soltando los párrafos del corpus de uno en uno, sin cargarlo entero en memoria.
    El corpus son 132 MB: troceado de golpe en una lista se come varios gigabytes."""
    ficheros = sorted(glob.glob(os.path.join(CORPUS, "*.txt")))
    assert ficheros, f"Se esperaban libros en «{CORPUS}»; no se encontró ninguno"
    for f in ficheros:
        with open(f, encoding="utf-8", errors="strict") as fh:
            resto = ""
            while True:
                bloque = fh.read(1 << 20)
                if not bloque:
                    break
                partes = (resto + bloque).split("\n\n")
                resto = partes.pop()
                for p in partes:
                    p = p.strip().replace("\n", " ")
                    if p:
                        yield p
            resto = resto.strip().replace("\n", " ")
            if resto:
                yield resto


def es_castellano(p):
    t = " " + p.lower() + " "
    return (sum(m in t for m in MARCAS_ES) >= MINIMO_MARCAS_ES
            and sum(m in t for m in MARCAS_EN) <= 1)


def parrafos_humanos(n):
    """Párrafos largos, en castellano y limpios. Deterministas: se toma uno de cada SALTO de
    los que cumplen, en el orden en que aparecen."""
    elegidos, vistos = [], 0
    for p in parrafos_del_corpus():
        if len(p) > 1200 and p.count(".") > 5 and es_castellano(p):
            if vistos % SALTO_PARRAFOS == 0:
                elegidos.append(p)
                if len(elegidos) == n:
                    return elegidos
            vistos += 1
    raise AssertionError(
        f"Se esperaban {n} párrafos largos en castellano; se encontraron {len(elegidos)}")


def relleno(minimo):
    """Texto castellano corrido, para usar de pajar. Determinista."""
    partes, total = [], 0
    for p in parrafos_del_corpus():
        if len(p) > 300 and es_castellano(p):
            partes.append(p)
            total += len(p)
            if total >= minimo:
                return "\n\n".join(partes)
    raise AssertionError(
        f"Se esperaban {minimo} caracteres de relleno en castellano; se juntaron {total}")


# ----------------------------- las cuatro mediciones -----------------------------

def medicion_a(tok, modelo, filas):
    print(f"\n--- A. La temperatura: «{ENUNCIADO_A}» ---")
    texto = con_formato(tok, ENUNCIADO_A)
    for t in TEMPERATURAS:
        muestra = generar(tok, modelo, texto, PASOS_A, t)
        bucles = sum(hay_bucle(generar(tok, modelo, texto, PASOS_A, t, SEMILLA + k))
                     for k in range(MUESTRAS_REPETICION))
        tasa = bucles / MUESTRAS_REPETICION
        print(f"\n  temperatura {t:.1f}   (acaban en bucle: {tasa*100:.0f}%)")
        for linea in re.findall(r".{1,86}(?:\s|$)", muestra.replace("\n", " ")):
            print(f"    {linea.rstrip()}")
        filas.append(["A_temperatura", f"{t:.1f}", "bucles", f"{tasa:.3f}", "", ""])


def medicion_b(tok, modelo, filas):
    print("\n--- B. ¿Qué le sorprende más, un párrafo humano o uno suyo? ---")
    humanos = maquinas = 0.0
    for p in parrafos_humanos(MUESTRAS_B):
        ids = tok(p, return_tensors="pt")["input_ids"]
        assert ids.shape[1] > ARRANQUE + LARGO_B, "Párrafo demasiado corto para la comparación"
        prefijo = tok.decode(ids[0, :ARRANQUE])
        humano = tok.decode(ids[0, ARRANQUE:ARRANQUE + LARGO_B])
        maquina = generar(tok, modelo, prefijo, LARGO_B, 0.0)
        humanos += sorpresa(tok, modelo, prefijo, humano)
        maquinas += sorpresa(tok, modelo, prefijo, maquina)
    h, m = humanos / MUESTRAS_B, maquinas / MUESTRAS_B
    print(f"  sorpresa media ante el párrafo humano:     {h:.3f}")
    print(f"  sorpresa media ante el párrafo de la máquina: {m:.3f}")
    print(f"  el humano le sorprende {h/m:.1f} veces más")
    filas.append(["B_sorpresa", "humano", "", f"{h:.3f}", "", ""])
    filas.append(["B_sorpresa", "maquina", "", f"{m:.3f}", "", ""])


def pajar(tok, largo, profundidad):
    """Un texto de `largo` trozos con la aguja metida a esa profundidad."""
    ids = tok(relleno(largo * 12), return_tensors="pt",
              add_special_tokens=False)["input_ids"][0, :largo]
    assert ids.shape[0] == largo, \
        f"Se esperaban {largo} trozos de pajar; se obtuvieron {ids.shape[0]}"
    corte = int(round(profundidad * largo))
    return tok.decode(ids[:corte]) + "\n" + AGUJA + "\n" + tok.decode(ids[corte:])


def medicion_c(tok, modelo, filas):
    print("\n--- C. La aguja en el pajar ---")
    print(f"  {'largo del texto':<18}" + "".join(f"{int(d*100):>9}%" for d in PROFUNDIDADES))
    for largo in LONGITUDES:
        celdas = []
        try:
            for d in PROFUNDIDADES:
                texto = pajar(tok, largo, d) + "\n\n" + PREGUNTA_AGUJA
                r = generar(tok, modelo, con_formato(tok, texto), 20, 0.0)
                celdas.append(RESPUESTA_AGUJA in r)
                filas.append(["C_aguja", str(largo), f"{d:.2f}",
                              "1" if celdas[-1] else "0", "", ""])
        except (RuntimeError, MemoryError) as e:
            # Que un texto largo no quepa en memoria es un dato del capítulo, no un fallo del
            # guion: se anota con el error exacto y se sigue con las longitudes que sí caben.
            print(f"  {largo:<18} NO CABE: {type(e).__name__}: {str(e)[:90]}")
            filas.append(["C_aguja", str(largo), "no cabe", type(e).__name__, "", ""])
            continue
        print(f"  {largo:<18}" + "".join("       sí" if c else "       NO" for c in celdas))


def medicion_d(tok, modelo, filas):
    print("\n--- D. Veinte preguntas sin respuesta posible ---")
    resultados = {}
    for nombre, plantilla in (("enunciado normal", ENUNCIADO_NORMAL),
                              ("con umbral explícito", ENUNCIADO_KALAI)):
        abst = 0
        ejemplo = None
        for p in SIN_RESPUESTA:
            r = generar(tok, modelo, con_formato(tok, plantilla.format(p=p)), MAX_NUEVOS_D, 0.0)
            if se_abstiene(r):
                abst += 1
            elif ejemplo is None:
                ejemplo = (p, r)
        resultados[nombre] = abst / len(SIN_RESPUESTA)
        print(f"  {nombre:<22} se abstiene en {abst} de {len(SIN_RESPUESTA)} "
              f"({resultados[nombre]*100:.0f}%)")
        if ejemplo:
            print(f"    ejemplo de respuesta inventada: {ejemplo[0]}")
            print(f"    -> {ejemplo[1][:150]!r}")
        filas.append(["D_abstencion", nombre, "", f"{resultados[nombre]:.3f}", "", ""])
    return resultados


# ----------------------------- selftest -----------------------------

def selftest():
    fallos = []
    tok, modelo = cargar()

    # 1. TEST NULO — dos pruebas de que el montaje no regala el resultado.
    #    (a) La pregunta de la aguja SIN el pajar delante: la aguja es inventada, así que
    #        acertarla sin el texto significaría que el montaje filtra la respuesta.
    #    (b) El detector de abstención sobre cinco respuestas seguras: no debe saltar ninguna.
    r = generar(tok, modelo, con_formato(tok, PREGUNTA_AGUJA), 20, 0.0)
    filtra = RESPUESTA_AGUJA in r
    falsos = sum(se_abstiene(t) for t in CONTROL_RESPONDE)
    print(f"[1] test nulo         la aguja sin el pajar: {'ACERTADA' if filtra else 'no la sabe'}")
    print(f"                      detector sobre respuestas seguras: {falsos} de "
          f"{len(CONTROL_RESPONDE)} falsos positivos")
    if filtra:
        fallos.append("test nulo: acierta la aguja sin el texto delante; el montaje la filtra")
    if falsos:
        fallos.append(f"test nulo: el detector de abstención salta en {falsos} respuestas seguras")

    # 2. SEÑAL IMPLANTADA — dos pruebas de que sí se detecta lo que hay que detectar.
    #    (a) La aguja en un pajar corto: tiene que encontrarla.
    #    (b) El detector sobre cinco negativas explícitas: tiene que saltar en las cinco.
    texto = pajar(tok, LARGO_CORTO, 0.5) + "\n\n" + PREGUNTA_AGUJA
    r = generar(tok, modelo, con_formato(tok, texto), 20, 0.0)
    corto_ok = RESPUESTA_AGUJA in r
    aciertos = sum(se_abstiene(t) for t in CONTROL_ABSTIENE)
    print(f"[2] señal implantada  la aguja en {LARGO_CORTO} trozos: "
          f"{'encontrada' if corto_ok else 'NO ENCONTRADA'}")
    print(f"                      detector sobre negativas explícitas: {aciertos} de "
          f"{len(CONTROL_ABSTIENE)}")
    if not corto_ok:
        fallos.append(f"señal implantada: no encuentra la aguja en {LARGO_CORTO} trozos; "
                      f"contestó {r[:60]!r}")
    if aciertos != len(CONTROL_ABSTIENE):
        fallos.append(f"señal implantada: el detector solo ve {aciertos} de "
                      f"{len(CONTROL_ABSTIENE)} negativas explícitas")

    # 3. INVARIANTE DEL DOMINIO — a temperatura 0 no hay sorteo: dos ejecuciones idénticas.
    #    Y la sorpresa de un texto ante sí mismo tiene que ser un número finito y positivo.
    a = generar(tok, modelo, con_formato(tok, ENUNCIADO_A), 30, 0.0)
    b = generar(tok, modelo, con_formato(tok, ENUNCIADO_A), 30, 0.0)
    s = sorpresa(tok, modelo, "El faro estaba", " abandonado desde hacía años.")
    print(f"[3] invariante        dos ejecuciones a temperatura 0 idénticas: "
          f"{'sí' if a == b else 'NO'}")
    print(f"                      la sorpresa es un número positivo y finito: {s:.3f}")
    if a != b:
        fallos.append(f"invariante: dos ejecuciones distintas a temperatura 0: {a[:40]!r} / {b[:40]!r}")
    if not (0.0 < s < 100.0):
        fallos.append(f"invariante: la sorpresa vale {s}, fuera de todo rango razonable")

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
    if args.selftest:
        sys.exit(selftest())

    print(f"modelo: {MODELO}   dispositivo: {dispositivo()}")
    tok, modelo = cargar()
    filas = []

    medicion_a(tok, modelo, filas)
    medicion_b(tok, modelo, filas)
    medicion_c(tok, modelo, filas)
    medicion_d(tok, modelo, filas)

    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["medicion", "clave", "sub", "a", "b", "c"]] + filas)
    print(f"\nEscrito {SALIDA_CSV}")


if __name__ == "__main__":
    main()
