#!/usr/bin/env python3
"""
Capítulo 1 — la escalera, paso a paso.

La sección «La escalera» enseña lo que escribe la máquina de `ngrama.py` con la perilla en cero,
una, dos, tres y cinco letras. El lector ve el resultado de doscientos sorteos por peldaño y no
los pasos que llevan a él (L11 y T27 de `notas/auditoria/PENDIENTES.md`). Este programa imprime
esos pasos, con las mismas muestras:

  1. cómo arranca cada muestra: las primeras letras no se sortean;
  2. un peldaño hecho a mano: los primeros sorteos de la muestra de dos letras, casilla a casilla;
  3. por qué al subir salen palabras: cuantas más letras mira la máquina, menos letras distintas
     caben en la urna de cada casilla;
  4. cuántas palabras del Quijote hay en cada muestra.

Las muestras NO se escriben aquí: se repiten los sorteos de `ngrama.py` con su misma semilla, en
su mismo orden, y el selftest revienta si no salen idénticas a las de `datos/salidas/ngrama.txt`.

Uso:
    python traza_escalera.py --selftest
    python traza_escalera.py > ../datos/salidas/traza_escalera.txt
"""

# ======================= CONSTANTES =======================

MUESTRAS = "../datos/salidas/ngrama.txt"   # lo que el libro imprime, tal como salió

PELDANO_A_MANO = 2        # la perilla del peldaño que se sigue sorteo a sorteo
SORTEOS_A_MANO = 6        # cuántos sorteos se enseñan
LETRAS_POR_URNA = 4       # cuántas letras de cada urna se enseñan una a una; el resto se agrupa

# El mismo momento de escribir, visto con la perilla en cada posición: la máquina acaba de
# escribir «quijo» y va a sortear la letra siguiente. Con la perilla en uno solo ve «o»; en dos,
# «jo»; en tres, «ijo»; en cinco, «quijo». Es la misma letra por sortear, con más o menos pasado.
MOMENTO = "quijo"

# Selftest
TROZO_NULO = 400_000              # letras del corpus con las que se hace el test nulo
PERILLA_NULA = 2
COCIENTE_NULO_MIN = 0.75          # barajado: la urna de dos letras casi tan llena como la de cero
COCIENTE_REAL_MAX = 0.65          # real: la urna de dos letras claramente más vacía
MARCA = "wñk"                     # señal implantada: una palabra que no existe (ni «wñ» en el libro)
REPETICIONES_MARCA = 300

# ==========================================================

import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import ngrama as N
from formato import ANCHO_CAJA, coma, comprobar_ancho, miles, pct

AQUI = Path(__file__).resolve().parent


def rotulo(orden):
    """El rótulo con el que `ngrama.py` encabeza cada muestra de letras."""
    if orden == 0:
        return "--- LETRAS, azar puro ---"
    return f"--- LETRAS, {orden} letra{'s' if orden > 1 else ''} de contexto ---"


def perilla(orden):
    return {0: "nada", 1: "1 letra"}.get(orden, f"{orden} letras")


def leer_muestras(ruta):
    """Las muestras de letras que imprimió `ngrama.py`, por orden."""
    lineas = Path(ruta).read_text(encoding="utf-8").split("\n")
    muestras = {}
    for orden in N.ORDENES_LETRA:
        r = rotulo(orden)
        assert r in lineas, f"Se esperaba el rótulo «{r}» en {ruta}; no está"
        muestras[orden] = lineas[lineas.index(r) + 1]
    return muestras


def generar_con_traza(tabla, orden, largo, rng):
    """Lo mismo que `ngrama.generar`, llamando al sorteo en el mismo orden, pero apuntando cada
    paso: con qué contexto, qué había en la urna y qué salió. Si no gasta el azar exactamente
    igual, las muestras salen distintas y el selftest lo dice."""
    if orden == 0:
        u = tabla[()]
        salida = [N.elegir(u, rng) for _ in range(largo)]
        return "".join(salida), None, [((), u, s) for s in salida]
    contextos = list(tabla.keys())
    arranque = rng.choice(contextos)
    estado, salida, pasos = list(arranque), list(arranque), []
    for _ in range(largo - orden):
        clave = tuple(estado[-orden:])
        assert clave in tabla, (
            f"Se esperaba que todo contexto escrito estuviera en la tabla (lo que sigue a un "
            f"contexto del libro también está en el libro); no está «{''.join(clave)}»")
        siguiente = N.elegir(tabla[clave], rng)
        pasos.append((clave, tabla[clave], siguiente))
        salida.append(siguiente)
        estado.append(siguiente)
    return "".join(salida), "".join(arranque), pasos


def repetir_sorteos(secuencia):
    """Las cinco muestras de letras, en el orden y con la semilla de `ngrama.py`."""
    rng = N.random.Random(N.SEMILLA)
    resultado = {}
    for orden in N.ORDENES_LETRA:
        tabla = N.construir(secuencia, orden)
        texto, arranque, pasos = generar_con_traza(tabla, orden, N.LARGO_MUESTRA_LETRAS, rng)
        resultado[orden] = (tabla, texto, arranque, pasos)
    return resultado


def letras_por_urna(tabla):
    """De media, cada vez que la máquina sortea, cuántas letras distintas hay en la urna. Cada
    casilla pesa tantas veces como se usa: una casilla que sale mil veces cuenta mil."""
    usos = sum(sum(c.values()) for c in tabla.values())
    return sum(len(c) * sum(c.values()) for c in tabla.values()) / usos


def letra(s):
    return "espacio" if s == " " else f"«{s}»"


def urna(contador, cuantas):
    """Las `cuantas` letras más frecuentes de una casilla, con su parte de cada cien, y el resto
    agrupado. Devuelve líneas."""
    total = sum(contador.values())
    orden = contador.most_common()
    partes = [f"{letra(s)} {pct(n / total, 1)}" for s, n in orden[:cuantas]]
    resto = orden[cuantas:]
    if resto:
        n = sum(v for _, v in resto)
        otras = "otra letra" if len(resto) == 1 else f"otras {len(resto)} letras"
        partes.append(f"{otras}: {pct(n / total, 1)}")
    lineas, fila = [], "      "
    for p in partes:
        if len(fila) + len(p) + 3 > ANCHO_CAJA:
            lineas.append(fila.rstrip(" ·"))
            fila = "      "
        fila += p + " · "
    lineas.append(fila.rstrip(" ·"))
    return lineas


def palabras_reales(muestra, vocabulario):
    """Igual que `casillas_vacias.palabras_reales`: los trozos entre espacios, sin los signos
    pegados, y cuáles están en el Quijote."""
    trozos = [t.strip(N.SIGNOS) for t in muestra.split() if t.strip(N.SIGNOS)]
    return trozos, [t for t in trozos if t in vocabulario]


# ---- selftest ------------------------------------------------------------------------

def selftest(secuencia, muestras_libro, vocabulario):
    fallos = []

    # 1. TEST NULO — con las letras barajadas no hay castellano que estreche las urnas: mirar dos
    #    letras atrás deja la urna casi tan llena como no mirar nada. En el texto real, no.
    barajado = list(secuencia[:TROZO_NULO])
    random.Random(N.SEMILLA).shuffle(barajado)
    real = secuencia[:TROZO_NULO]
    c_nulo = letras_por_urna(N.construir(barajado, PERILLA_NULA)) / letras_por_urna(N.construir(barajado, 0))
    c_real = letras_por_urna(N.construir(real, PERILLA_NULA)) / letras_por_urna(N.construir(real, 0))
    print(f"[1] test nulo         urna con {PERILLA_NULA} letras / urna sin mirar: "
          f"barajado={coma(c_nulo, 2)}  real={coma(c_real, 2)}")
    if not (c_nulo > COCIENTE_NULO_MIN and c_real < COCIENTE_REAL_MAX):
        fallos.append(f"test nulo: se esperaba barajado > {COCIENTE_NULO_MIN} y real < "
                      f"{COCIENTE_REAL_MAX}; salió barajado={c_nulo:.3f}, real={c_real:.3f}")

    # 2. SEÑAL IMPLANTADA — una palabra inventada, repetida, deja una casilla con una sola letra,
    #    y exactamente tantas veces como se metió.
    implantado = secuencia[:200_000] + (" " + MARCA) * REPETICIONES_MARCA
    casilla = N.construir(implantado, 2)[tuple(MARCA[:2])]
    print(f"[2] señal implantada  casilla «{MARCA[:2]}»: {dict(casilla)}")
    if dict(casilla) != {MARCA[2]: REPETICIONES_MARCA}:
        fallos.append(f"señal implantada: se esperaba la casilla «{MARCA[:2]}» = "
                      f"{{'{MARCA[2]}': {REPETICIONES_MARCA}}}; salió {dict(casilla)}")

    # 3. INVARIANTES — los sorteos repetidos dan las muestras del libro, letra por letra; cada
    #    letra que sale estaba en su urna; y lo que se cuenta de las palabras cuadra.
    repetidas = repetir_sorteos(secuencia)
    distintas = [o for o in N.ORDENES_LETRA if repetidas[o][1].strip() != muestras_libro[o].strip()]
    fuera = sum(1 for o in N.ORDENES_LETRA for _, u, s in repetidas[o][3] if s not in u)
    print(f"[3] invariante        muestras repetidas iguales a las de ngrama.txt: "
          f"{len(N.ORDENES_LETRA) - len(distintas)} de {len(N.ORDENES_LETRA)}; "
          f"letras sorteadas que no estaban en su urna: {fuera}")
    if distintas:
        fallos.append(f"invariante: las muestras de las perillas {distintas} no salen iguales "
                      f"que en {MUESTRAS}; este programa ya no repite los sorteos de ngrama.py")
    if fuera:
        fallos.append(f"invariante: {fuera} letras sorteadas no estaban en su urna")
    trozos, reales = palabras_reales(muestras_libro[1], vocabulario)
    print(f"[3] invariante        muestra de 1 letra: {len(reales)} palabras del Quijote en "
          f"{len(trozos)} trozos (casillas_vacias.txt dice 12 en 40)")
    if (len(reales), len(trozos)) != (12, 40):
        fallos.append(f"invariante: se esperaban 12 palabras en 40 trozos, como en "
                      f"casillas_vacias.txt; salieron {len(reales)} en {len(trozos)}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1, repetidas
    print("SELFTEST: las tres pruebas pasan.")
    return 0, repetidas


# ---- la salida -------------------------------------------------------------------------

def main():
    secuencia = N.normalizar(N.cargar_corpus(str(AQUI / N.CORPUS)), N.ALFABETO)
    vocabulario = set(N.solo_palabras(N.trocear(secuencia)))
    muestras_libro = leer_muestras(AQUI / MUESTRAS)

    print("--- selftest ---")
    codigo, repetidas = selftest(secuencia, muestras_libro, vocabulario)
    if codigo or "--selftest" in sys.argv:
        sys.exit(codigo)

    L = [
        "",
        "########## capítulo 1: la escalera, paso a paso ##########",
        f"fecha: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        f"máquina: {platform.system()} {platform.machine()}. La medición no",
        "depende de la máquina: repite los sorteos de ngrama.py con su",
        f"semilla ({N.SEMILLA}) y sale lo mismo en cualquier ordenador",
        "",
        "1. CÓMO ARRANCA CADA MUESTRA",
        "",
        "Las primeras letras no se sortean: la máquina coge al azar un",
        "contexto de la tabla, lo escribe, y empieza a sortear detrás.",
        "",
        f"  {'perilla':<10}{'arranque':<12}principio de la muestra",
        f"  {'-------':<10}{'--------':<12}-----------------------",
    ]
    for orden in N.ORDENES_LETRA:
        _, texto, arranque, _ = repetidas[orden]
        a = "(ninguno)" if arranque is None else f"«{arranque}»"
        L.append(f"  {perilla(orden):<10}{a:<12}{texto[:24]}…")

    tabla, texto, arranque, pasos = repetidas[PELDANO_A_MANO]
    L += [
        "",
        f"2. UN PELDAÑO A MANO: PERILLA EN {PELDANO_A_MANO}, LOS {SORTEOS_A_MANO} PRIMEROS SORTEOS",
        "",
        f"Arranque: «{arranque}». Detrás, cada sorteo mira las {PELDANO_A_MANO} últimas letras,",
        "va a su casilla y saca una papeleta de su urna.",
    ]
    escrito = arranque
    for clave, u, sale in pasos[:SORTEOS_A_MANO]:
        total = sum(u.values())
        L += ["", f"  escrito «{escrito}»: mira «{''.join(clave)}»",
              f"      urna de «{''.join(clave)}»: {miles(total)} papeletas, {len(u)} letras distintas"]
        L += urna(u, LETRAS_POR_URNA)
        L.append(f"      sale: {letra(sale)}")
        escrito += sale
    L.append(f"\n  escrito al final: «{escrito}»")

    L += [
        "",
        "3. LAS URNAS SE ESTRECHAN",
        "",
        "De media, cada vez que la máquina sortea, cuántas letras",
        "distintas hay en la urna de la casilla que le toca.",
        "",
        f"  {'perilla':<10}{'casillas':>10}{'letras distintas en la urna':>32}",
        f"  {'-------':<10}{'--------':>10}{'---------------------------':>32}",
    ]
    for orden in N.ORDENES_LETRA:
        t = repetidas[orden][0]
        L.append(f"  {perilla(orden):<10}{miles(len(t)):>10}{coma(letras_por_urna(t), 1):>32}")
    L += [
        "",
        f"El mismo momento, con la perilla en cada posición: la máquina",
        f"ha escrito «{MOMENTO}» y va a sortear la letra siguiente.",
    ]
    for orden in N.ORDENES_LETRA:
        t = repetidas[orden][0]
        clave = tuple(MOMENTO[len(MOMENTO) - orden:]) if orden else ()
        u = t[clave]
        visto = "no mira nada" if orden == 0 else f"ve «{''.join(clave)}»"
        L += ["", f"  perilla {perilla(orden)}: {visto} · {len(u)} letras en la urna"]
        L += urna(u, 3)

    L += [
        "",
        "4. PALABRAS DEL QUIJOTE EN CADA MUESTRA",
        "",
        "Trozos entre espacios, sin los signos pegados, que son",
        "palabras que están en el Quijote.",
        "",
        f"  {'perilla':<10}{'trozos':>8}{'palabras del Quijote':>24}{'de cada cien':>16}",
        f"  {'-------':<10}{'------':>8}{'--------------------':>24}{'------------':>16}",
    ]
    for orden in N.ORDENES_LETRA:
        trozos, reales = palabras_reales(muestras_libro[orden], vocabulario)
        L.append(f"  {perilla(orden):<10}{len(trozos):>8}{len(reales):>24}"
                 f"{pct(len(reales) / len(trozos), 0):>16}")

    for l in comprobar_ancho("\n".join(L).split("\n"), ANCHO_CAJA):
        print(l)


if __name__ == "__main__":
    main()
