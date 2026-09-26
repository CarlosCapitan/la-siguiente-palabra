#!/usr/bin/env python3
"""
Capítulo 1 — la escalera, paso a paso.

La sección «La escalera» enseña lo que escribe la máquina de `ngrama.py` con la perilla en cero,
una, dos, tres y cinco letras. El lector ve el resultado de doscientos sorteos por peldaño y no
los pasos que llevan a él (L11 y T27 de `notas/auditoria/PENDIENTES.md`). Este programa imprime
esos pasos, con las mismas muestras:

  1. cómo arranca cada muestra: las primeras letras no salen de un sorteo;
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
SORTEOS_PALABRAS = 4      # cuántos sorteos de la máquina de palabras (perilla en dos) se enseñan
LETRAS_POR_URNA = 4       # cuántas letras de cada urna se enseñan una a una; el resto se agrupa

# El mismo momento de escribir, visto con la perilla en cada posición: la máquina acaba de
# escribir «quijo» y va a sacar en un sorteo la letra siguiente. Con la perilla en uno solo ve «o»; en dos,
# «jo»; en tres, «ijo»; en cinco, «quijo». Es la misma letra por salir, con más o menos pasado.
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


def rotulo_palabras(orden):
    """El rótulo con el que `ngrama.py` encabeza cada muestra de palabras."""
    return ("--- PALABRAS, frecuencias sueltas ---" if orden == 1
            else f"--- PALABRAS, {orden} palabras de contexto ---")


def leer_muestras(ruta):
    """Las muestras que imprimió `ngrama.py`: las de letras con la clave del orden, y las de
    palabras con la clave ("palabras", orden)."""
    lineas = Path(ruta).read_text(encoding="utf-8").split("\n")
    muestras = {}
    claves = [(o, rotulo(o)) for o in N.ORDENES_LETRA]
    claves += [(("palabras", o), rotulo_palabras(o)) for o in N.ORDENES_PALABRA]
    for orden, r in claves:
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
        return salida, None, [((), u, s) for s in salida]
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
    return salida, arranque, pasos


def repetir_sorteos(secuencia):
    """Las muestras de letras y luego las de palabras, en el orden y con la semilla de
    `ngrama.py`: las de palabras gastan el azar que dejan las de letras, así que no se pueden
    repetir sin repetir antes aquellas."""
    rng = N.random.Random(N.SEMILLA)
    resultado = {}
    for orden in N.ORDENES_LETRA:
        tabla = N.construir(secuencia, orden)
        salida, arranque, pasos = generar_con_traza(tabla, orden, N.LARGO_MUESTRA_LETRAS, rng)
        resultado[orden] = (tabla, "".join(salida),
                            None if arranque is None else "".join(arranque), pasos)
    palabras = N.trocear(secuencia)
    for orden in N.ORDENES_PALABRA:
        tabla = N.construir(palabras, orden)
        salida, arranque, pasos = generar_con_traza(tabla, orden, N.LARGO_MUESTRA_PALABRAS, rng)
        resultado[("palabras", orden)] = (tabla, " ".join(salida), arranque, pasos)
    return resultado


def letras_por_urna(tabla):
    """De media, cada vez que la máquina hace un sorteo, cuántas letras distintas pueden venir en su casilla. Cada
    casilla pesa tantas veces como se usa: una casilla que sale mil veces cuenta mil."""
    usos = sum(sum(c.values()) for c in tabla.values())
    return sum(len(c) * sum(c.values()) for c in tabla.values()) / usos


def cuantas(n, singular, plural):
    """«1 papeleta», «24 papeletas»: el número con su nombre bien concordado."""
    return f"{miles(n)} {singular if n == 1 else plural}"


def letra(s):
    return "espacio" if s == " " else f"«{s}»"


def nombre_simbolo(s):
    """Cómo se escribe una letra en la columna de una tabla: el espacio, con su nombre."""
    return "espacio" if s == " " else s


def la_que_mas(contador, palabra=False):
    """La letra (o palabra) que más sale en una casilla, con cuántas veces de cada cien. Si
    empatan varias en cabeza, lo dice, porque nombrar una sola sería elegir por el lector."""
    total = sum(contador.values())
    orden = contador.most_common()
    arriba = [s for s, n in orden if n == orden[0][1]]
    parte = pct(orden[0][1] / total, 0)
    if len(arriba) > 1:
        return f"{len(arriba)} empatadas, {parte}"
    s = arriba[0]
    return f"{'«' + s + '»' if palabra else letra(s)}, {parte}"


def urna(contador, cuantas, unidad="letra"):
    """Las `cuantas` letras más frecuentes de una casilla, con su parte de cada cien, y el resto
    agrupado. Devuelve líneas."""
    total = sum(contador.values())
    orden = contador.most_common()
    nombre = letra if unidad == "letra" else (lambda s: f"«{s}»")
    partes = [f"{nombre(s)} {pct(n / total, 1)}" for s, n in orden[:cuantas]]
    resto = orden[cuantas:]
    if resto:
        n = sum(v for _, v in resto)
        otras = (f"otra {unidad}" if len(resto) == 1
                 else f"otras {miles(len(resto))} {unidad}s")
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
    print(f"[1] test nulo         letras posibles mirando {PERILLA_NULA} / sin mirar: "
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
    todas = list(muestras_libro)
    distintas = [o for o in todas if repetidas[o][1].strip() != muestras_libro[o].strip()]
    fuera = sum(1 for o in todas for _, u, s in repetidas[o][3] if s not in u)
    print(f"[3] invariante        muestras repetidas iguales a las de ngrama.txt: "
          f"{len(todas) - len(distintas)} de {len(todas)}; "
          f"letras que salieron sin estar en su casilla: {fuera}")
    if distintas:
        fallos.append(f"invariante: las muestras de las perillas {distintas} no salen iguales "
                      f"que en {MUESTRAS}; este programa ya no repite los sorteos de ngrama.py")
    if fuera:
        fallos.append(f"invariante: {fuera} letras salieron sin estar en su casilla")
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
        "Las primeras letras no salen de un sorteo: la máquina coge al azar",
        "un contexto de la tabla, lo escribe, y detrás empiezan los sorteos.",
        "",
        f"  {'perilla':<10}{'arranque':<12}principio de la muestra",
        f"  {'-------':<10}{'--------':<12}-----------------------",
    ]
    for orden in N.ORDENES_LETRA:
        _, texto, arranque, _ = repetidas[orden]
        a = "(ninguno)" if arranque is None else f"«{arranque}»"
        L.append(f"  {perilla(orden):<10}{a:<12}{texto[:24]}…")

    tabla, texto, arranque, pasos = repetidas[PELDANO_A_MANO]
    clave, u, sale = pasos[0]
    total = sum(u.values())
    L += [
        "",
        f"2. EL PRIMER SORTEO DE LA PERILLA EN {PELDANO_A_MANO}, ENTERO",
        "",
        f"La máquina arranca con «{arranque}» y va a la casilla de «{''.join(clave)}»:",
        f"las {miles(total)} veces que aparece «{''.join(clave)}» en el Quijote",
        "",
        "  letra que sigue    veces    de cada cien veces",
        "  ---------------   ------   -------------------",
    ]
    for s, n in u.most_common():
        L.append(f"  {nombre_simbolo(s):^15}   {miles(n):>6}   {pct(n / total, 2):>19}")
    L += ["", f"  sale: {letra(sale)}. Escrito: «{arranque + sale}»"]

    L += [
        "",
        f"3. LOS {SORTEOS_A_MANO} PRIMEROS SORTEOS, UNO POR FILA",
        "",
        "mira: las dos últimas letras escritas, que dicen a qué casilla va",
        "pueden venir: cuántas letras distintas hay en esa casilla",
        "la que más sale: y cuántas veces de cada cien",
        "",
        f"  {'ha escrito':<12}{'mira':<8}{'pueden venir':>12}   {'la que más sale':<18}{'sale':<9}",
        f"  {'----------':<12}{'----':<8}{'------------':>12}   {'---------------':<18}{'----':<9}",
    ]
    escrito = arranque
    for clave, u, sale in pasos[:SORTEOS_A_MANO]:
        L.append(f"  {'«' + escrito + '»':<12}{'«' + ''.join(clave) + '»':<8}{len(u):>12}   "
                 f"{la_que_mas(u):<18}{letra(sale):<9}")
        escrito += sale
    L += ["", f"  escrito al final: «{escrito}»"]

    tabla, texto, arranque, pasos = repetidas[("palabras", 2)]
    L += [
        "",
        f"3 BIS. LA MÁQUINA DE PALABRAS, PERILLA EN 2: LOS {SORTEOS_PALABRAS} PRIMEROS SORTEOS",
        "",
        f"Arranque: «{' '.join(arranque)}». Cada sorteo mira las dos últimas",
        "palabras; «pueden venir» cuenta palabras distintas.",
        "",
        f"  {'mira':<16}{'veces':>6}{'pueden venir':>13}  {'la que más sale':<18}{'sale'}",
        f"  {'----':<16}{'-----':>6}{'------------':>13}  {'---------------':<18}{'----'}",
    ]
    escrito = list(arranque)
    for clave, u, sale in pasos[:SORTEOS_PALABRAS]:
        L.append(f"  {'«' + ' '.join(clave) + '»':<16}{miles(sum(u.values())):>6}{len(u):>13}  "
                 f"{la_que_mas(u, palabra=True):<18}{'«' + sale + '»'}")
        escrito.append(sale)
    L += ["", f"  escrito al final: «{' '.join(escrito)}»"]

    L += [
        "",
        "4. POR QUÉ AL SUBIR SALEN PALABRAS",
        "",
        f"La máquina ha escrito «{MOMENTO}» y tiene que sacar en un sorteo",
        "la letra siguiente. Lo que mira depende de la perilla.",
        "",
        f"  {'perilla':<10}{'mira':<10}{'pueden venir':>14}{'la «t», de cada cien':>24}",
        f"  {'-------':<10}{'----':<10}{'------------':>14}{'--------------------':>24}",
    ]
    for orden in N.ORDENES_LETRA:
        t = repetidas[orden][0]
        clave = tuple(MOMENTO[len(MOMENTO) - orden:]) if orden else ()
        u = t[clave]
        mira = "nada" if orden == 0 else "«" + "".join(clave) + "»"
        L.append(f"  {perilla(orden):<10}{mira:<10}{len(u):>14}"
                 f"{coma(100 * u['t'] / sum(u.values()), 1):>24}")
    L += [
        "",
        "Y de media, en todos los sorteos: cuántas letras distintas",
        "pueden venir en la casilla que le toca a la máquina.",
        "",
        f"  {'perilla':<10}{'casillas':>10}{'pueden venir, de media':>28}",
        f"  {'-------':<10}{'--------':>10}{'----------------------':>28}",
    ]
    for orden in N.ORDENES_LETRA:
        t = repetidas[orden][0]
        L.append(f"  {perilla(orden):<10}{miles(len(t)):>10}{coma(letras_por_urna(t), 1):>28}")

    L += [
        "",
        "5. PALABRAS DEL QUIJOTE EN CADA MUESTRA",
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
    _, reales = palabras_reales(muestras_libro[0], vocabulario)
    L += ["", "  las de la perilla en nada: " + " ".join(f"«{r}»" for r in reales)]

    for l in comprobar_ancho([l.rstrip() for l in "\n".join(L).split("\n")], ANCHO_CAJA):
        print(l)


if __name__ == "__main__":
    main()
