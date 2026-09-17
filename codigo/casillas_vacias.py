#!/usr/bin/env python3
"""Las cuentas del capítulo 1, medidas en vez de supuestas.

El capítulo 1 hace seis afirmaciones con número y ninguna tenía un programa detrás
(regla 1 bis). `ngrama.py` produce las muestras que el capítulo enseña, pero no cuenta
nada de esto. Aquí se cuenta, sobre el mismo Quijote y con la misma limpieza que usa
`ngrama.py`, para que los dos hablen del mismo texto:

  (a) «unos dos millones de letras, trescientas ochenta y cinco mil palabras»
  (b) «después de "qu" vino una "e" tantas miles de veces, una "i" tantas, y nada más,
      porque en castellano detrás de "qu" no va otra cosa»
  (c) «En el Quijote hay treinta y seis mil palabras distintas»
  (d) «las casillas posibles son treinta y seis mil por treinta y seis mil: mil
      trescientos millones. ¿Cuántas de ellas aparecen realmente en el Quijote?
      Ciento setenta y ocho mil. El cero coma cero uno por ciento.»
  (e) «de esas ciento setenta y ocho mil combinaciones que sí aparecen, ciento cuarenta
      mil aparecen una sola vez en todo el libro. Casi cuatro de cada cinco.»
  (f) «harían falta unos cuarenta y siete billones de casillas»

Y una séptima, que es la única del capítulo que se podía comprobar sobre la propia
muestra impresa:

  (g) «Han aparecido palabras reales. "Ya", "y", "de", "cue".»

      La muestra no se vuelve a generar aquí: se lee de `datos/salidas/ngrama.txt`, que
      es la que está impresa en el libro. Medir sobre otra muestra sería medir otra cosa.

POR QUÉ ESTA MEDICIÓN NO DEPENDE DE LA MÁQUINA. Se cuentan letras y palabras de un
fichero de texto. El resultado es el mismo en cualquier ordenador y en cualquier día;
lo único que lo cambiaría es cambiar el corpus.

Uso:
    python casillas_vacias.py
    python casillas_vacias.py --selftest
"""

# ======================= CONSTANTES =======================

CORPUS = "../datos/quijote.txt"
MUESTRA = "../datos/salidas/ngrama.txt"
ROTULO_MUESTRA = "--- LETRAS, 1 letra de contexto ---"

PAR_VIGILADO = "qu"          # el par de letras que el capítulo dice que solo admite dos
SIGNOS = ",.;:¿?¡!"          # lo que se quita del borde de una palabra antes de buscarla

# ==========================================================

import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from formato import ANCHO_CAJA, coma, comprobar_ancho, miles, pct
from ngrama import ALFABETO, cargar_corpus, normalizar

AQUI = Path(__file__).resolve().parent


# ---- cada cuenta en una función que se puede probar sola ---------------------------

def tras(secuencia, par):
    """Qué letra viene detrás de cada aparición de `par`, y cuántas veces cada una."""
    ancho = len(par)
    cuenta = Counter()
    for i in range(len(secuencia) - ancho):
        if secuencia[i:i + ancho] == par:
            cuenta[secuencia[i + ancho]] += 1
    return cuenta


def parejas(palabras):
    """Cada pareja de palabras seguidas, y cuántas veces aparece."""
    return Counter(zip(palabras, palabras[1:]))


def casillas(vocabulario, palabras_de_contexto):
    """Casillas que tendría la tabla si se miran esas palabras hacia atrás.

    Una casilla por cada combinación posible de palabras de contexto; dentro de cada
    casilla van los recuentos de lo que vino después. Es el vocabulario elevado al
    número de palabras que se miran, que es como las cuenta el capítulo: mirando una
    palabra hay tantas casillas como palabras distintas, y mirando dos hay ese número
    al cuadrado."""
    return vocabulario ** palabras_de_contexto


def de_una_sola_vez(cuenta):
    """Cuántas de las cosas contadas aparecieron exactamente una vez."""
    return sum(1 for n in cuenta.values() if n == 1)


def leer_muestra(ruta, rotulo):
    """La muestra impresa en el libro, tal como la dejó `ngrama.py`."""
    lineas = Path(ruta).read_text(encoding="utf-8").split("\n")
    i = lineas.index(rotulo)
    return lineas[i + 1].strip()


def palabras_reales(muestra, vocabulario):
    """Los trozos entre espacios de la muestra que son palabras del Quijote."""
    trozos = [t.strip(SIGNOS) for t in muestra.split() if t.strip(SIGNOS)]
    return trozos, [t for t in trozos if t in vocabulario]


# ---- selftest de tres partes -------------------------------------------------------

def selftest(secuencia=None, palabras=None):
    print("--- selftest ---")

    # [1] TEST NULO — el mismo texto con las letras barajadas y con las palabras
    #     barajadas. Ahí no queda ninguna regla del castellano. Si estas cuentas no
    #     distinguen el Quijote de su propia baraja, no están midiendo nada.
    import random
    baraja_letras = list(secuencia[:400_000])
    random.Random(1).shuffle(baraja_letras)
    tras_barajado = tras("".join(baraja_letras), PAR_VIGILADO)
    tras_real = tras(secuencia[:400_000], PAR_VIGILADO)

    baraja_palabras = list(palabras[:200_000])
    random.Random(1).shuffle(baraja_palabras)
    hapax_barajado = de_una_sola_vez(parejas(baraja_palabras)) / len(parejas(baraja_palabras))
    hapax_real = de_una_sola_vez(parejas(palabras[:200_000])) / len(parejas(palabras[:200_000]))

    print(f"[1] test nulo         letras distintas tras «{PAR_VIGILADO}»: "
          f"barajado={len(tras_barajado)}  real={len(tras_real)}; "
          f"parejas de una sola vez: barajado={pct(hapax_barajado, 0)} "
          f"real={pct(hapax_real, 0)}")
    assert len(tras_barajado) > 2 * len(tras_real), (len(tras_barajado), len(tras_real))
    assert hapax_barajado > hapax_real, (hapax_barajado, hapax_real)

    # [2] SEÑAL IMPLANTADA — un texto diminuto cuyas respuestas se cuentan con el dedo.
    #     «a b a b a c»: parejas ab, ba, ab, ba, ac -> 3 distintas, de ellas 1 una sola
    #     vez; vocabulario de 3 palabras -> 9 casillas posibles. Y «quxquxqux»: detrás
    #     de «qu» solo hay una letra, tres veces.
    mini = "a b a b a c".split()
    p = parejas(mini)
    assert len(p) == 3, p
    assert de_una_sola_vez(p) == 1, p
    assert casillas(len(set(mini)), 1) == 3, casillas(len(set(mini)), 1)
    assert casillas(len(set(mini)), 2) == 9, casillas(len(set(mini)), 2)
    assert casillas(len(set(mini)), 3) == 27, casillas(len(set(mini)), 3)
    t = tras("quxquxqux", PAR_VIGILADO)
    assert t == Counter({"x": 3}), t
    trozos, reales = palabras_reales("hola, xqz hola", {"hola"})
    assert (len(trozos), len(reales)) == (3, 2), (trozos, reales)
    print("[2] señal implantada  texto de seis palabras: 3 parejas, 1 de una sola vez, "
          "9 casillas mirando dos palabras; «quxquxqux»: una letra detrás, 3 veces")

    # [3] INVARIANTE DEL DOMINIO — cosas que tienen que ser verdad pase lo que pase,
    #     comprobadas sobre el Quijote entero.
    p = parejas(palabras)
    vocabulario = len(set(palabras))
    t = tras(secuencia, PAR_VIGILADO)
    assert sum(p.values()) == len(palabras) - 1, (sum(p.values()), len(palabras))
    assert len(p) <= casillas(vocabulario, 2), (len(p), vocabulario)
    assert de_una_sola_vez(p) <= len(p)
    assert set(t) <= set(ALFABETO), sorted(set(t) - set(ALFABETO))
    assert casillas(vocabulario, 3) == casillas(vocabulario, 2) * vocabulario
    print(f"[3] invariante        parejas contadas = palabras menos una ({miles(sum(p.values()))}); "
          f"parejas vistas <= casillas; todo lo que va tras «{PAR_VIGILADO}» está en el alfabeto")

    print("\nSELFTEST: las tres pruebas pasan.\n")


# ---- lo que se imprime --------------------------------------------------------------

def main():
    crudo = cargar_corpus(str(AQUI / CORPUS))
    secuencia = normalizar(crudo, ALFABETO)
    palabras = [p for p in secuencia.split() if p]
    vocabulario = sorted(set(palabras))

    selftest(secuencia, palabras)
    if "--selftest" in sys.argv:
        return

    t = tras(secuencia, PAR_VIGILADO)
    total_par = sum(t.values())
    p = parejas(palabras)
    posibles = casillas(len(vocabulario), 2)
    una_vez = de_una_sola_vez(p)
    muestra = leer_muestra(AQUI / MUESTRA, ROTULO_MUESTRA)
    trozos, reales = palabras_reales(muestra, set(vocabulario))

    lineas = [
        f"########## capítulo 1: lo que no cabe en la tabla ##########",
        f"fecha: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        "máquina: contenedor Linux en la nube (auditoría). La medición no",
        "depende de la máquina: cuenta letras y palabras de un fichero",
        "corpus: datos/quijote.txt, limpiado como lo limpia ngrama.py",
        "",
        "EL TEXTO QUE SE CUENTA",
        "",
        f"  letras (incluidos los espacios)        {miles(len(secuencia)):>14}",
        f"  palabras                               {miles(len(palabras)):>14}",
        f"  palabras distintas                     {miles(len(vocabulario)):>14}",
        "",
        f"QUÉ LETRA VIENE DETRÁS DE «{PAR_VIGILADO.upper()}»",
        f"las {miles(total_par)} veces que aparece «{PAR_VIGILADO}» en el Quijote",
        "",
        "  letra que sigue    veces    de cada cien veces",
        "  ---------------   ------   -------------------",
    ]
    for letra, n in t.most_common():
        parte = pct(n / total_par, 2)
        if parte.startswith("0,00"):
            parte = "menos de 0,01 %"      # un recuento pequeño no es un cero
        lineas.append(f"  {letra:^15}   {miles(n):>6}   {parte:>19}")
    lineas += [
        "",
        f"  símbolos distintos detrás de «{PAR_VIGILADO}»: {len(t)}, de los "
        f"{len(ALFABETO)} que",
        "  admite el programa (letras, espacio y signos de puntuación)",
        "",
        "LA TABLA DE PAREJAS DE PALABRAS",
        "una casilla por cada pareja de palabras que podría existir",
        "",
        f"  casillas posibles (palabras distintas al cuadrado)"
        f"{miles(posibles):>16}",
        f"  casillas con algo dentro (parejas vistas)        "
        f"{miles(len(p)):>16}",
        f"  casillas con algo dentro, por cada cien posibles "
        f"{pct(len(p) / posibles, 4):>16}",
        "",
        f"  una casilla con algo dentro de cada            "
        f"{miles(round(posibles / len(p))):>18}",
        "",
        f"  parejas que hay dentro del libro (palabras menos una)"
        f"{miles(len(palabras) - 1):>12}",
        f"  de cada cien de esas parejas, cuántas son distintas"
        f"{pct(len(p) / (len(palabras) - 1), 1):>14}",
        "",
        f"  parejas vistas una sola vez en todo el libro     "
        f"{miles(una_vez):>16}",
        f"  de cada cien parejas vistas, las de una sola vez "
        f"{pct(una_vez / len(p), 1):>16}",
        "",
        "  Quijotes de texto que harían falta para poner una sola",
        "  palabra en cada casilla                         "
        f"{miles(round(posibles / len(palabras))):>16}",
        "",
        "SI SE MIRARAN TRES PALABRAS HACIA ATRÁS",
        "",
        f"  casillas posibles                       "
        f"{miles(casillas(len(vocabulario), 3)):>16}",
        "  Quijotes de texto que harían falta para poner una sola",
        "  palabra en cada casilla                 "
        f"{miles(round(casillas(len(vocabulario), 3) / len(palabras))):>16}",
        "",
        "PALABRAS REALES EN LA MUESTRA DE UNA LETRA DE CONTEXTO",
        "la muestra impresa en el libro, leída de datos/salidas/ngrama.txt",
        "",
        f"  trozos entre espacios                           {miles(len(trozos)):>16}",
        f"  de ellos, palabras que están en el Quijote      "
        f"{miles(len(reales)):>16}",
        f"  de cada cien trozos, palabras del Quijote       "
        f"{pct(len(reales) / len(trozos), 1):>16}",
        "",
        "  las palabras del Quijote que salieron, en orden:",
    ]
    fila = "   "
    for palabra in reales:
        if len(fila) + len(palabra) + 3 > ANCHO_CAJA:
            lineas.append(fila)
            fila = "   "
        fila += f" «{palabra}»"
    lineas.append(fila)

    for l in comprobar_ancho(lineas, ANCHO_CAJA):
        print(l)


if __name__ == "__main__":
    main()
