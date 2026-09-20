#!/usr/bin/env python3
"""Cómo se imprimen las tablas del libro: lo ancha que puede ser una línea y cómo se
escribe un número en castellano.

Vive en un solo sitio porque todos los programas del libro imprimen para la misma página.
Si cada uno lleva su propia copia, al mes hay tres anchos distintos y dos maneras de
escribir el mismo decimal, y el lector lo nota aunque no sepa decir qué le chirría.

EL ANCHO. Una tabla que no cabe sale partida o pisando el margen, y entonces no es una
tabla: es un borrón. El ancho está MEDIDO, no supuesto: se compuso una página de prueba
con el preámbulo real del libro (papel de 6 x 9 pulgadas, caja de 4,45 pulgadas, los
bloques de programa a \\footnotesize) y se fue alargando una línea de caracteres iguales
hasta que xelatex avisó de que se salía.

  - bloque normal:             con 68 no avisa; con 69 avisa (se sale 4,42 pt).
  - bloque dentro de una cita: con 64 no avisa; con 65 avisa (se sale 0,85 pt).

La cita es más estrecha porque su entorno mete margen por los dos lados. Si cambia el
papel, la caja o el cuerpo de letra, estos dos números se vuelven a MEDIR.

LA COMA. El libro está en castellano, y en castellano el decimal va con coma y el signo
de porcentaje con un espacio delante: 90,4 %. Que el programa imprima 90.4% y el libro
escriba 90,4 % es una diferencia invisible mientras se escribe y muy visible impresa.
"""

ANCHO_CAJA = 68
ANCHO_CAJA_CITA = 62   # medido el 20 sep 2026 con el preámbulo actual (cita con margen de 1,4 em)


def comprobar_ancho(lineas, limite=ANCHO_CAJA):
    """Revienta si alguna línea no cabe. Se llama justo antes de imprimir una tabla: más
    vale que falle aquí, donde se lee el mensaje, que en la página impresa."""
    for l in lineas:
        assert len(l) <= limite, (
            f"Se esperaba una línea de {limite} caracteres como mucho; "
            f"se encontró una de {len(l)}: {l!r}"
        )
    return lineas


def coma(x, decimales=1):
    """Un número con coma decimal. 90.4 -> '90,4'."""
    return f"{x:.{decimales}f}".replace(".", ",")


def pct(x, decimales=1, de_uno=True):
    """Un porcentaje a la castellana: coma decimal y espacio antes del signo.

    de_uno=True espera la proporción (0,904 -> '90,4 %'); de_uno=False espera el número
    ya en porcentaje (90.4 -> '90,4 %'). Se dice cuál se le pasa porque confundirlos
    multiplica o divide por cien en silencio, y un dato cien veces mayor es un dato falso."""
    valor = 100 * x if de_uno else x
    return coma(valor, decimales) + " %"


# ---- La tabla de «qué trozo puede venir ahora» -----------------------------------
# Sale en el capítulo 7 y vuelve en el 12, y tiene que salir IGUAL las dos veces: el
# lector reconoce una tabla que ya ha visto mucho antes que un dato que ya ha leído.
ANCHO_TROZO = 12
ANCHO_PROB = 12
ESCALA_BARRA = 28    # almohadillas que valen el 100 %


def tabla_de_probabilidades(pares, con_barra=True, decimales=2):
    """La lista de continuaciones posibles, con rótulo de columna y barra.

    Los rótulos no son adorno. Sin ellos son dos columnas de cifras, y el lector que se
    encuentre la tabla al volver la página no tiene manera de saber qué es cada una.
    La barra no añade ningún dato: deja ver de un vistazo lo que el número dice de una
    en una, que para eso el ojo humano es mucho mejor que para comparar cifras."""
    lineas = [
        f"{'trozo':>{ANCHO_TROZO}}   {'probabilidad':>{ANCHO_PROB}}",
        f"{'-' * ANCHO_TROZO}   {'-' * ANCHO_PROB}",
    ]
    # El trozo llega ya escrito como se va a leer: el guion bajo que marca el espacio
    # lo pone quien conoce el troceador, no esta función, que si no acaba convirtiendo
    # en guion bajo el espacio de un rótulo como «(el resto)».
    for palabra, v in pares:
        fila = f"{palabra:>{ANCHO_TROZO}}   {pct(v, decimales):>{ANCHO_PROB}}"
        if con_barra:
            fila += "  " + "#" * max(1, int(round(v * ESCALA_BARRA)))
        lineas.append(fila.rstrip())
    return comprobar_ancho(lineas, ANCHO_CAJA_CITA)


def miles(n):
    """Un número entero con el separador de miles del castellano: 151936 -> '151.936'.

    Python escribe 151,936 con la coma inglesa, y en un libro en castellano eso se lee
    como un decimal. No es una coquetería: cambia el número que el lector entiende."""
    return f"{int(n):,}".replace(",", ".")
