#!/usr/bin/env python3
"""
Sitios del libro donde el lector puede tener que imaginar (L24).

Criterio de Carlos, 27 de septiembre de 2026: «Obligamos al lector a imaginar, e imaginar es igual
a abstracción. No tiene que imaginar: tiene que verlo escrito y dibujado en el libro, muy clarito.»

Este programa NO decide nada: marca candidatos para leerlos. Busca, página a página del PDF de
lectura, frases con alguna de estas señales:

  pide          el texto le pide al lector que imagine, suponga, piense o cuente él
  remite        el texto se apoya en algo que no está delante: «recuerda», «como vimos», «el
                capítulo 4», «ya sabías»
  vago          la frase se queda en lo general: «de algún modo», «una especie de», «cualquier»
  compara       una comparación que puede estar haciendo el trabajo de la explicación
  grande        una cantidad enorme que el lector tiene que figurarse

La lectura de verdad la hace una persona, con el candidato delante. Que una frase no salga aquí no
quiere decir que se entienda: la tabla de «acierta» de L23 no tenía ninguna de estas palabras.

Entrada: el texto del PDF sacado con `pdftotext -layout` (las páginas separadas por salto de
página). El número de página es el que va impreso al pie, no el del PDF.

Uso:
    python candidatos_imaginar.py --selftest
    python candidatos_imaginar.py lectura.txt --desde 51 > candidatos.txt
"""

# ======================= CONSTANTES =======================

SENALES = {
    "pide": [r"\bimagin[ae]\w*", r"\bimagínate\b", r"\bsup[oó]n\w*", r"\bsupongamos\b",
             r"\bpiensa\b", r"\bpensemos\b", r"\bfigúrate\b", r"\bcomo si\b", r"\bhaz (?:la|tú)\b",
             r"\bcuenta tú\b", r"\bcompruéba\w*", r"\bprueba a\b", r"\bbusca\b", r"\bfíjate\b",
             r"\bmira (?:la|el|los|las)\b", r"\bsigue\w* con el dedo\b"],
    "remite": [r"\brecuerd\w*", r"\bacuérdate\b", r"\bcomo (?:vimos|hemos visto|ya vimos)\b",
               r"\bya sab\w+\b", r"\bcapítulo \d+\b", r"\bcapítulo anterior\b",
               r"\b(?:tabla|figura|bloque|cuadro|cuenta|pregunta) de (?:arriba|antes)\b",
               r"\bmás arriba\b", r"\bmás adelante\b"],
    "vago": [r"\bde algún modo\b", r"\bde alguna manera\b", r"\ben cierto sentido\b",
             r"\buna especie de\b", r"\balgo así como\b", r"\bpor así decirlo\b",
             r"\ben general\b", r"\bcualquier\w*\b", r"\bde alguna forma\b", r"\bpor dentro\b"],
    "compara": [r"\bes como\b", r"\bson como\b", r"\bse parece a\b", r"\bigual que\b",
                r"\bparecido a\b", r"\bviene a ser\b"],
    "grande": [r"\bmil(?:es)? de millones\b", r"\bbillones\b", r"\bmillones\b", r"\bbillón\b"],
}

# ==========================================================

import argparse
import re
import sys

PATRONES = {cat: re.compile("|".join(ps), re.I) for cat, ps in SENALES.items()}


def paginas(texto):
    """[(página impresa, texto de la página)]. La impresa es el último número suelto de la página;
    las que no llevan número (portada, índice) se saltan."""
    salida = []
    for bruto in texto.split("\f"):
        lineas = [l for l in bruto.splitlines() if l.strip()]
        if not lineas or not re.fullmatch(r"\s*\d+\s*", lineas[-1]):
            continue
        numero = int(lineas[-1])
        cuerpo = "\n".join(lineas[1:-1])           # fuera el encabezado y el número
        cuerpo = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", cuerpo)   # palabras partidas a final de línea
        salida.append((numero, cuerpo))
    return salida


def frases(cuerpo):
    plano = re.sub(r"\s+", " ", cuerpo).strip()
    return [f.strip() for f in re.split(r"(?<=[.!?»:])\s+(?=[¿¡«A-ZÁÉÍÓÚÑ0-9])", plano) if f.strip()]


def candidatos(texto, desde):
    salida = []
    for numero, cuerpo in paginas(texto):
        if numero < desde:
            continue
        for f in frases(cuerpo):
            cats = [c for c, p in PATRONES.items() if p.search(f)]
            if cats:
                salida.append((numero, cats, f))
    return salida


def selftest():
    fallos = []
    # 1. TEST NULO — una página sin ninguna señal no da candidatos.
    nulo = "Encabezado\nEl segmento de abajo izquierda suma dos puntos. El listón es 0,5.\n7\n"
    r = candidatos(nulo, 1)
    print(f"[1] test nulo         página sin señales: {len(r)} candidatos")
    if r:
        fallos.append(f"test nulo: da candidatos donde no hay señales: {r}")
    # 2. SEÑAL IMPLANTADA — una frase con «imagina» en la página impresa 9 sale en la 9, con la
    #    categoría «pide», y la página 8 (antes de --desde) no sale.
    senal = ("Cabeza\nImagina un reloj de siete segmentos. Otra frase.\n8\n\f"
             "Cabeza\nNada aquí. Ahora imagina que el reloj se apaga-\nra del todo.\n9\n")
    r = candidatos(senal, 9)
    ok = len(r) == 1 and r[0][0] == 9 and r[0][1] == ["pide"] and "apagara" in r[0][2]
    print(f"[2] señal implantada  {r}")
    if not ok:
        fallos.append("señal implantada: no sale en la página 9, o sin «pide», o sin unir «apaga-ra»")
    # 3. INVARIANTE — cada candidato es una frase entera de su página, y la página impresa no se
    #    confunde con la del PDF (la portada sin número no cuenta).
    inv = "Portada sin número\n\fCabeza\nRecuerda la tabla. Fin.\n3\n"
    r = candidatos(inv, 1)
    ok = r == [(3, ["remite"], "Recuerda la tabla.")]
    print(f"[3] invariante        {r}")
    if not ok:
        fallos.append("invariante: la página o la frase no son las que tocan")
    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texto", nargs="?")
    ap.add_argument("--desde", type=int, default=1, help="primera página impresa que se mira")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    codigo = selftest()
    if codigo or args.selftest:
        sys.exit(codigo)
    assert args.texto, "falta el texto del PDF (pdftotext -layout)"
    r = candidatos(open(args.texto, encoding="utf-8").read(), args.desde)
    print()
    print(f"candidatos desde la página {args.desde}: {len(r)}")
    for cat in SENALES:
        print(f"  {cat:<10}{sum(cat in c for _, c, _ in r):>5}")
    print()
    for numero, cats, f in r:
        print(f"p. {numero:<4} [{', '.join(cats)}] {f}")


if __name__ == "__main__":
    main()
