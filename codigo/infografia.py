#!/usr/bin/env python3
"""
El sistema visual de las infografías del libro.

Una sola hoja de estilo para todas las figuras, para que las treinta que vienen se lean como
una sola cosa y no como treinta. Aquí no hay ningún dato: solo cómo se dibuja. Los datos los
pone cada figura, y salen de `datos/salidas/`.

Dos paletas con los mismos papeles, para poder decidir con las dos delante:
  - COLOR: como las maquetas de Carlos.
  - GRIS:  la misma figura imprimible en negro, sin que ningún significado dependa del color.

Uso:
    from infografia import Lienzo, COLOR, GRIS
    python infografia.py --selftest
"""

# ======================= CONSTANTES =======================

# La caja de texto del libro, medida en el propio libro.log: \textwidth = 321,60 pt y
# \textheight = 523,96 pt, que en pulgadas son 4,45 por 7,25. Como libro.tex mete todas las
# figuras con width=\linewidth, dibujar a 4,45 es dibujar al tamaño real: nada se reescala y un
# cuerpo de 7,2 puntos en el dibujo sale a 7,2 puntos en el papel.
ANCHO_PAGINA = 4.45         # pulgadas útiles de ancho
ALTO_PAGINA = 7.25          # pulgadas útiles de alto
ALTO_MAXIMO = 6.55          # lo que le queda a la imagen con su pie de tres líneas debajo
PUNTOS = 300                # puntos por pulgada: impresión, no pantalla
FUENTE = "Carlito"          # humanista, con acentos y eñes completos

class Paleta:
    def __init__(self, tinta, suave, marco, acento, contra, neutro, fondo, papel):
        self.tinta = tinta      # texto principal
        self.suave = suave      # texto secundario, pies
        self.marco = marco      # bordes de los paneles
        self.acento = acento    # lo que suma, lo que se destaca
        self.contra = contra    # lo que resta, la otra categoría
        self.neutro = neutro    # lo que no aporta
        self.fondo = fondo      # relleno de los paneles
        self.papel = papel

COLOR = Paleta(tinta="#17303c", suave="#5d6b73", marco="#c9d3d8", acento="#d6a12a",
               contra="#1d7d80", neutro="#c4cbd0", fondo="#f7f9fa", papel="white")
GRIS  = Paleta(tinta="#000000", suave="#585858", marco="#b5b5b5", acento="#3d3d3d",
               contra="#9a9a9a", neutro="#d8d8d8", fondo="#f4f4f4", papel="white")

# El libro se imprime en negro (decidido el 20 sep 2026: en KDP el interior en color encarece
# cada ejemplar). Así que cuando dos cosas tienen que distinguirse y el gris no basta, se
# distinguen con trama. Éstas son las que sobreviven al papel: probadas a 4,6 pulgadas de ancho
# y 300 ppp, que es el tamaño real de una figura de este libro. Las descartadas —la cruz «+»,
# la malla y los círculos— se llenan y quedan de color barro.
GROSOR_TRAMA = 0.8          # medido en la hoja de prueba: con 0,5 desaparece y con 1,2 empasta
TRAMAS = {
    "llena":   (None,   "acento"),   # lo destacado
    "media":   (None,   "contra"),   # la otra categoría
    "clara":   (None,   "neutro"),   # lo que no decide
    "rayas":   ("///",  "papel"),    # una categoría más, sobre blanco
    "cruz":    ("xxx",  "papel"),    # y otra
    "puntos":  ("...",  "papel"),    # y otra
}
ALTO_MINIMO_TRAMA = 2.5     # en unidades del lienzo: por debajo de esto, la trama se llena y
                            # hay que volver al gris. Una barra fina no lleva trama.
# Lo que hace falta para que dos rellenos se distingan en el papel, medido sobre la hoja de
# prueba a 4,45 pulgadas y 300 ppp. Basta con que se separen por uno cualquiera de los tres: las
# llenas se distinguen por el tono y las rayadas por el número de saltos, porque entre ellas el
# tono es casi el mismo (rayas y puntos se llevan ocho milésimas de gris, y sin embargo se ven
# distintas porque una tiene líneas y la otra motas).
UMBRAL_TINTA = 0.15         # a partir de aquí un punto cuenta como tinta y no como papel
SEPARACION_TONO = 0.08      # diferencia de gris medio
SEPARACION_COBERTURA = 0.05 # diferencia de superficie cubierta
SEPARACION_SALTOS = 0.80    # diferencia de pasos de blanco a tinta por fila

# ==========================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow


class Lienzo:
    """Una página de infografía: título, subtítulo, pasos numerados y pie."""

    def __init__(self, titulo, subtitulo, paleta, alto=6.0, ancho=ANCHO_PAGINA):
        # Una figura más alta que esto no cabe en la página con su pie, y LaTeX la encoge hasta
        # que quepa: los rótulos se van con ella y dejan de leerse. Que no quepa es un fallo de
        # la figura, no un detalle de la maquetación, así que revienta aquí.
        assert alto <= ALTO_MAXIMO, (
            f"la figura mide {alto} pulgadas de alto y en la página caben {ALTO_MAXIMO}; "
            f"pártela en dos o quítale cosas")
        self.p = paleta
        plt.rcParams["font.family"] = FUENTE
        plt.rcParams["hatch.linewidth"] = GROSOR_TRAMA
        self.fig = plt.figure(figsize=(ancho, alto), facecolor=paleta.papel)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100 * alto / ancho)
        self.ax.axis("off")
        self.alto_u = 100 * alto / ancho
        self.y = self.alto_u - 4
        self.ax.text(4, self.y, titulo, ha="left", va="top", fontsize=15,
                     fontweight="bold", color=paleta.tinta)
        self.y -= 6.2
        if subtitulo:
            self.ax.text(4, self.y, subtitulo, ha="left", va="top", fontsize=8.4,
                         color=paleta.suave, linespacing=1.35)
            self.y -= 4.0 + 3.4 * subtitulo.count("\n")
        self.y -= 1.5

    def panel(self, numero, titulo, alto):
        """Un paso: caja con su número y su título. Devuelve (x0, y_tope, ancho) del interior."""
        p, ax = self.p, self.ax
        y0 = self.y - alto
        ax.add_patch(FancyBboxPatch((4, y0), 92, alto,
                                    boxstyle="round,pad=0,rounding_size=1.4",
                                    facecolor=p.fondo, edgecolor=p.marco, linewidth=0.9))
        ax.add_patch(Circle((9.5, self.y - 4.2), 2.6, facecolor=p.acento, edgecolor="none"))
        ax.text(9.5, self.y - 4.2, str(numero), ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")
        ax.text(14.5, self.y - 4.2, titulo, ha="left", va="center", fontsize=10.2,
                fontweight="bold", color=p.tinta)
        interior = (8.0, self.y - 8.4, 84.0)
        self.y = y0 - 4.6
        return interior

    def flecha(self):
        """La flecha que baja de un paso al siguiente. Vive en el hueco entre paneles, nunca
        encima de uno: si se dibujara dentro, taparía el dato que el panel acaba de enseñar."""
        arriba = self.y + 4.0
        self.ax.add_patch(FancyArrow(50, arriba, 0, -3.0, width=0.6, head_width=2.2,
                                     head_length=1.2, length_includes_head=True,
                                     facecolor=self.p.acento, edgecolor="none"))

    def ficha(self, x, y, texto, ancho, alto=3.4, relleno=None, tinta=None, negrita=False,
              tam=7.6, mono=False, trama=None):
        """Una pastilla con texto dentro: un trozo, un número, una etiqueta."""
        p = self.p
        self.ax.add_patch(FancyBboxPatch((x, y - alto / 2), ancho, alto,
                                         boxstyle="round,pad=0,rounding_size=0.8",
                                         facecolor=relleno or "white",
                                         hatch=trama if alto >= ALTO_MINIMO_TRAMA else None,
                                         edgecolor=p.marco, linewidth=0.8))
        self.ax.text(x + ancho / 2, y, texto, ha="center", va="center", fontsize=tam,
                     color=tinta or p.tinta, fontweight="bold" if negrita else "normal",
                     family="DejaVu Sans Mono" if mono else FUENTE)

    def texto(self, x, y, s, tam=7.8, color=None, ha="left", negrita=False, mono=False):
        self.ax.text(x, y, s, ha=ha, va="center", fontsize=tam,
                     color=color or self.p.tinta, fontweight="bold" if negrita else "normal",
                     family="DejaVu Sans Mono" if mono else FUENTE)

    def pie(self, s):
        self.ax.text(4, 2.6, s, ha="left", va="center", fontsize=6.8, color=self.p.suave,
                     linespacing=1.4)

    def guardar(self, ruta):
        # Los selftests dibujan la figura entera para que salten sus asserts, pero no la guardan:
        # pasan "/dev/null". Sin extensión, matplotlib le añade «.png» y escribe /dev/null.png,
        # que solo funciona siendo root. Se dibuja en memoria (25 de septiembre de 2026).
        if ruta == "/dev/null":
            import io
            self.fig.savefig(io.BytesIO(), format="png", dpi=PUNTOS, facecolor=self.p.papel)
        else:
            self.fig.savefig(ruta, dpi=PUNTOS, facecolor=self.p.papel)
        plt.close(self.fig)


# ======================= SELFTEST =======================
# Esta hoja de estilo no lleva datos, pero sí lleva tres cosas que, si se rompen, rompen en
# silencio todas las figuras del libro a la vez: el tamaño de la página, el vocabulario de
# tramas y la escala del lienzo. Por eso tiene selftest.

def _selftest():
    import numpy as np
    fallos = []

    # 1. TEST NULO — una figura que no cabe en la página tiene que ser rechazada. Si no lo
    #    fuera, LaTeX la encogería hasta que quepa y se llevaría los rótulos con ella: la
    #    figura seguiría saliendo, ilegible, y nadie se enteraría hasta ver el papel.
    alta = ALTO_MAXIMO + 0.5
    try:
        Lienzo("prueba", "", GRIS, alto=alta)
        revienta = False
    except AssertionError:
        revienta = True
    plt.close("all")
    print(f"[1] test nulo         con {alta} pulgadas de alto, "
          f"{'revienta' if revienta else 'NO revienta'}")
    if not revienta:
        fallos.append("test nulo: se deja dibujar una figura que no cabe en la página")

    # 2. SEÑAL IMPLANTADA — las seis tramas, dibujadas al alto mínimo que se declara y al tamaño
    #    real de impresión, salen distinguibles unas de otras. Se miden las tres cosas por las
    #    que el ojo las separa en el papel: el tono, cuánta superficie cubren y cuántas veces se
    #    pasa de blanco a tinta a lo largo de una fila. Dos tramas valen si se separan por una
    #    cualquiera de las tres: el gris medio distingue las llenas, y el número de saltos
    #    distingue las rayadas, que tienen casi el mismo tono entre ellas.
    import tempfile, os
    import numpy as np
    L = Lienzo("tramas", "", GRIS, alto=2.0)
    nombres = list(TRAMAS)
    ancho = 92.0 / len(nombres)
    for i, n in enumerate(nombres):
        trama, relleno = TRAMAS[n]
        L.ficha(4 + i * ancho, 20, "", ancho - 1.0, alto=ALTO_MINIMO_TRAMA * 2,
                relleno=getattr(L.p, relleno), trama=trama)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    L.guardar(tmp.name)
    img = 1 - plt.imread(tmp.name)[:, :, :3].mean(axis=2)
    os.unlink(tmp.name)
    alto_px, ancho_px = img.shape
    unidades = 100 * 2.0 / ANCHO_PAGINA
    firma = {}
    for i, n in enumerate(nombres):
        x0 = int((4 + i * ancho + 1.5) / 100 * ancho_px)
        y = int((1 - 20 / unidades) * alto_px)
        h = int(ALTO_MINIMO_TRAMA / unidades * alto_px)
        p = img[y - h:y + h, x0:x0 + 140]
        hay = p > UMBRAL_TINTA
        saltos = (np.diff(hay.astype(int), axis=1) == 1).sum() / hay.shape[0]
        firma[n] = (p.mean(), hay.mean(), saltos)
    peor, pareja = 9.0, None
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            d = max(abs(firma[a][0] - firma[b][0]) / SEPARACION_TONO,
                    abs(firma[a][1] - firma[b][1]) / SEPARACION_COBERTURA,
                    abs(firma[a][2] - firma[b][2]) / SEPARACION_SALTOS)
            if d < peor:
                peor, pareja = d, (a, b)
    print(f"[2] señal implantada  las {len(nombres)} tramas se separan; la pareja más parecida "
          f"es {pareja[0]} y {pareja[1]}, con {peor:.2f} veces el mínimo")
    if peor < 1.0:
        fallos.append(f"señal implantada: {pareja[0]} y {pareja[1]} no se distinguen "
                      f"({peor:.2f} veces el mínimo)")

    # 3. INVARIANTE DEL DOMINIO — el lienzo se dibuja al tamaño real de la caja del libro y con
    #    la unidad cuadrada. Si la unidad dejara de ser cuadrada, todo lo que se dibuja contando
    #    unidades —las rejillas, las fichas, los huecos— saldría deformado sin previo aviso.
    L = Lienzo("escala", "", GRIS, alto=3.0)
    px_x = L.fig.get_size_inches()[0] * PUNTOS
    unidad_x = px_x / 100
    unidad_y = (L.fig.get_size_inches()[1] * PUNTOS) / L.alto_u
    plt.close("all")
    cuadrada = abs(unidad_x - unidad_y) < 1e-6
    print(f"[3] invariante        caja de {ANCHO_PAGINA} x {ALTO_PAGINA} pulgadas, "
          f"{px_x:.0f} px de ancho a {PUNTOS} ppp; unidad cuadrada: "
          f"{'sí' if cuadrada else 'NO'}")
    if not cuadrada:
        fallos.append("invariante: la unidad del lienzo no es cuadrada")
    if not ALTO_MAXIMO < ALTO_PAGINA:
        fallos.append("invariante: el alto máximo de una figura no deja sitio para su pie")

    print()
    if fallos:
        for f in fallos: print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="la hoja de estilo de las infografías del libro")
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        sys.exit(_selftest())
    print("Esto es una hoja de estilo: no dibuja nada por su cuenta. Usa --selftest.")
