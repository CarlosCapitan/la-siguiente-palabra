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
"""

# ======================= CONSTANTES =======================

ANCHO_PAGINA = 4.6          # pulgadas útiles en una página de 6 por 9 con sus márgenes
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

# ==========================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow


class Lienzo:
    """Una página de infografía: título, subtítulo, pasos numerados y pie."""

    def __init__(self, titulo, subtitulo, paleta, alto=7.2, ancho=ANCHO_PAGINA):
        self.p = paleta
        plt.rcParams["font.family"] = FUENTE
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
              tam=7.6, mono=False):
        """Una pastilla con texto dentro: un trozo, un número, una etiqueta."""
        p = self.p
        self.ax.add_patch(FancyBboxPatch((x, y - alto / 2), ancho, alto,
                                         boxstyle="round,pad=0,rounding_size=0.8",
                                         facecolor=relleno or "white",
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
        self.fig.savefig(ruta, dpi=PUNTOS, facecolor=self.p.papel)
        plt.close(self.fig)
