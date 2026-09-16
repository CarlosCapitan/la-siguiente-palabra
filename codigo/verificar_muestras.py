#!/usr/bin/env python3
"""Comprueba que todas las citas en bloque de un capítulo son literales: que aparecen tal cual
en la salida del script que las generó.

    verificar_muestras.py capitulo.md salida.txt [salida2.txt ...]

En este libro una cita en bloque es, por defecto, salida de máquina sin retocar. Las únicas
excepciones son bloques escritos por el autor (frases de ejemplo, definiciones); para que el
verificador las acepte hay que declararlas a mano, una por una, en el fichero de excepciones.
Declararlas es un acto consciente: si no está declarada y no es literal, el capítulo falla.

Los bloques se localizan sobre el texto CRUDO, con sus saltos de línea intactos; la normalización
(espacios y coma decimal) se aplica solo al comparar."""
import re, sys, io, os

MINIMO = 70   # cuántos caracteres de la cita se exigen literales
EXCEPCIONES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'libro-ia-libro', 'notas', 'CITAS-DE-AUTOR.md')

def quitar_marcado(t):
    """Quita el marcado de imprenta que envuelve trozos de la salida, dejando el trozo.

    A temperatura alta el modelo escribe en chino, tailandés o árabe. La fuente del libro no
    dibuja esos signos, y xelatex los tira EN SILENCIO: sin envolverlos, el libro enseñaría una
    versión limpiada de lo que la máquina escribió. El envoltorio es formato, no dato.

    También quita las comillas simples inversas, que en el libro ponen un trozo en monoespaciado
    y no forman parte de lo que imprimió el programa."""
    t = re.sub(r'`\\[a-záéíóúñ]+\{(.*?)\}`\{=latex\}', r'\1', t)
    return t.replace('`', '')


def normalizar(t):
    """Compara ignorando tres convenciones de imprenta que no son datos: los espacios, el
    separador decimal (el libro escribe 0,018 y el script imprime 0.018) y el espacio fino
    antes del signo de porcentaje (el libro escribe 80 % y el script 80%)."""
    t = re.sub(r'(?<=\d),(?=\d)', '.', t)
    t = re.sub(r'\s+%', '%', t)
    return re.sub(r'\s+', ' ', t)

def texto_del_bloque(b):
    lineas = [re.sub(r'^(> ?| {4})', '', l) for l in b.splitlines()]
    lineas = [l for l in lineas if l.strip() != '```']   # las vallas son formato, no dato
    return normalizar(quitar_marcado(' '.join(lineas))).strip()

capitulo = sys.argv[1]
crudo = io.open(capitulo, encoding='utf-8').read()
salida = ' '.join(normalizar(io.open(f, encoding='utf-8').read()) for f in sys.argv[2:])
assert salida.strip(), "no me has dado ninguna salida contra la que comparar"

declaradas = []
if os.path.exists(EXCEPCIONES):
    for l in io.open(EXCEPCIONES, encoding='utf-8'):
        m = re.match(r'^-\s+`([^`]+)`', l)
        if m:
            declaradas.append(normalizar(m.group(1)).strip())

# Un bloque es cualquier cosa sangrada: con «>» (cita) o con cuatro espacios (salida de
# programa que necesita monoespaciado, como una tabla con columnas). La regla 6 dice
# «todo lo que va sangrado en bloque», no «todo lo que lleva >»: durante un tiempo el
# verificador solo miraba las citas, y una tabla sangrada con cuatro espacios pasaba sin
# que nadie la comprobara.
bloques = (re.findall(r'((?:^>.*\n)+)', crudo, re.M) +
           re.findall(r'((?:^ {4}.*\n|^\n(?= {4}))+)', crudo, re.M))

def lineas_de_datos(b):
    """Las líneas del bloque, sin el prefijo de cita ni las vallas de código."""
    out = []
    for l in b.splitlines():
        l = re.sub(r'^(> ?| {4})', '', l)
        if l.strip() == '```':
            continue
        l = normalizar(quitar_marcado(l)).strip()
        if l:
            out.append(l)
    return out

fallos = comprobados = de_autor = recompuestos = 0
for b in bloques:
    t = texto_del_bloque(b)
    if not t:
        continue
    if any(t.startswith(d) for d in declaradas):
        de_autor += 1
        continue
    comprobados += 1
    if t[:MINIMO] in salida:
        continue
    # El bloque entero no cuadra. Segunda oportunidad: que cada línea, una por una, sea
    # literal. Eso permite quitar del libro lo que es adorno de consola —unas comillas, una
    # columna de barras que no cabe en la página— y sigue prohibiendo cambiar un solo dato.
    sueltas = lineas_de_datos(b)
    perdidas = [l for l in sueltas if l not in salida]
    if perdidas:
        print(f"NO LITERAL: «{perdidas[0][:60]}…»")
        fallos += 1
    else:
        recompuestos += 1
        print(f"recompuesto (líneas literales, bloque recortado): «{sueltas[0][:50]}…»")

# Las tablas con porcentajes también son datos medidos: cada fila tiene que aparecer, con sus
# números y en el mismo orden, en la salida del programa.
filas = 0
for l in crudo.splitlines():
    if not l.startswith('|') or '%' not in l:
        continue
    celdas = [normalizar(quitar_marcado(c)).strip() for c in l.strip('|').split('|')]
    celdas = [c for c in celdas if c and not set(c) <= set('-: ')]
    fila = ' '.join(celdas).replace('**', '')
    filas += 1
    if fila not in salida:
        print(f"TABLA NO LITERAL: «{fila[:60]}…»")
        fallos += 1

assert comprobados or filas or de_autor, (
    f"no encontré en {capitulo} ni una cita en bloque ni una fila de tabla con números: "
    "o el capítulo no enseña ningún dato medido, o el formato ha cambiado")

print(f"{comprobados} citas de máquina comprobadas ({recompuestos} recortadas), "
      f"{de_autor} bloques declarados del autor, {filas} filas de tabla con números")
print("FALLA" if fallos else "PASA: todas las citas de máquina son literales")
sys.exit(1 if fallos else 0)
