#!/usr/bin/env python3
"""Comprueba que las salidas de los programas escriben los números en castellano.

    verificar_cifras.py ../datos/salidas
    verificar_cifras.py --selftest

La regla 9 dice coma decimal y punto de millar: 2.095.402 y 90,4 %. Los ficheros de
`datos/salidas/` los cita el libro **literalmente** (regla 6), así que si un programa escribe
`2,095,402`, el libro tiene dos salidas y las dos son malas: copiarlo tal cual, y que la página
enseñe un número a la inglesa; o arreglarlo a mano al copiar, que es exactamente lo que la regla 6
prohíbe. Eso último ya pasó dos veces, en `retropropagacion.py` y en `ngrama.py`, y lo cazó un
auditor leyendo, no un verificador.

`formato.py` existe para esto: `miles()` y `coma()`. Lo que faltaba era alguien que lo comprobara.

Lo que NO se marca, porque no son cantidades y marcarlas convertiría esto en un verificador que
nadie mira: versiones (`Linux 6.18.44`), fechas ISO (`2026-09-18`), notación científica
(`3.7e-12`), nombres de fichero y rutas."""
import os, re, sys

# Una cantidad con separador de millar inglés. Aquí pasa lo mismo que con el punto, en espejo:
# «0,018» es cero coma cero dieciocho en castellano, y «7,192» puede ser siete coma ciento noventa
# y dos. Con UNA sola coma no hay manera de saberlo. Con DOS o más (2,095,402) sí: en castellano
# eso no es nada.
MILES_INGLES = re.compile(r'(?<![\d.,])\d{1,3}(?:,\d{3}){2,}(?![\d.,])')
# Una sola coma seguida de tres cifras: ambiguo. Se avisa, no se suspende, salvo que la parte
# entera tenga más de tres cifras (12345,678 no es un decimal castellano con sentido).
MILES_DUDOSO = re.compile(r'(?<![\d.,])(\d{1,3}),(\d{3})(?![\d.,])')
# Un decimal con punto: 90.4, 0.933. Se excluye lo que tenga otro punto cerca (versiones, rutas),
# lo que lleve una e detrás (notación científica) y lo que vaya pegado a una letra.
#
# Y hay un caso que NO se puede decidir mirando el número: «1.024» es mil veinticuatro en
# castellano y uno coma cero veinticuatro en inglés, y las dos lecturas son legítimas. Un
# verificador que adivina acaba marcando texto bueno, y un verificador que marca texto bueno acaba
# desconectado. Así que se marca solo lo que es inequívoco:
#   - la parte decimal NO tiene exactamente tres cifras  (90.4, 0.9331) — nunca es millar
#   - o la parte entera es un cero solo                  (0.933)       — «cero mil» no existe
# «1.024», «15.048» y compañía se dejan pasar, y esta decisión está escrita aquí a propósito para
# que quien se encuentre uno sepa que no es un descuido: es que no se puede saber.
DECIMAL_CANDIDATO = re.compile(r'(?<![\w.,])(\d+)\.(\d+)(?![\d.]|\s*[eE][-+]?\d)(?![^\s]*[/\\])')


def es_decimal_ingles(entera, decimal):
    return len(decimal) != 3 or entera == "0"


def revisar_texto(texto):
    """Devuelve [(número de línea, clase, trozo)] de lo que está escrito a la inglesa."""
    fallos = []
    for i, l in enumerate(texto.splitlines(), 1):
        # Una línea de traceback o de ruta no es una tabla de datos del libro.
        if l.lstrip().startswith(('File "', 'Traceback', '  File')):
            continue
        for m in MILES_INGLES.finditer(l):
            fallos.append((i, 'miles', m.group(0)))
        for m in MILES_DUDOSO.finditer(l):
            # «0,018» no es millar en ningún idioma: la parte entera de un millar nunca es cero.
            if m.group(1) != "0":
                fallos.append((i, 'dudoso', m.group(0)))
        for m in DECIMAL_CANDIDATO.finditer(l):
            if es_decimal_ingles(m.group(1), m.group(2)):
                fallos.append((i, 'decimal', m.group(0)))
    return fallos


def selftest():
    """Tres pruebas sobre lo único que decide: si un número está escrito a la inglesa."""
    fallos = []

    # 1. TEST NULO — todo lo que es correcto o no es una cantidad. Si algo de esto saltara, el
    #    verificador marcaría salidas buenas y acabaríamos desconectándolo, que es como se
    #    pierden los verificadores.
    bueno = "\n".join([
        "  aciertos          90,4 %",
        "  corpus         2.095.402 palabras",
        "Máquina: x86_64, Linux 6.18.44-fc-v33.",      # una versión no es una cantidad
        "Medido el 2026-09-18. Semilla: 20260914.",    # una fecha tampoco
        "diferencia máxima 3.7e-12",                   # notación científica
        "Escrito datos/salidas/que_mira.csv",          # una ruta
        "de 1.024 bytes justos",            # mil veinticuatro: ambiguo, no se marca
        "el kilobyte son 15.048 y pico",    # ídem
        "  salidas: 0,018  0,984  0,980",   # decimales castellanos, no millares
    ])
    sale = revisar_texto(bueno)
    print(f"[1] test nulo         salida correcta: {len(sale)} incidencias"
          + (f" ({sale})" if sale else ""))
    if sale:
        fallos.append(f"test nulo: marca salida buena: {sale}")

    # 2. SEÑAL IMPLANTADA — los dos casos reales que ya ocurrieron en este libro, uno de cada
    #    clase, y cada uno tiene que cazarse por su nombre.
    implantadas = {'miles':   "Corpus: 2,095,402 caracteres de texto.",
                   'decimal': "un solo tribunal, sin capa     90.4 %"}
    cazadas = {c: c in {f[1] for f in revisar_texto(l)} for c, l in implantadas.items()}
    print("[2] señal implantada  " + "; ".join(
        f"{c}: {'la caza' if v else 'SE LE ESCAPA'}" for c, v in cazadas.items()))
    if not all(cazadas.values()):
        fallos.append("señal implantada: se le escapa "
                      + ", ".join(c for c, v in cazadas.items() if not v))

    # 3. INVARIANTE DEL DOMINIO — el mismo número escrito bien y mal en la misma línea: tiene que
    #    marcar uno y solo uno. Si marcara los dos, no está distinguiendo nada; si no marcara
    #    ninguno, tampoco.
    mezcla = revisar_texto("antes 2.095.402 y ahora 2,095,402")
    print(f"[3] invariante        el mismo número bien y mal en una línea: marca {len(mezcla)} "
          f"({[f[2] for f in mezcla]})")
    if len(mezcla) != 1 or mezcla[0][2] != "2,095,402":
        fallos.append(f"invariante: tenía que marcar solo el malo y marcó {mezcla}")

    # 4. LO QUE NO SE PUEDE DECIDIR — «1.024» es mil veinticuatro o uno coma cero veinticuatro, y
    #    no hay manera de saberlo mirando el número. Se deja pasar A PROPÓSITO, y esta prueba está
    #    aquí para que si alguien «mejora» el verificador y empieza a marcarlo, se entere.
    ambiguo = revisar_texto("tenía 1.024 bytes")
    claro = revisar_texto("acierta 90.4 % y 0.933")
    print(f"[4] lo ambiguo        «1.024»: {len(ambiguo)} incidencias (tiene que ser 0); "
          f"«90.4» y «0.933»: {len(claro)} (tienen que ser 2)")
    if ambiguo or len(claro) != 2:
        fallos.append("lo ambiguo: 1.024 se deja pasar; 90.4 y 0.933 no")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las cuatro pruebas pasan.")
    return 0


if '--selftest' in sys.argv:
    sys.exit(selftest())

carpeta = sys.argv[1]
total = malos = avisos = 0
for nombre in sorted(os.listdir(carpeta)):
    if not nombre.endswith('.txt'):
        continue
    total += 1
    todos = revisar_texto(open(os.path.join(carpeta, nombre), encoding='utf-8').read())
    ciertos = [f for f in todos if f[1] != 'dudoso']
    dudosos = [f for f in todos if f[1] == 'dudoso']
    if ciertos:
        malos += 1
        muestra = ", ".join(f"«{f[2]}» (l. {f[0]})" for f in ciertos[:3])
        print(f"  {nombre:<34} {len(ciertos):>3} a la inglesa: {muestra}")
    if dudosos:
        avisos += len(dudosos)
        muestra = ", ".join(f"«{f[2]}»" for f in dudosos[:2])
        print(f"  {nombre:<34} {'':>3} AVISO: {muestra} puede ser millar inglés o decimal nuestro")

if avisos:
    print(f"\n{avisos} caso(s) que no se pueden decidir mirando el número: una coma sola seguida")
    print("de tres cifras es millar en inglés y decimal en castellano. Avisa y no suspende.")
if malos:
    print(f"\nFALLA: {malos} de {total} ficheros de salida escriben números a la inglesa.")
    print("Se arregla en el programa que los imprime, con miles() y coma() de formato.py,")
    print("nunca a mano en el libro al copiarlos (regla 6).")
    sys.exit(1)
print(f"PASA: los {total} ficheros de salida escriben los números en castellano.")
