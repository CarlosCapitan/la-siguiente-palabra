#!/usr/bin/env python3
"""Comprueba que todos los programas tienen selftest, y opcionalmente que lo pasan.

    verificar_selftests.py .              rápido: que cada programa tenga --selftest
    verificar_selftests.py . --ejecutar   lento: los ejecuta todos (más de diez minutos)
    verificar_selftests.py --selftest

Nace del fallo 4.29. `reparto_atencion.py` —el programa que produce TODAS las cifras de atención
del capítulo 8, el que el libro llama su corazón— suspendía su propio test nulo, de forma
determinista, en cualquier máquina, y llevaba así desde el primer día. Nadie lo había ejecutado
porque el programa «funciona»: la ruta que imprime los números del libro iba tan tranquila.

El barrido del 18 de septiembre: 31 de 32 pasan, falla ése.

La comprobación rápida es estática y tarda un segundo, así que se puede pasar siempre. La lenta
ejecuta de verdad y tarda más de diez minutos: se pasa al cerrar un capítulo, o cuando se toca
un programa. Un verificador que tarda diez minutos por defecto no lo pasa nadie, y un verificador
que no se pasa es peor que no tenerlo."""
import os, subprocess, sys

# Estos no llevan selftest a propósito, y el motivo está aquí para que no se discuta cada vez.
SIN_SELFTEST = {
    'formato.py':          'no mide nada: es el formato de imprenta compartido',
    'descargar_corpus.py': 'no calcula: baja ficheros de internet y no hay nada que comprobar',
}
# Estos solo corren en el portátil del autor (bibliotecas de su tarjeta gráfica).
SOLO_EN_EL_MAC = {'comprimir.py': 'necesita mlx_lm, que solo está en el Mac'}
LIMITE = 420   # segundos por programa


def programas(carpeta):
    return sorted(f for f in os.listdir(carpeta)
                  if f.endswith('.py') and not f.startswith('verificar_'))


def tiene_selftest(ruta):
    return '--selftest' in open(ruta, encoding='utf-8').read()


def revisar_estatico(nombres, con_selftest):
    """nombres: lista de ficheros. con_selftest: conjunto de los que lo tienen.
    Devuelve los que deberían tenerlo y no lo tienen."""
    return [n for n in nombres if n not in SIN_SELFTEST and n not in con_selftest]


def selftest():
    """Tres pruebas sobre lo único que decide en su modo rápido: si a un programa le falta el
    selftest que debería tener."""
    fallos = []

    # 1. TEST NULO — todo en orden, incluidos los dos que no llevan selftest a propósito. Si
    #    aquí saltara algo, el verificador pediría añadir selftest a formato.py, que no mide
    #    nada, y acabaríamos desconectándolo.
    sale = revisar_estatico(['perceptron.py', 'formato.py', 'descargar_corpus.py'],
                            {'perceptron.py'})
    print(f"[1] test nulo         todo en orden: {len(sale)} sin selftest"
          + (f" ({sale})" if sale else "") + "; los dos exentos, exentos")
    if sale:
        fallos.append(f"test nulo: pide selftest a los que están exentos: {sale}")

    # 2. SEÑAL IMPLANTADA — un programa de medición sin selftest. Tiene que cazarlo.
    sale = revisar_estatico(['perceptron.py', 'inventado.py'], {'perceptron.py'})
    print(f"[2] señal implantada  un programa de medición sin selftest: {sale}")
    if sale != ['inventado.py']:
        fallos.append(f"señal implantada: esperaba ['inventado.py'] y salió {sale}")

    # 3. INVARIANTE DEL DOMINIO — la exención va por nombre exacto, no por parecido. Un
    #    «formato_nuevo.py» no hereda la exención de «formato.py»: si la heredara, bastaría
    #    llamar a un programa de cierta manera para saltarse la comprobación.
    sale = revisar_estatico(['formato_nuevo.py', 'formato.py'], set())
    print(f"[3] invariante        «formato_nuevo.py» NO hereda la exención de «formato.py»: "
          f"{'bien' if sale == ['formato_nuevo.py'] else 'MAL'}")
    if sale != ['formato_nuevo.py']:
        fallos.append(f"invariante: la exención se hereda por parecido de nombre: {sale}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


if '--selftest' in sys.argv:
    sys.exit(selftest())

carpeta = os.path.dirname(os.path.abspath(__file__))
nombres = programas(carpeta)
con = {n for n in nombres if tiene_selftest(os.path.join(carpeta, n))}
faltan = revisar_estatico(nombres, con)

for n in faltan:
    print(f"  {n:<32} no tiene --selftest")

suspenden = []
if '--ejecutar' in sys.argv:
    print(f"\nEjecutando {len(con)} selftest. Esto tarda más de diez minutos.\n")
    for n in sorted(con):
        if n in SOLO_EN_EL_MAC:
            print(f"  {n:<32} SALTADO: {SOLO_EN_EL_MAC[n]}")
            continue
        r = subprocess.run([sys.executable, n, '--selftest'], cwd=carpeta,
                           capture_output=True, text=True, timeout=LIMITE)
        ok = 'SELFTEST:' in r.stdout and 'FALLA' not in r.stdout
        print(f"  {n:<32} {'pasa' if ok else 'SUSPENDE'}")
        if not ok:
            suspenden.append(n)
            for l in (r.stdout + r.stderr).splitlines():
                if 'FALLA' in l or 'Error' in l:
                    print(f"      {l.strip()[:90]}")

if faltan or suspenden:
    print(f"\nFALLA: {len(faltan)} programa(s) sin selftest, {len(suspenden)} que suspenden.")
    print("Un programa cuyo selftest suspende no puede avalar ni una cifra del libro.")
    sys.exit(1)
if '--ejecutar' in sys.argv:
    print(f"\nPASA: los {len(con) - len(set(SOLO_EN_EL_MAC) & con)} selftest ejecutados pasan.")
else:
    print(f"PASA: los {len(nombres) - len(set(SIN_SELFTEST) & set(nombres))} programas de medición "
          f"tienen selftest. Para ejecutarlos: --ejecutar.")
