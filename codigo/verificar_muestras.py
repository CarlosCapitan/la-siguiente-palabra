#!/usr/bin/env python3
"""Comprueba que todas las citas en bloque de un capítulo son literales: que aparecen tal cual
en la salida del script que las generó. Uso: verificar_muestras.py capitulo.md salida.txt"""
import re, sys, io
def normalizar(t):
    """Compara ignorando espacios y el separador decimal: el libro escribe 0,018 y el
    script imprime 0.018. Localizar la coma decimal no es inventarse un dato."""
    t = re.sub(r'(?<=\d),(?=\d)', '.', t)
    return re.sub(r'\s+', ' ', t)

cap = normalizar(io.open(sys.argv[1], encoding='utf-8').read())
sal = normalizar(io.open(sys.argv[2], encoding='utf-8').read())
fallos = 0
for b in re.findall(r'((?:^> .*\n)+)', cap, re.M):
    t = re.sub(r'\s+', ' ', re.sub(r'^> ', '', b, flags=re.M)).strip()
    if t[:70] not in sal:
        print(f"NO LITERAL: «{t[:60]}…»"); fallos += 1
print("FALLA" if fallos else "PASA: todas las citas son literales")
sys.exit(1 if fallos else 0)
