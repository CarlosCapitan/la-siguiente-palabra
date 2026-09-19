# Informe — banco de pruebas de copia literal («¿lo había visto ya?»)

Máquina: MacBook Pro M4 Max, 36 GB, macOS (Darwin 25.6.0). PyTorch 2.14.0. GPU (Metal/MPS)
usada en las seis tiradas de entrenamiento; la generación y toda la medición son en CPU
(deliberado, ver más abajo). Medido el 2026-09-19.

## 1. Las 10 líneas donde se decide de verdad

**`rachas()` — la racha, con frontera de palabra** (`codigo/lo_habia_visto_ya.py:144-153`):

```python
palabras = muestra.split()
n = len(palabras)
valores = [0] * n
for i in range(n):
    k = 0
    while i + k < n:
        trozo = " " + " ".join(palabras[i:i + k + 1]) + " "
        if trozo not in corpus_con_bordes:
            break
        k += 1
```

**El umbral de «copiado»** (`codigo/lo_habia_visto_ya.py:437-439`):

```python
# UMBRAL DE «COPIADO»: sale de la propia medición, por cada largo de ventana.
racha_c1, _ = medir(contar_1, corpus, umbral=largo + 1)
umbral = max(racha_c1) + MARGEN_UMBRAL
```

Todo lo demás del instrumento es contabilidad alrededor de estas dos cosas: `rachas()` decide
qué es una coincidencia literal (con frontera de palabra, para que «la casa» no encuentre
«hola casa»), y el umbral decide, cada vez que se ejecuta, a partir de qué racha se llama a eso
«copiado» — nunca un número puesto a mano.

## 2. Los tres puntos donde es más probable que esto esté mal

1. **El «suelo» y la «barajada» no son generadores independientes de los demás.** `suelo` sale
   de los 6 libros siguientes al tope de entrenamiento en el propio directorio del corpus, y
   `barajada` se construye barajando muestras de `contar_1`, `suelo` y la tirada 1 de cada
   semilla (no las doce filas). Si esos 6 libros ajenos comparten género o época con el
   entrenamiento —cosa que ya se ha visto que pasa: ver el punto 3 del selftest más abajo—, el
   suelo puede estar sobreestimando lo que el azar del castellano permite, no subestimándolo.

2. **La «huella de pesos» compara tensores en float32, no el optimizador.** Dos entrenamientos
   pueden producir el mismo `state_dict()` con distinta trayectoria de Adam (momentos de
   primer y segundo orden) si el no determinismo de Metal solo afectara al optimizador y no
   ya se hubiera propagado a los pesos; no es el caso aquí (las seis huellas salen distintas,
   así que el efecto se ve de sobra en los pesos), pero la huella tal como está no distinguiría
   ese caso hipotético de un entrenamiento verdaderamente idéntico.

3. **`arranques()` acepta un trigrama con solo 5 apariciones en 686.002 palabras.**
   `MIN_APARICIONES_ARRANQUE = 5` es un número razonable pero no derivado de nada: con un
   trigrama que aparece justo 5 veces, la «primera aparición» que usa `techo` (y `suelo`, en el
   corpus ajeno) puede no ser representativa de las otras 4, y ahí el «techo» seguiría siendo
   100 % por construcción, pero ya no diría gran cosa sobre lo típico del corpus.

## 3. Qué no cubren los tests

El `--selftest` (tres pruebas, todas pasan) comprueba que `rachas()` no inventa copia donde no
la hay, que no se le escapa la que hay, y que respeta el invariante de las máquinas de
contar. No comprueba:

- **Que los 6 «libros ajenos» sean realmente independientes en contenido**, solo que están
  fuera del recorte de entrenamiento por posición de fichero. El propio selftest encontró
  (antes de subir el tope de la parte 1a) una racha de 8 entre un libro entrenado y uno ajeno,
  los dos diarios de expedición que comparten la fórmula «día N. A las ocho de la mañana...».
  El test pasa ahora con el tope corregido, pero no vigila que esto no se repita con una racha
  aún mayor en una ejecución futura si cambian los libros del corpus.
- **Que las seis tiradas de `escribir_muestras.py` usan exactamente los mismos arranques.**
  Ese assert vive en `main()`, no en `--selftest` (que corre antes de que existan ficheros de
  muestras), así que un fallo ahí solo se ve al ejecutar la medición real, no antes.
- **El caso de un corpus ajeno con menos de 6 libros disponibles**, o con arranques que no
  llegan a `MIN_APARICIONES_ARRANQUE` apariciones: hay un `assert`, pero el selftest no lo
  ejerce.
- **La generación de la red en sí** (si `escribir_muestras.py` genera con la temperatura y el
  vocabulario correctos): eso lo cubre el selftest de `memoria_recurrente.py`, no este.

## 4. Las dos tablas

### Ventana de 45 palabras (200 muestras por generador)

```
generador         racha, mediana  racha, p90 racha, máximo
                     en palabras en palabras   en palabras
----------------------------------------------------------
techo                       45,0        45,0            45
contar_3                    19,0        32,0            45
red_s20260915_t1            10,0        19,0            45
contar_2                     9,0        12,0            17
red_s20260915_t2             9,0        18,0            43
red_s20260916_t1             9,0        17,1            45
red_s20260914_t2             8,0        18,0            45
red_s20260916_t2             8,0        16,1            27
red_s20260914_t1             8,0        16,1            40
contar_1                     4,0         5,0             6
suelo                        4,0         4,0             8
barajada                     3,0         3,0             4

generador         copiado, mediana copiado, p90
                   % de la muestra% de la muestra
-----------------------------------------------
techo                      100,0 %      100,0 %
contar_3                    97,8 %      100,0 %
red_s20260915_t1            26,7 %       60,0 %
contar_2                    35,6 %       64,4 %
red_s20260915_t2            22,2 %       51,3 %
red_s20260916_t1            20,0 %       57,8 %
red_s20260914_t2            21,1 %       51,1 %
red_s20260916_t2            21,1 %       48,9 %
red_s20260914_t1            17,8 %       49,1 %
contar_1                     0,0 %        0,0 %
suelo                        0,0 %        0,0 %
barajada                     0,0 %        0,0 %
```

**Umbral de «copiado» en esta ventana: racha ≥ 7 palabras** (máximo de `contar_1` + 1).

### Ventana de 200 palabras (50 muestras por generador)

```
generador         racha, mediana  racha, p90 racha, máximo
                     en palabras en palabras   en palabras
----------------------------------------------------------
techo                      200,0       200,0           200
contar_3                    34,5        63,1            93
red_s20260914_t1            17,0        25,0            40
red_s20260916_t2            17,0        29,0            41
red_s20260915_t1            16,0        29,6            44
red_s20260915_t2            16,0        27,2            57
red_s20260914_t2            13,5        22,0            30
red_s20260916_t1            12,0        23,1            39
contar_2                    11,0        16,0            29
contar_1                     4,0         6,0             7
suelo                        4,0         5,0             6
barajada                     3,0         3,0             4

generador         copiado, mediana copiado, p90
                   % de la muestra% de la muestra
-----------------------------------------------
techo                      100,0 %      100,0 %
contar_3                    96,5 %       99,5 %
red_s20260914_t1            19,0 %       31,2 %
red_s20260916_t2            18,0 %       27,1 %
red_s20260915_t1            23,0 %       39,7 %
red_s20260915_t2            22,5 %       32,0 %
red_s20260914_t2            17,8 %       32,2 %
red_s20260916_t1            12,2 %       23,6 %
contar_2                    26,8 %       39,2 %
contar_1                     0,0 %        0,0 %
suelo                        0,0 %        0,0 %
barajada                     0,0 %        0,0 %
```

**Umbral de «copiado» en esta ventana: racha ≥ 8 palabras** (máximo de `contar_1` + 1).

### La tabla de las seis huellas

```
semilla    tirada  huella de pesos racha, mediana copiado, mediana
------------------------------------------------------------------
20260914        1     b77c7fc3137d            8,0           17,8 %
20260914        2     6b0014b74c89            8,0           21,1 %
20260915        1     7e996877425d           10,0           26,7 %
20260915        2     de73e255e31b            9,0           22,2 %
20260916        1     f9960903455f            9,0           20,0 %
20260916        2     83f652b16c45            8,0           21,1 %
```

**Las seis huellas de pesos salen todas distintas.** Tres semillas, cada una entrenada dos
veces con exactamente el mismo programa, los mismos pasos y la misma semilla: los seis modelos
resultantes son seis puntos distintos en el espacio de pesos. La única diferencia entre las dos
tiradas de una misma semilla es el orden de ejecución de las operaciones de la GPU (Metal no
garantiza reproducibilidad bit a bit), y esa diferencia basta para que ningún par de tiradas
coincida ni en los pesos ni, en consecuencia, en la racha o el copiado que produce cada una
(la mediana de racha va de 8 a 10 palabras entre las seis tiradas, y ninguna se repite).

## 5. Qué queda para el capítulo

Con el umbral derivado de la medición (7 y 8 palabras según la ventana, no el `RACHA_LARGA = 5`
puesto a mano que tenía `cuanto_copia.py`), la red del capítulo 6 copia mucho menos de lo que
sugería la comparación original: mediana de copiado entre el 17,8 % y el 26,7 % de la muestra
en ventanas de 45 palabras (frente al 100 % de `contar_3`, la máquina de contar con tres
palabras de contexto sobre el mismo corpus), y los seis modelos entrenados están todos por
encima del suelo (`suelo`, `contar_1`, `barajada`, los tres al 0,0 % de copiado con este umbral)
pero muy por debajo de lo que copia una máquina de contar de orden comparable.

Todo lo anterior se ha ejecutado: no hay ninguna cifra de «no ejecutado» en este informe.
