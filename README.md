# Código compañero

Repositorio público con el código, los datos y las figuras del libro. El libro cuenta qué salió;
aquí está cómo se obtuvo, para que cualquiera pueda repetirlo.

Máquina de referencia: MacBook Pro M4 Max, 36 GB de memoria unificada, macOS.

- `codigo/` — un fichero por experimento, sin rutas absolutas, con semilla fija.
- `datos/` — entradas reproducibles o el guion para descargarlas.
- `figuras/` — las figuras del libro, generadas por el código de `codigo/`.

## Dónde se ejecuta cada cosa

Las mediciones se dividen en dos clases y no se ejecutan en el mismo sitio.

**Independientes de la máquina.** Aciertos, recuentos, proporciones, vecinas más próximas,
comprobaciones de gradiente. Con la misma semilla dan el mismo resultado en cualquier ordenador,
así que da igual dónde se corran.

**Dependientes de la máquina.** Cualquier cifra que lleve un tiempo dentro: cuánto entrena un
modelo en X minutos, cuántos pasos da, cuánto tarda algo. Estas **solo** valen si se ejecutan en
el portátil del que habla el libro, un MacBook Pro M4 Max con 36 GB, y usando su GPU.

Para eso está `codigo/medir_en_mac.sh`, que hay que lanzar desde el Terminal de macOS —no desde
la máquina virtual Linux de la aplicación de escritorio, que no alcanza la GPU del Mac—:

    cd ~/Documents/libro-ia-publico/codigo
    bash medir_en_mac.sh

Deja los resultados en `resultados_mac.txt`, con la máquina y la versión de las bibliotecas
anotadas en la cabecera.
