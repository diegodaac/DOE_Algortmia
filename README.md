# Tarea 3: Ajuste de parámetros con Diseño de Experimentos (DOE)

**Curso:** Algoritmos y Optimización, FCFM UANL
**Alumno:** Diego Alejandro Alvarado Castañón (matrícula 1971967)

Ajuste de la temperatura inicial T0 del Recocido Simulado con vecindad 2-opt para el
TSP (continuación de la Tarea 2, Ejercicio 4) mediante un **diseño factorial completo
3 x 3 (Opción A)**: factor temperatura (3 niveles), bloque tamaño n en {5, 10, 15},
10 réplicas por punto de diseño (90 corridas).

## Estructura

```
tarea3_doe/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py        # parámetros fijos del experimento
│   ├── tsp.py           # instancias, costo de ruta y óptimo exacto (Held-Karp)
│   ├── recocido.py      # Recocido Simulado con movimientos 2-opt
│   ├── experimento.py   # genera r, niveles de temperatura y ejecuta las 90 corridas
│   └── analisis.py      # ANOVA de dos factores, bootstrap y gráficas
├── resultados/          # CSV con corridas, tablas ANOVA, bootstrap y salida de texto
├── figuras/             # efectos principales, interacción y diagramas de caja
└── reporte/             # reporte final en PDF
```

## Reproducir

```bash
pip install -r requirements.txt
python src/experimento.py   # -> resultados/corridas.csv, resultados/parametros.json
python src/analisis.py      # -> resultados/*.csv, resultados/salida_analisis.txt, figuras/*.png
```

Todo es determinista: r se obtiene con `random.seed(1971967); r = random.random()`,
las instancias usan la semilla `1971967 + n` y cada corrida tiene su propia semilla
(`197196700 + índice`), por lo que volver a ejecutar produce exactamente los mismos
resultados.

## Resumen de resultados

| n | T0 con mejor desempeño | Mediana bootstrap | IC 95 % |
|---|---|---|---|
| 5 | empate en los tres niveles | 239.922 | [239.922, 239.922] |
| 10 | 136.11 (bajo) | 306.901 | [306.901, 306.901] |
| 15 | 136.11 (bajo) | 316.073 | [315.957, 316.377] |

El análisis completo está en `reporte/Tarea3_DOE_1971967.pdf`.
