"""Parámetros fijos del experimento (Tarea 3, Opción A: factorial completo 3 x 3)."""
MATRICULA = 1971967

TAMANOS = (5, 10, 15)          # bloque: tamaño de la instancia TSP
REPLICAS = 10                  # corridas por punto de diseño
NIVELES = ("bajo", "medio", "alto")

# Parámetros del Recocido Simulado que se mantienen constantes (no son factor)
ALFA = 0.95                    # enfriamiento geométrico
ETAPAS = 100                   # etapas de temperatura
ITER_POR_ETAPA = lambda n: 2 * n   # movimientos 2-opt propuestos por etapa

# Criterio para fijar el rango [T_min, T_max]: probabilidad de aceptar un
# empeoramiento típico (mediana de los Delta > 0 de movimientos 2-opt aleatorios)
PROB_ACEPTACION_MIN = 0.01     # en T_min el algoritmo es prácticamente codicioso
PROB_ACEPTACION_MAX = 0.99     # en T_max es prácticamente una caminata aleatoria
MUESTRAS_DELTA = 5000

BOOTSTRAP_B = 10_000
ALFA_SIGNIF = 0.05
