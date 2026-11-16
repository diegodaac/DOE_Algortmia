"""Recocido Simulado con vecindad 2-opt para TSP (base: Tarea 2, Ejercicio 4)."""
import math
import random

from tsp import costo_ruta


def delta_2opt(ruta, D, i, k):
    """Cambio de costo al invertir el segmento ruta[i..k] (movimiento 2-opt).

    Se reemplazan las aristas (a,b) y (c,d) por (a,c) y (b,d), donde
    a = ruta[i-1], b = ruta[i], c = ruta[k], d = ruta[k+1].
    """
    n = len(ruta)
    a, b = ruta[i - 1], ruta[i]
    c, d = ruta[k], ruta[(k + 1) % n]
    return D[a][c] + D[b][d] - D[a][b] - D[c][d]


def recocido_simulado(D, T0, semilla, alfa=0.95, etapas=100, iter_por_etapa=None):
    """Ejecuta Recocido Simulado con enfriamiento geométrico T_{k+1} = alfa * T_k.

    Parámetros
    ----------
    D : matriz de distancias.
    T0 : temperatura inicial (factor del experimento).
    semilla : semilla de la corrida (réplica).
    alfa : factor de enfriamiento geométrico (fijo).
    etapas : número de etapas de temperatura (fijo).
    iter_por_etapa : movimientos 2-opt propuestos por etapa (por defecto 2n).

    El presupuesto total de evaluaciones es fijo (etapas * iter_por_etapa), de modo
    que T0 es la única fuente de diferencia entre niveles.

    Regresa (mejor_costo, mejor_ruta).
    """
    rng = random.Random(semilla)
    n = len(D)
    if iter_por_etapa is None:
        iter_por_etapa = 2 * n

    ruta = list(range(n))
    rng.shuffle(ruta)
    costo = costo_ruta(ruta, D)
    mejor_ruta, mejor_costo = ruta[:], costo

    T = T0
    for _ in range(etapas):
        for _ in range(iter_por_etapa):
            # Elegir dos posiciones 1 <= i < k <= n-1 (no se invierte el ciclo completo)
            i, k = sorted(rng.sample(range(1, n), 2)) if n > 3 else (1, 2)
            delta = delta_2opt(ruta, D, i, k)
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                ruta[i:k + 1] = reversed(ruta[i:k + 1])
                costo += delta
                if costo < mejor_costo - 1e-12:
                    mejor_costo, mejor_ruta = costo, ruta[:]
        T *= alfa
    return mejor_costo, mejor_ruta
