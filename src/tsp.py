"""Utilidades para el Problema del Agente Viajero (TSP) euclidiano."""
import itertools
import math
import random


def generar_instancia(n, semilla, lado=100.0):
    """Genera n ciudades con coordenadas uniformes en [0, lado]^2."""
    rng = random.Random(semilla)
    return [(rng.uniform(0, lado), rng.uniform(0, lado)) for _ in range(n)]


def matriz_distancias(ciudades):
    n = len(ciudades)
    return [[math.dist(ciudades[i], ciudades[j]) for j in range(n)] for i in range(n)]


def costo_ruta(ruta, D):
    """Longitud del ciclo hamiltoniano definido por la permutación `ruta`."""
    return sum(D[ruta[i]][ruta[(i + 1) % len(ruta)]] for i in range(len(ruta)))


def optimo_held_karp(D):
    """Costo óptimo exacto por programación dinámica de Held-Karp, O(n^2 2^n).

    Se usa solo como referencia para calcular la brecha (gap) de cada corrida.
    """
    n = len(D)
    if n <= 3:
        return costo_ruta(list(range(n)), D)
    # dp[(mascara, j)]: costo mínimo de salir de 0, visitar 'mascara' y terminar en j
    dp = {}
    for j in range(1, n):
        dp[(1 << j, j)] = D[0][j]
    for tam in range(2, n):
        for sub in itertools.combinations(range(1, n), tam):
            mascara = 0
            for v in sub:
                mascara |= 1 << v
            for j in sub:
                previa = mascara ^ (1 << j)
                dp[(mascara, j)] = min(dp[(previa, k)] + D[k][j] for k in sub if k != j)
    completa = (1 << n) - 2
    return min(dp[(completa, j)] + D[j][0] for j in range(1, n))
