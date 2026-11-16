"""Ejecuta el diseño factorial completo 3 (temperatura) x 3 (tamaño) x 10 réplicas.

Uso:  python src/experimento.py
Salida: resultados/corridas.csv y resultados/parametros.json
"""
import csv
import json
import math
import os
import random
import statistics
import time

import config as cfg
from recocido import delta_2opt, recocido_simulado
from tsp import generar_instancia, matriz_distancias, optimo_held_karp

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RES = os.path.join(RAIZ, "resultados")


def fracciones_opcion_a(r):
    """Niveles de la Opción A como fracción del rango [T_min, T_max]."""
    return {"bajo": 0.3 * r, "medio": 0.3 * r + 0.3, "alto": 0.3 * r + 0.6}


def mediana_deltas_positivos(instancias, semilla):
    """Mediana de los incrementos de costo (Delta > 0) de movimientos 2-opt
    aplicados a rutas aleatorias, agrupando las tres instancias."""
    rng = random.Random(semilla)
    deltas = []
    for D in instancias.values():
        n = len(D)
        for _ in range(cfg.MUESTRAS_DELTA):
            ruta = list(range(n))
            rng.shuffle(ruta)
            i, k = sorted(rng.sample(range(1, n), 2))
            d = delta_2opt(ruta, D, i, k)
            if d > 0:
                deltas.append(d)
    return statistics.median(deltas)


def main():
    os.makedirs(DIR_RES, exist_ok=True)

    # 1) Número aleatorio r: se genera UNA sola vez con la matrícula como semilla
    random.seed(cfg.MATRICULA)
    r = random.random()

    # 2) Una instancia por tamaño (semilla fija y reproducible)
    instancias, optimos = {}, {}
    for n in cfg.TAMANOS:
        D = matriz_distancias(generar_instancia(n, cfg.MATRICULA + n))
        instancias[n] = D
        optimos[n] = optimo_held_karp(D)

    # 3) Rango de temperatura [T_min, T_max] a partir de la probabilidad de aceptación
    delta_med = mediana_deltas_positivos(instancias, cfg.MATRICULA)
    t_min = round(-delta_med / math.log(cfg.PROB_ACEPTACION_MIN))
    t_max = int(round(-delta_med / math.log(cfg.PROB_ACEPTACION_MAX), -1))

    # 4) Niveles de temperatura: T = T_min + fracción * (T_max - T_min)
    fracs = fracciones_opcion_a(r)
    temps = {niv: round(t_min + f * (t_max - t_min), 2) for niv, f in fracs.items()}

    parametros = {
        "matricula": cfg.MATRICULA, "r": r, "delta_mediana": delta_med,
        "T_min": t_min, "T_max": t_max, "fracciones": fracs, "temperaturas": temps,
        "optimos": optimos, "alfa": cfg.ALFA, "etapas": cfg.ETAPAS,
        "iter_por_etapa": {n: cfg.ITER_POR_ETAPA(n) for n in cfg.TAMANOS},
        "replicas": cfg.REPLICAS,
    }
    with open(os.path.join(DIR_RES, "parametros.json"), "w", encoding="utf-8") as f:
        json.dump(parametros, f, indent=2, ensure_ascii=False)

    print(f"r = {r:.6f}")
    print(f"Mediana Delta>0 = {delta_med:.3f}  ->  [T_min, T_max] = [{t_min}, {t_max}]")
    for niv in cfg.NIVELES:
        print(f"  {niv:5s}: fracción = {fracs[niv]:.4f}  ->  T0 = {temps[niv]}")

    # 5) Corridas: 3 tamaños x 3 temperaturas x 10 réplicas, semilla distinta por corrida
    filas, idx = [], 0
    for n in cfg.TAMANOS:
        for niv in cfg.NIVELES:
            for rep in range(1, cfg.REPLICAS + 1):
                semilla = cfg.MATRICULA * 100 + idx
                idx += 1
                t0 = time.perf_counter()
                costo, ruta = recocido_simulado(instancias[n], temps[niv], semilla,
                                                alfa=cfg.ALFA, etapas=cfg.ETAPAS,
                                                iter_por_etapa=cfg.ITER_POR_ETAPA(n))
                seg = time.perf_counter() - t0
                filas.append({
                    "tamano": n, "nivel": niv, "fraccion": round(fracs[niv], 6),
                    "temperatura": temps[niv], "replica": rep, "semilla": semilla,
                    "costo": round(costo, 6), "optimo": round(optimos[n], 6),
                    "gap_pct": round(100 * (costo / optimos[n] - 1), 6),
                    "tiempo_s": round(seg, 6), "ruta": "-".join(map(str, ruta)),
                })
    with open(os.path.join(DIR_RES, "corridas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"{len(filas)} corridas guardadas en resultados/corridas.csv")


if __name__ == "__main__":
    main()
