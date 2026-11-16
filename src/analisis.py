
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import config as cfg

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RES = os.path.join(RAIZ, "resultados")
DIR_FIG = os.path.join(RAIZ, "figuras")

AZUL, NARANJA, VERDE = "#1F3864", "#C55A11", "#548235"
COLORES = {5: VERDE, 10: NARANJA, 15: AZUL}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def bootstrap_mediana(x, B, rng, nivel=0.95):
    """Mediana bootstrap (media de las medianas remuestreadas), mediana muestral
    e intervalo percentil al `nivel` de confianza."""
    x = np.asarray(x, dtype=float)
    muestras = rng.choice(x, size=(B, len(x)), replace=True)
    medianas = np.median(muestras, axis=1)
    a = (1 - nivel) / 2
    return {
        "mediana_muestral": float(np.median(x)),
        "mediana_bootstrap": float(np.mean(medianas)),
        "ic_inf": float(np.quantile(medianas, a)),
        "ic_sup": float(np.quantile(medianas, 1 - a)),
    }


def main():
    os.makedirs(DIR_FIG, exist_ok=True)
    df = pd.read_csv(os.path.join(DIR_RES, "corridas.csv"))
    with open(os.path.join(DIR_RES, "parametros.json"), encoding="utf-8") as f:
        par = json.load(f)
    temps = [par["temperaturas"][niv] for niv in cfg.NIVELES]
    salida = []

    def log(txt=""):
        print(txt)
        salida.append(txt)

    # ---------------- 1) ANOVA de dos factores ----------------
    modelo = ols("costo ~ C(temperatura) * C(tamano)", data=df).fit()
    anova = sm.stats.anova_lm(modelo, typ=2)
    anova.to_csv(os.path.join(DIR_RES, "anova_dos_factores.csv"))
    log("=== ANOVA de dos factores (typ=2), respuesta: costo ===")
    log(anova.to_string(float_format=lambda v: f"{v:.6g}"))
    sig = {fuente: anova.loc[fuente, "PR(>F)"] < cfg.ALFA_SIGNIF for fuente in anova.index[:-1]}
    log("Significativo (p < 0.05): " + ", ".join(f"{k}: {'sí' if v else 'no'}" for k, v in sig.items()))

    # Supuestos del modelo (diagnóstico)
    sw = stats.shapiro(modelo.resid)
    grupos = [g["costo"].values for _, g in df.groupby(["tamano", "temperatura"])]
    lev = stats.levene(*grupos, center="median")
    log(f"\nShapiro-Wilk residuos: W = {sw.statistic:.4f}, p = {sw.pvalue:.3g}")
    log(f"Levene (Brown-Forsythe) entre las 9 celdas: W = {lev.statistic:.4f}, p = {lev.pvalue:.3g}")

    # Robustez: el mismo modelo de dos factores ajustado sobre los rangos del costo
    # (Conover e Iman, 1981), que no supone normalidad ni es sensible a valores atípicos
    df["rango"] = df["costo"].rank()
    anova_r = sm.stats.anova_lm(ols("rango ~ C(temperatura) * C(tamano)", data=df).fit(), typ=2)
    anova_r.to_csv(os.path.join(DIR_RES, "anova_rangos.csv"))
    log("\n=== ANOVA sobre rangos (verificación no paramétrica) ===")
    log(anova_r.to_string(float_format=lambda v: f"{v:.6g}"))

    # Sensibilidad: ¿qué tanto depende la conclusión paramétrica de un solo valor extremo?
    # (solo diagnóstico; las decisiones se toman con los 90 datos completos)
    extremo = df["costo"].idxmax()
    sub = df.drop(index=extremo)
    anova_s = sm.stats.anova_lm(ols("costo ~ C(temperatura) * C(tamano)", data=sub).fit(), typ=2)
    anova_s.to_csv(os.path.join(DIR_RES, "anova_sensibilidad_sin_extremo.csv"))
    fila = df.loc[extremo]
    log(f"\n=== Sensibilidad: ANOVA paramétrico sin la corrida extrema "
        f"(n = {fila.tamano}, T0 = {fila.temperatura}, costo = {fila.costo:.2f}) ===")
    log(anova_s.to_string(float_format=lambda v: f"{v:.6g}"))

    # ---------------- 2) Efectos simples: temperatura dentro de cada tamaño ----------------
    log("\n=== Efecto de la temperatura dentro de cada tamaño ===")
    simples = []
    for n, g in df.groupby("tamano"):
        muestras = [g.loc[g.temperatura == t, "costo"].values for t in temps]
        if np.allclose(g["costo"].std(), 0):
            log(f"n = {n}: las 30 corridas obtienen el mismo costo ({g.costo.iloc[0]:.2f}); varianza nula, "
                f"no hay efecto que probar.")
            simples.append({"tamano": n, "F": np.nan, "p_anova": np.nan, "H": np.nan, "p_kruskal": np.nan})
            continue
        f1 = stats.f_oneway(*muestras)
        kw = stats.kruskal(*muestras)
        log(f"n = {n}: ANOVA un factor F = {f1.statistic:.3f}, p = {f1.pvalue:.4g} | "
            f"Kruskal-Wallis H = {kw.statistic:.3f}, p = {kw.pvalue:.4g}")
        tk = pairwise_tukeyhsd(g["costo"], g["temperatura"], alpha=cfg.ALFA_SIGNIF)
        log(str(tk.summary()))
        simples.append({"tamano": n, "F": f1.statistic, "p_anova": f1.pvalue,
                        "H": kw.statistic, "p_kruskal": kw.pvalue})
    pd.DataFrame(simples).to_csv(os.path.join(DIR_RES, "efectos_simples.csv"), index=False)

    # ---------------- 3) Bootstrap de la mediana por punto de diseño ----------------
    rng = np.random.default_rng(cfg.MATRICULA)
    filas = []
    for n in cfg.TAMANOS:
        for niv, t in zip(cfg.NIVELES, temps):
            x = df.loc[(df.tamano == n) & (df.temperatura == t), "costo"].values
            b = bootstrap_mediana(x, cfg.BOOTSTRAP_B, rng)
            filas.append({"tamano": n, "nivel": niv, "temperatura": t,
                          "media": x.mean(), "desv_est": x.std(ddof=1), **b})
    boot = pd.DataFrame(filas)
    boot.to_csv(os.path.join(DIR_RES, "bootstrap_medianas.csv"), index=False)
    log("\n=== Mediana bootstrap (B = %d) e IC 95%% percentil por punto de diseño ===" % cfg.BOOTSTRAP_B)
    log(boot.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------------- 4) Temperatura óptima por tamaño ----------------
    # Criterio: menor mediana bootstrap; desempate por menor media y luego menor desviación.
    log("\n=== Temperatura con mejor desempeño por tamaño ===")
    mejores = []
    for n, g in boot.groupby("tamano"):
        g = g.round({"mediana_bootstrap": 6, "media": 6, "desv_est": 6})
        orden = g.sort_values(["mediana_bootstrap", "media", "desv_est"])
        best = orden.iloc[0]
        empate = (g[["mediana_bootstrap", "media", "desv_est"]].nunique() == 1).all()
        mejores.append({**best.to_dict(), "empate_total": bool(empate)})
        log(f"n = {n}: T* = {best.temperatura} ({best.nivel}); mediana bootstrap = "
            f"{best.mediana_bootstrap:.3f}, IC 95% = [{best.ic_inf:.3f}, {best.ic_sup:.3f}]"
            + ("  (empate en los tres niveles)" if empate else ""))
    pd.DataFrame(mejores).to_csv(os.path.join(DIR_RES, "temperatura_optima.csv"), index=False)

    with open(os.path.join(DIR_RES, "salida_analisis.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(salida) + "\n")

    # ---------------- 5) Gráficas ----------------
    graficas(df, temps, boot)


def graficas(df, temps, boot):
    etiquetas_t = [f"{t:g}" for t in temps]
    media_gral = df["costo"].mean()

    # Efectos principales (medias marginales del costo)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    mt = df.groupby("temperatura")["costo"].mean().reindex(temps)
    mn = df.groupby("tamano")["costo"].mean()
    ax[0].plot(etiquetas_t, mt.values, "o-", color=AZUL, lw=2)
    ax[0].set_xlabel("Temperatura inicial T0")
    ax[0].set_ylabel("Costo medio de la ruta")
    ax[0].set_title("Efecto principal: temperatura")
    ax[1].plot([str(n) for n in mn.index], mn.values, "o-", color=NARANJA, lw=2)
    ax[1].set_xlabel("Tamaño del problema n")
    ax[1].set_ylabel("Costo medio de la ruta")
    ax[1].set_title("Efecto principal: tamaño")
    for a in ax:
        a.axhline(media_gral, color="gray", ls="--", lw=1)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, "efectos_principales.png"), dpi=200)
    plt.close(fig)

    # Interacción temperatura x tamaño (costo)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n, g in df.groupby("tamano"):
        m = g.groupby("temperatura")["costo"].mean().reindex(temps)
        ax.plot(etiquetas_t, m.values, "o-", lw=2, color=COLORES[n], label=f"n = {n}")
    ax.set_xlabel("Temperatura inicial T0")
    ax.set_ylabel("Costo medio de la ruta")
    ax.set_title("Interacción temperatura x tamaño")
    ax.legend(title="Tamaño", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, "interaccion.png"), dpi=200)
    plt.close(fig)

    # Interacción en brecha porcentual respecto al óptimo (escala común entre tamaños)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n, g in df.groupby("tamano"):
        m = g.groupby("temperatura")["gap_pct"].mean().reindex(temps)
        ax.plot(etiquetas_t, m.values, "o-", lw=2, color=COLORES[n], label=f"n = {n}")
    ax.set_xlabel("Temperatura inicial T0")
    ax.set_ylabel("Brecha media respecto al óptimo (%)")
    ax.set_title("Interacción en escala relativa (brecha %)")
    ax.legend(title="Tamaño", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, "interaccion_brecha.png"), dpi=200)
    plt.close(fig)

    # Diagramas de caja por punto de diseño, con mediana bootstrap e IC
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.8))
    for a, n in zip(ax, cfg.TAMANOS):
        g = df[df.tamano == n]
        datos = [g.loc[g.temperatura == t, "costo"].values for t in temps]
        a.boxplot(datos, tick_labels=etiquetas_t, widths=0.5,
                  medianprops=dict(color=COLORES[n], lw=2),
                  flierprops=dict(marker="o", ms=4, mfc="none"))
        b = boot[boot.tamano == n]
        a.errorbar(np.arange(1, 4) + 0.33, b.mediana_bootstrap,
                   yerr=[np.clip(b.mediana_bootstrap - b.ic_inf, 0, None),
                         np.clip(b.ic_sup - b.mediana_bootstrap, 0, None)],
                   fmt="D", color="black", ms=4, capsize=3, label="Mediana bootstrap e IC 95%")
        a.set_title(f"n = {n}")
        a.set_xlabel("T0")
        # Si hay un valor atípico extremo, se recorta la escala y se anota su valor
        # (se considera extremo un costo más de 10 % arriba del mejor costo del bloque)
        es_extremo = g.costo > 1.10 * g.costo.min()
        extremos = g.loc[es_extremo]
        if len(extremos):
            normales = g.loc[~es_extremo, "costo"]
            lim = normales.max() + 0.25 * (normales.max() - normales.min())
            a.set_ylim(g.costo.min() - 2, lim)
            for _, fila in extremos.iterrows():
                x = temps.index(fila.temperatura) + 1
                a.annotate(f"atípico: {fila.costo:.2f}\n(fuera de escala)", xy=(x, lim),
                           xytext=(x - 1.0, lim - 0.10 * (lim - g.costo.min())), fontsize=8,
                           arrowprops=dict(arrowstyle="->", color="gray"))
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Costo de la ruta")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(DIR_FIG, "cajas_bootstrap.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
