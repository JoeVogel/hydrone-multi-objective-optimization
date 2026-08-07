import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from mpl_toolkits.mplot3d import art3d
from matplotlib.patches import Circle
from collections import Counter

# ------ Propeller plotting utils ------
airfoil_coords_base_path = Path("../../data/airfoil_coords")

e63_coords_path      = airfoil_coords_base_path / "e63-il.csv"
naca0012_coords_path = airfoil_coords_base_path / "naca4412-il.csv"

def load_airfoil_csv(path: Path):
    # pula a primeira linha (cabeçalho: X,Y ou algo do tipo)
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    # garante que é Nx2
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr

airfoil_coords_dict = {
    "NACA0012": load_airfoil_csv(naca0012_coords_path),
    "E63":      load_airfoil_csv(e63_coords_path),
}

def _parse_list_field(field, cast_func=None):
    """
    Converte automaticamente:
    - listas reais
    - tuplas
    - strings no formato "v1;v2;v3" (com ou sem colchetes)
    """
    if isinstance(field, list):
        return field
    if isinstance(field, tuple):
        return list(field)
    if isinstance(field, str):
        parts = field.replace("[", "").replace("]", "").split(";")
        parts = [p.strip() for p in parts if p.strip() != ""]
        if cast_func:
            return [cast_func(p) for p in parts]
        return parts
    raise ValueError(f"Campo não reconhecido como lista: {field!r}")

def plot_propeller_from_row(row, title_suffix="Propeller"):
    """
    Plota em 3D a geometria de um hélice a partir de uma linha de DataFrame.

    Campos esperados em `row`:
      - D: diâmetro do hélice
      - B: número de pás
      - hub_radius: raio do hub
      - number_of_sections: número de seções na pá
      - foil_list: lista (ou string) com o aerofólio de cada seção
      - chord_list: lista (ou string) com corda de cada seção (mesmas unidades de D)
      - pitch_list: lista (ou string) com o *twist* da seção
        (em graus OU em radianos – detectado automaticamente)

    Desenha apenas o SEGMENTO DE CORDA de cada seção, com cor
    dependente do aerofólio (E63 vermelho, NACA0012 azul).
    """

    # --- parâmetros básicos ---
    D = float(row["D"])
    B = int(row["B"])
    hub_radius = float(row["hub_radius"])
    n_sec = int(row["number_of_sections"])

    # --- converte campos em listas/arrays ---
    foil_list  = _parse_list_field(row["foil_list"],  cast_func=str)
    chord_list = np.asarray(_parse_list_field(row["chord_list"], cast_func=float))
    pitch_raw  = np.asarray(_parse_list_field(row["pitch_list"],  cast_func=float))

    if len(foil_list) != n_sec or len(chord_list) != n_sec or len(pitch_raw) != n_sec:
        raise ValueError(
            "foil_list, chord_list e pitch_list devem ter tamanho igual a number_of_sections"
        )

    R = D / 2.0
    r_phys = np.linspace(hub_radius, R, n_sec)
    r_nd   = r_phys / R  # r/R

    # --- detectar se pitch_raw está em graus ou em radianos ---
    max_abs = float(np.nanmax(np.abs(pitch_raw)))
    if max_abs > np.pi * 1.5:           # valores grandes → provavelmente graus
        twist = np.deg2rad(pitch_raw)
    else:
        twist = pitch_raw               # já está em rad

    # mapa de cores por aerofólio
    foil_colors = {
        "E63": "m",
        "NACA0012": "b",
    }

    def data_for_cylinder_along_z(cx, cy, radius, z_min, z_max=0.0):
        z = np.linspace(z_min, z_max, 40)
        theta = np.linspace(0, 2 * np.pi, 40)
        tgrid, zgrid = np.meshgrid(theta, z)
        x = radius * np.cos(tgrid) + cx
        y = radius * np.sin(tgrid) + cy
        return x, y, zgrid

    # --- figura 3D ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # para cada pá
    for blade_index in range(B):
        psi = 2.0 * np.pi * blade_index / B

        # bases locais no disco
        e_r = np.array([np.cos(psi), np.sin(psi), 0.0])   # radial
        e_t = np.array([-np.sin(psi), np.cos(psi), 0.0])  # tangencial (corda sem twist)
        e_n = np.cross(e_r, e_t)                          # normal da pá
        e_n /= np.linalg.norm(e_n) + 1e-15

        for i in range(n_sec):
            beta = twist[i]  # twist/aoa geométrico da seção (rad)

            # direção da corda após twist (no plano tangencial–normal)
            c_dir = np.cos(beta) * e_t + np.sin(beta) * e_n

            # ponto central da seção no raio r/R
            r0 = r_nd[i] * e_r

            # corda normalizada
            c_nd = chord_list[i] / R

            # extremos da corda (linha de seção)
            p1 = r0 - 0.5 * c_nd * c_dir
            p2 = r0 + 0.5 * c_nd * c_dir

            x = [p1[0], p2[0]]
            y = [p1[1], p2[1]]
            z = [p1[2], p2[2]]

            foil_name = foil_list[i]
            color = foil_colors.get(foil_name, "k")  # default preto

            ax.plot3D(x, y, z, color=color, linewidth=1.2)

    # --- hub ---
    hub_nd = hub_radius / R
    Xc, Yc, Zc = data_for_cylinder_along_z(0.0, 0.0, hub_nd, -0.05, 0.0)
    ax.plot_surface(Xc, Yc, Zc, color="r", antialiased=False, alpha=0.8)

    p = Circle((0, 0), hub_nd, color="r")
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0.0)

    # --- ajustes de visual ---
    ax.set_title(f"Blade Planform ({title_suffix})", fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-0.4, 0.4)

    ax.set_xlabel("X/R")
    ax.set_ylabel("Y/R")
    ax.set_zlabel("Z/R")
    ax.grid(True)
    ax.azim = -50
    ax.elev = 35
    ax.set_box_aspect((1, 1, 0.5))

    plt.tight_layout()
    plt.show()

def plot_propeller_from_row_plotly(row, title_suffix="Propeller"):
    """
    Versão interativa em Plotly: plota o hélice em 3D usando segmentos de corda
    coloridos por aerofólio.

    Campos esperados em `row`:
      - D: diâmetro do hélice
      - B: número de pás
      - hub_radius: raio do hub
      - number_of_sections: número de seções na pá
      - foil_list: lista (ou string) com o aerofólio de cada seção
      - chord_list: lista (ou string) com corda de cada seção (mesmas unidades de D)
      - pitch_list: lista (ou string) com o *twist* da seção
        (em graus OU em radianos – detectado automaticamente)

    Retorna um plotly.graph_objects.Figure.
    """

    # --- parâmetros básicos ---
    D = float(row["D"])
    B = int(row["B"])
    hub_radius = float(row["hub_radius"])
    n_sec = int(row["number_of_sections"])

    # --- converte campos em listas/arrays ---
    foil_list  = _parse_list_field(row["foil_list"],  cast_func=str)
    chord_list = np.asarray(_parse_list_field(row["chord_list"], cast_func=float))
    pitch_raw  = np.asarray(_parse_list_field(row["pitch_list"],  cast_func=float))

    if len(foil_list) != n_sec or len(chord_list) != n_sec or len(pitch_raw) != n_sec:
        raise ValueError(
            "foil_list, chord_list e pitch_list devem ter tamanho igual a number_of_sections"
        )

    R = D / 2.0
    r_phys = np.linspace(hub_radius, R, n_sec)
    r_nd   = r_phys / R  # r/R

    # --- detectar se pitch_raw está em graus ou em radianos ---
    max_abs = float(np.nanmax(np.abs(pitch_raw)))
    if max_abs > np.pi * 1.5:      # valores grandes → provavelmente graus
        twist = np.deg2rad(pitch_raw)
    else:
        twist = pitch_raw          # já está em rad

    # mapa de cores por aerofólio
    foil_colors = {
        "E63": "olive",
        "NACA4412": "blue",
    }

    # figura plotly
    fig = go.Figure()

    # --- pás ---
    for blade_index in range(B):
        psi = 2.0 * np.pi * blade_index / B

        # bases locais no disco
        e_r = np.array([np.cos(psi), np.sin(psi), 0.0])   # radial
        e_t = np.array([-np.sin(psi), np.cos(psi), 0.0])  # tangencial (corda sem twist)
        e_n = np.cross(e_r, e_t)                          # normal da pá
        e_n /= np.linalg.norm(e_n) + 1e-15

        for i in range(n_sec):
            beta = twist[i]  # twist/aoa geométrico (rad)

            # direção da corda após twist (no plano tangencial–normal)
            c_dir = np.cos(beta) * e_t + np.sin(beta) * e_n

            # ponto central da seção no raio r/R
            r0 = r_nd[i] * e_r

            # corda normalizada
            c_nd = chord_list[i] / R

            # extremos da corda (linha de seção)
            p1 = r0 - 0.5 * c_nd * c_dir
            p2 = r0 + 0.5 * c_nd * c_dir

            foil_name = foil_list[i]
            color = foil_colors.get(foil_name, "black")

            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False  # pra não explodir a legenda
            ))

    # --- hub como “cilindro” simples ---
    hub_nd = hub_radius / R
    z_cyl = np.linspace(-0.05, 0.0, 20)
    theta = np.linspace(0, 2*np.pi, 40)
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    x_cyl = hub_nd * np.cos(theta_grid)
    y_cyl = hub_nd * np.sin(theta_grid)

    fig.add_surface(
        x=x_cyl,
        y=y_cyl,
        z=z_grid,
        showscale=False,
        colorscale=[[0, "rgba(200,0,0,0.8)"], [1, "rgba(200,0,0,0.8)"]],
        opacity=0.9,
    )

    # layout
    fig.update_layout(
        title=f"Blade Planform ({title_suffix})",
        scene=dict(
            xaxis_title="X/R",
            yaxis_title="Y/R",
            zaxis_title="Z/R",
            xaxis=dict(range=[-1.1, 1.1]),
            yaxis=dict(range=[-1.1, 1.1]),
            zaxis=dict(range=[-0.4, 0.4]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
    )

    return fig

# --------------------------------------------

# --- Optimization run analysis edits ---

def plot_generational_metrics_interactive(df_evaluations: pd.DataFrame):
    """
    Gera um gráfico interativo por geração com:
      - média e melhor (maior) aerial_fitness
      - média e melhor (maior) aquatic_fitness
      - média e melhor (menor) cavitating_proportion
      - média e melhor (maior) QI
      - médias e melhores (menores) penalidades aéreas

    Retorna um plotly.graph_objects.Figure.
    """

    required_cols = [
        "generation",
        "B",
        "aerial_fitness",
        "aquatic_fitness",
        "cavitating_proportion",
        "QI",
        "FM",
        "aerial_blade_count_penalty",
        "aerial_solidity_penalty",
        "aquatic_solidity_penalty",
    ]
    for col in required_cols:
        if col not in df_evaluations.columns:
            raise ValueError(f"Coluna obrigatória ausente em df_evaluations: '{col}'")

    grouped = (
        df_evaluations
        .groupby("generation")
        .agg(
            b_mean=("B", "mean"),

            # Fitness e métricas (best = maior)
            aerial_mean=("aerial_fitness", "mean"),
            aerial_best=("aerial_fitness", "max"),

            aquatic_mean=("aquatic_fitness", "mean"),
            aquatic_best=("aquatic_fitness", "max"),

            qi_mean=("QI", "mean"),
            qi_best=("QI", "max"),

            fm_mean=("FM", "mean"),
            fm_best=("FM", "max"),

            # Penalidades (best = menor)
            blade_count_mean=("aerial_blade_count_penalty", "mean"),
            blade_count_best=("aerial_blade_count_penalty", "min"),

            cav_mean=("cavitating_proportion", "mean"),
            cav_best=("cavitating_proportion", "min"), 

            aerial_solidity_penalty_mean=("aerial_solidity_penalty", "mean"),
            aerial_solidity_penalty_best=("aerial_solidity_penalty", "min"),

            aquatic_solidity_penalty_mean=("aquatic_solidity_penalty", "mean"),
            aquatic_solidity_penalty_best=("aquatic_solidity_penalty", "min"),
        )
        .reset_index()
        .sort_values("generation")
    )

    x = grouped["generation"]

    # Cores fixas por métrica
    colors = {
        "total_b": "#e377c2",           # rosa
        "aerial": "#1f77b4",            # azul
        "aquatic": "#2ca02c",           # verde
        "qi": "#9467bd",                # roxo
        "fm": "#ff7f0e",                # laranja
        "cav": "#d62728",               # vermelho
        "aerial_solidity": "#8c564b",   # marrom
        "aquatic_solidity": "#7f7f7f",  # cinza
        "blade_count": "#17becf",       # ciano
    }

    fig = go.Figure()

    # Número médio de pás
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["b_mean"],
        mode="lines",
        name="Número médio de pás",
        line=dict(color=colors["total_b"], dash="solid")
    ))

    # Aerial fitness (best = maior)
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["aerial_mean"],
        mode="lines",
        name="Aerial fitness (média)",
        line=dict(color=colors["aerial"], dash="solid")
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["aerial_best"],
        mode="lines",
        name="Aerial fitness (melhor)",
        line=dict(color=colors["aerial"], dash="dash")
    ))

    # Aquatic fitness (best = maior)
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["aquatic_mean"],
        mode="lines",
        name="Aquatic fitness (média)",
        line=dict(color=colors["aquatic"], dash="solid")
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["aquatic_best"],
        mode="lines",
        name="Aquatic fitness (melhor)",
        line=dict(color=colors["aquatic"], dash="dash")
    ))

    # QI (best = maior)
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["qi_mean"],
        mode="lines",
        name="QI (média)",
        line=dict(color=colors["qi"], dash="solid")
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["qi_best"],
        mode="lines",
        name="QI (melhor)",
        line=dict(color=colors["qi"], dash="dash")
    ))

    # FM (best = maior)
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["fm_mean"],
        mode="lines",
        name="FM (média)",
        line=dict(color=colors["fm"], dash="solid")
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["fm_best"],
        mode="lines",
        name="FM (melhor)",
        line=dict(color=colors["fm"], dash="dash")
    ))

    # Penalidades (best = menor)

    # Blade count penalty
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["blade_count_mean"],
        mode="lines",
        name="Blade count penalty (média)",
        line=dict(color=colors["blade_count"], dash="solid")
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["blade_count_best"],
        mode="lines",
        name="Blade count penalty (melhor/menor)",
        line=dict(color=colors["blade_count"], dash="dash")
    ))

    # Cavitation proportion
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["cav_mean"],
        mode="lines",
        name="Cavitating proportion (média)",
        line=dict(color=colors["cav"], dash="solid")
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["cav_best"],
        mode="lines",
        name="Cavitating proportion (melhor/menor)",
        line=dict(color=colors["cav"], dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["aerial_solidity_penalty_mean"],
        mode="lines",
        name="Aerial solidity penalty (média)",
        line=dict(color=colors["aerial_solidity"], dash="solid")
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["aerial_solidity_penalty_best"],
        mode="lines",
        name="Aerial solidity penalty (melhor/menor)",
        line=dict(color=colors["aerial_solidity"], dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["aquatic_solidity_penalty_mean"],
        mode="lines",
        name="Aquatic solidity penalty (média)",
        line=dict(color=colors["aquatic_solidity"], dash="solid")
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=grouped["aquatic_solidity_penalty_best"],
        mode="lines",
        name="Aquatic solidity penalty (melhor/menor)",
        line=dict(color=colors["aquatic_solidity"], dash="dash")
    ))

    fig.update_layout(
        title="Evolução por geração das métricas de fitness, cavitação e penalidades",
        xaxis_title="Geração",
        yaxis_title="Valor da métrica",
        hovermode="x unified",
    )

    return fig

def plot_pareto_front(df_pareto: pd.DataFrame,
                      title="Pareto Front - Aquatic vs Aerial Fitness"):
    """
    Plota a frente de Pareto usando Plotly, comparando:
      - eixo X: aquatic_fitness
      - eixo Y: aerial_fitness

    Se houver coluna 'generation', usa como cor para facilitar a visualização;
    caso contrário, plota tudo na mesma cor.

    Retorna um plotly.graph_objects.Figure.
    """

    required = ["aquatic_fitness", "aerial_fitness"]
    for col in required:
        if col not in df_pareto.columns:
            raise ValueError(f"Coluna obrigatória ausente em df_pareto: '{col}'")

    color_col = "generation" if "generation" in df_pareto.columns else None

    fig = px.scatter(
        df_pareto,
        x="aquatic_fitness",
        y="aerial_fitness",
        color=color_col,
        title=title,
        labels={
            "aquatic_fitness": "Aquatic fitness",
            "aerial_fitness": "Aerial fitness",
            "generation": "Generation",
        },
        hover_data={"aquatic_fitness": True,
                    "aerial_fitness": True}
    )

    fig.update_layout(
        xaxis_title="Aquatic fitness",
        yaxis_title="Aerial fitness",
        legend_title="Generation" if color_col else None,
    )

    return fig

def plot_pareto_front_matplot(df_pareto: pd.DataFrame):
    # Plot Pareto fronts
    plt.figure(figsize=(9, 6))

    x = df_pareto["aquatic_fitness"].to_numpy()
    y = df_pareto["aerial_fitness"].to_numpy()

    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]

    plt.scatter(x, y, s=20, marker="o",
                facecolors="blue", edgecolors="black", linewidths=0.6,
                alpha=0.9, label="Front")

    plt.title("Pareto Front — NSGA-II", pad=10)
    plt.ylabel("Aerial fitness")
    plt.xlabel("Aquatic fitness")
    plt.grid(True, which="both", ls=":", lw=0.8, alpha=0.6)

    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.show()

def plot_pareto_front_with_solutions(df_pareto: pd.DataFrame, intermediate_representative_idx):

    x = df_pareto['aquatic_fitness'].to_numpy()
    y = df_pareto['aerial_fitness'].to_numpy()

    best_aerial_pos = np.argmax(y)
    best_aquatic_pos = np.argmax(x)

    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]

    plt.figure(figsize=(9, 6))

    # Pareto points
    plt.scatter(x, y, s=20, marker="o",
                facecolors="blue", edgecolors="black", linewidths=0.6,
                alpha=0.9, label="Front")

    # Highlight solutions
    plt.scatter(x[best_aerial_pos], y[best_aerial_pos],
                marker='s', s=120, color="orange",
                edgecolors="black", linewidths=0.8,
                label='Best aerial')

    plt.scatter(x[best_aquatic_pos], y[best_aquatic_pos],
                marker='^', s=120, color="green",
                edgecolors="black", linewidths=0.8,
                label='Best aquatic')

    plt.scatter(x[intermediate_representative_idx], y[intermediate_representative_idx],
                marker='*', s=220, color="red",
                edgecolors="black", linewidths=0.8,
                label='Intermediate')

    plt.title("Pareto Front with Representative Propeller Designs", pad=10)
    plt.ylabel("Aerial fitness")
    plt.xlabel("Aquatic fitness")

    plt.grid(True, which="both", ls=":", lw=0.8, alpha=0.6)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    plt.show()

def plot_hypervolume_evolution(generations, hv_values):
    plt.figure(figsize=(10, 5))
    plt.scatter(generations, hv_values, marker='o', color='b', s=10, edgecolors='black')
    plt.title('Evolution of Hypervolume Indicator')
    plt.xlabel('Generation')
    plt.ylabel('Hypervolume (HV)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# --------------------------------------------

# --- Multiple run analysis edits ---

def plot_aerial_boxplot(df_runs: pd.DataFrame):
    """
    Plots a boxplot of the aerial_fitness across multiple runs.
    Expects df_runs to have columns: 'run_id', 'aerial_fitness'.
    """

    if "run_id" not in df_runs.columns or "aerial_fitness" not in df_runs.columns:
        raise ValueError("DataFrame must contain 'run_id' and 'aerial_fitness' columns.")

    fig = px.box(
        df_runs,
        y="aerial_fitness",
        points="all",
        title="Aerial Fitness across Multiple Runs",
        labels={"aerial_fitness": "Aerial Fitness"}
    )

    fig.update_layout(
        yaxis_title="Aerial Fitness",
    )

    return fig

def plot_aquatic_boxplot(df_runs: pd.DataFrame):
    """
    Plots a boxplot of the aquatic_fitness across multiple runs.
    Expects df_runs to have columns: 'run_id', 'aquatic_fitness'.
    """

    if "run_id" not in df_runs.columns or "aquatic_fitness" not in df_runs.columns:
        raise ValueError("DataFrame must contain 'run_id' and 'aquatic_fitness' columns.")

    fig = px.box(
        df_runs,
        y="aquatic_fitness",
        points="all",
        title="Aquatic Fitness across Multiple Runs",
        labels={"aquatic_fitness": "Aquatic Fitness"}
    )

    fig.update_layout(
        yaxis_title="Aquatic Fitness",
    )

    return fig

def plot_multiple_boxplots(df_runs: pd.DataFrame):
    """
    Plots boxplots for both aerial and aquatic fitness across multiple runs.
    Expects df_runs to have columns: 'run_id', 'aerial_fitness', 'aquatic_fitness'.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.boxplot(
        data=df_runs,
        x="run_id",
        y="aerial_fitness",
        ax=axes[0],
        palette="viridis"
    )
    axes[0].set_title("Aerial Fitness")
    axes[0].grid(True, ls="--", alpha=0.4)

    sns.boxplot(
        data=df_runs,
        x="run_id",
        y="aquatic_fitness",
        ax=axes[1],
        palette="magma"
    )
    axes[1].set_title("Aquatic Fitness")
    axes[1].grid(True, ls="--", alpha=0.4)

    plt.suptitle("Multiple Runs Fitness Comparison")
    plt.show()

def plot_combined_boxplots(df_runs: pd.DataFrame):
    """
    Plots combined boxplots for aerial and aquatic fitness across multiple runs.
    Expects df_runs to have columns: 'run_id', 'aerial_fitness', 'aquatic_fitness'.
    """

    df_melted = df_runs.melt(
        id_vars=["run_id"],
        value_vars=["aerial_fitness", "aquatic_fitness"],
        var_name="fitness_type",
        value_name="fitness_value"
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_melted,
        x="run_id",
        y="fitness_value",
        hue="fitness_type",
        palette="Set2"
    )
    plt.title("Combined Aerial and Aquatic Fitness across Multiple Runs")
    plt.xlabel("Run ID")
    plt.ylabel("Fitness Value")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(title="Fitness Type")
    plt.show()

# --------------------------------------------

# --- Plot chord e pitch---

def _parse_array(s):
    return np.array([float(v) for v in str(s).split(";")])

def _radius_distribution(row):
    n = int(row["number_of_sections"])
    hub_radius = float(row["hub_radius"])
    R = float(row["D"]) / 2000.0  # mm -> m -> radius

    r = np.linspace(hub_radius, R, n)
    return r / R

def plot_representative_chord(df):

    labels = ["Best aerial", "Intermediate", "Best aquatic"]

    plt.figure(figsize=(9,6))

    for i, (_, row) in enumerate(df.iterrows()):

        chord = _parse_array(row["chord_list"])
        r_over_R = _radius_distribution(row)

        label = f"{labels[i]} (B={int(row['B'])})"

        plt.plot(r_over_R, chord,
                 marker="o",
                 linewidth=1.8,
                 label=label)

    plt.title("Chord distribution of representative propeller designs", pad=10)
    plt.xlabel("Normalized radius (r/R)")
    plt.ylabel("Chord [m]")

    plt.grid(True, which="both", ls=":", lw=0.8, alpha=0.6)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    plt.show()

def plot_representative_pitch(df):

    labels = ["Best aerial", "Intermediate", "Best aquatic"]

    plt.figure(figsize=(9,6))

    for i, (_, row) in enumerate(df.iterrows()):

        pitch = _parse_array(row["pitch_list"])
        r_over_R = _radius_distribution(row)

        label = f"{labels[i]} (B={int(row['B'])})"

        plt.plot(r_over_R, pitch,
                 marker="o",
                 linewidth=1.8,
                 label=label)

    plt.title("Pitch distribution of representative propeller designs", pad=10)
    plt.xlabel("Normalized radius (r/R)")
    plt.ylabel("Pitch angle [deg]")

    plt.grid(True, which="both", ls=":", lw=0.8, alpha=0.6)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    plt.show()

def build_representative_summary(df):

    labels = ["Best aerial", "Intermediate", "Best aquatic"]
    rows = []

    for i, (_, row) in enumerate(df.iterrows()):

        chord = _parse_array(row["chord_list"])
        pitch = _parse_array(row["pitch_list"])
        foils = row["foil_list"].split(";")

        foil_count = Counter(foils)
        foil_summary = ", ".join([f"{k} ({v})" for k, v in foil_count.items()])

        rows.append({
            "Design": labels[i],
            "Blades": int(row["B"]),
            "Aerial fitness": row["aerial_fitness"],
            "Aquatic fitness": row["aquatic_fitness"],
            "Mean chord [m]": chord.mean(),
            "Max chord [m]": chord.max(),
            "Mean pitch [deg]": pitch.mean(),
            "Airfoils": foil_summary
        })

    return pd.DataFrame(rows)

# --------------------------------------------

