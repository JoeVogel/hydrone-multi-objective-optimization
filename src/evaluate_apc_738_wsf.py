import logging
import math

# Configurar o log para imprimir no console
logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Formato da saída
)

logger = logging.getLogger(__name__)

from scenario                   import Scenario
from rotor                      import Rotor
from evaluation.aerial_methods  import AerialBEMT
from evaluation.aquatic_methods import WaterBEMT

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from matplotlib.ticker import FuncFormatter

base_dir = Path(__file__).resolve().parent.parent 

def aerial_bemt_evaluations(rotor, list_of_scenarios):
    print("Evaluation Type: ", "Aerial BEMT")

    results = []

    for scenario in list_of_scenarios:
        air_solver = AerialBEMT(
            scenario=scenario
        )

        print("Scenario RPM: ", scenario.rpm)

        T, Q, P, J, CT, CQ, CP, eta = air_solver.evaluate(rotor)

        print("Thrust: ", T)
        print("Power:  ", P)

        results.append((scenario.rpm, T, Q, P, J, CT, CQ, CP, eta))
    
    return results

def aquatic_bemt_evaluations(rotor, list_of_scenarios):
    print("Evaluation Type: ", "Water BEMT")

    results = []

    for scenario in list_of_scenarios:
        water_solver = WaterBEMT(
            scenario=scenario
        )

        print("Scenario RPM: ", scenario.rpm)

        T, Q, P, J, CT, CQ, CP, eta, cavitating_proportion = water_solver.evaluate(rotor)     
        print("Cavitating Proportion", cavitating_proportion)

        results.append((scenario.rpm, T, Q, P, J, CT, CQ, CP, eta, cavitating_proportion))
    
    return results

def plot_aerial_results(results):
    # Converte para arrays
    rpm       = np.array([r[0] for r in results], dtype=float)
    T_single  = np.array([r[1] for r in results], dtype=float)
    Q_single  = np.array([r[2] for r in results], dtype=float)  # N·m
    P_single  = np.array([r[3] for r in results], dtype=float)  # W

    # Dobra para 2 hélices em paralelo
    T = 2.0 * T_single
    Q = 2.0 * Q_single
    P = 2.0 * P_single

    # Range comum para desenhar curvas suavizadas
    rpm_fit = np.linspace(rpm.min(), rpm.max(), 300)

    # --- Dados Horn 2019
    df_horn_T = pd.read_csv(base_dir / "analysis" / "horn2019" / "propeller_7_data_T_aerial.csv")
    df_horn_P = pd.read_csv(base_dir / "analysis" / "horn2019" / "propeller_7_data_P_aerial.csv")

    deg_T = min(3, max(1, len(df_horn_T) - 1))
    poly_horn_T = np.poly1d(np.polyfit(df_horn_T['rpm'].to_numpy(), df_horn_T['T'].to_numpy(), deg_T))
    hT_min, hT_max = df_horn_T['rpm'].min(), df_horn_T['rpm'].max()
    horn_T = poly_horn_T(rpm_fit)

    deg_P = min(3, max(1, len(df_horn_P) - 1))
    poly_horn_P = np.poly1d(np.polyfit(df_horn_P['rpm'].to_numpy(), df_horn_P['P'].to_numpy(), deg_P))
    hP_min, hP_max = df_horn_P['rpm'].min(), df_horn_P['rpm'].max()
    horn_P = poly_horn_P(rpm_fit)

    # --- Curvas suavizadas do BEMT
    poly_T = np.poly1d(np.polyfit(rpm, T, 3))
    poly_Q = np.poly1d(np.polyfit(rpm, Q, 3))
    poly_P = np.poly1d(np.polyfit(rpm, P, 3))

    T_fit = poly_T(rpm_fit)
    Q_fit = poly_Q(rpm_fit)
    P_fit = poly_P(rpm_fit)

    # Plot
    kfmt = FuncFormatter(lambda x, pos: f"{int(x):,}")

    line_color = "#c00000"      # suas curvas
    reference_color = "#2B9E44" # Horn
    marker_style = dict(marker="o", linestyle="None", markersize=6,
                        markerfacecolor="none", markeredgewidth=1.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)

    # --- Subplot T ---
    ax1.plot(rpm_fit, T_fit, color=line_color, linewidth=2.0, label="BEMT")
    ax1.plot(rpm, T, color=line_color, **marker_style, label="BEMT (pts)")
    ax1.plot(rpm_fit, horn_T, color=reference_color, linestyle="--", linewidth=2.0, label="Horn 2019")
    ax1.plot(df_horn_T["rpm"], df_horn_T["T"], color=reference_color, marker="o", linestyle="None", label="Horn 2019 (pts)")
    ax1.set_xlabel(r"$\omega$ [rpm]")
    ax1.set_xlim(0, 9000)
    ax1.set_ylabel(r"$T$ [N]")
    ax1.xaxis.set_major_formatter(kfmt)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(frameon=True, fontsize=9)

    # --- Subplot P ---
    ax2.plot(rpm_fit, P_fit, color=line_color, linewidth=2.0, label="BEMT")
    ax2.plot(rpm, P, color=line_color, **marker_style, label="BEMT (pts)")
    ax2.plot(rpm_fit, horn_P, color=reference_color, linestyle="--", linewidth=2.0, label="Horn 2019")
    ax2.plot(df_horn_P["rpm"], df_horn_P["P"], color=reference_color, marker="o", linestyle="None", label="Horn 2019 (pts)")
    ax2.set_xlabel(r"$\omega$ [rpm]")
    ax2.set_xlim(0, 9000)
    ax2.set_ylabel(r"$P$ [W]")
    ax2.xaxis.set_major_formatter(kfmt)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_aquatic_results(results):
    # Converte para arrays
    rpm       = np.array([r[0] for r in results], dtype=float)
    T_single  = np.array([r[1] for r in results], dtype=float)
    Q_single  = np.array([r[2] for r in results], dtype=float)  # N·m
    P_single  = np.array([r[3] for r in results], dtype=float)  # W

    # dobra para 2 hélices em paralelo
    T = 2.0 * T_single
    Q = 2.0 * Q_single
    P = 2.0 * P_single

    rpm_fit = np.linspace(250, 750, 300)

    # Horn,2019 data
    # df_horn_T = pd.read_csv(base_dir / "analysis" / "horn2019" / "propeller_10_data_T.csv")
    # df_horn_P = pd.read_csv(base_dir / "analysis" / "horn2019" / "propeller_10_data_P.csv")

    # horn_T = np.poly1d(np.polyfit(df_horn_T['rpm'].to_numpy(), df_horn_T['T'].to_numpy(), 3))(rpm_fit)
    # horn_P = np.poly1d(np.polyfit(df_horn_P['rpm'].to_numpy(), df_horn_P['P'].to_numpy(), 3))(rpm_fit)

    # Curvas suavizadas (polinômio grau 3)
    T_fit = np.poly1d(np.polyfit(rpm, T, 3))(rpm_fit)
    Q_fit = np.poly1d(np.polyfit(rpm, Q, 3))(rpm_fit)
    P_fit = np.poly1d(np.polyfit(rpm, P, 3))(rpm_fit)

    # --- diferenças ponto a ponto (BEMT - Horn) ---
    # mask = (rpm_fit >= 1000) & (rpm_fit <= 6000)

    # T_ref, T_hat = horn_T[mask], T_fit[mask]
    # P_ref, P_hat = horn_P[mask], P_fit[mask]

    # eT = T_hat - T_ref
    # eP = P_hat - P_ref

    # MAE_T  = np.mean(np.abs(eT))
    # RMSE_T = np.sqrt(np.mean(eT**2))
    # MAE_P  = np.mean(np.abs(eP))
    # RMSE_P = np.sqrt(np.mean(eP**2))

    # # Normalização opcional
    # NRMSE_T = RMSE_T / (T_ref.max() - T_ref.min())
    # NRMSE_P = RMSE_P / (P_ref.max() - P_ref.min())

    # # Texto compacto para cada subplot
    # txt_T = f"MAE = {MAE_T:.2f} N\nRMSE = {RMSE_T:.2f} N\nNRMSE = {NRMSE_T*100:.1f}%"
    # txt_P = f"MAE = {MAE_P:.2f} W\nRMSE = {RMSE_P:.2f} W\nNRMSE = {NRMSE_P*100:.1f}%"

    # Formatador para eixo x com separador de milhar "1,000"
    kfmt = FuncFormatter(lambda x, pos: f"{int(x):,}")

    # Cores/estilos (ajuste como quiser)
    line_color = "#c00000"  # vermelho (T)
    reference_color = "#2B9E44"
    marker_style = dict(marker="o", linestyle="None", markersize=6,
                        markerfacecolor="none", markeredgewidth=1.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)

    # --- Subplot T ---
    ax1.plot(rpm_fit, T_fit, color=line_color, linewidth=2.0, label="BEMT - adjusted curve")
    ax1.plot(rpm, T, color=line_color, **marker_style, label="BEMT - experiments")
    # ax1.plot(rpm_fit, horn_T, color=reference_color, linestyle="--", linewidth=2.0, label="Horn, 2019")
    ax1.set_xlabel(r"$\omega$ [rpm]")
    ax1.set_xlim(250, 750)
    ax1.set_ylabel(r"$T$ [N]")
    ax1.set_ylim(0, 150)
    ax1.xaxis.set_major_formatter(kfmt)

    # ax1.text(
    #     0.02, 0.98, txt_T,
    #     transform=ax1.transAxes, ha="left", va="top",
    #     fontsize=9,
    #     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85)
    # )
    
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(frameon=True, fontsize=9)

    # --- Subplot Q ---
    ax2.plot(rpm_fit, P_fit, color=line_color, linewidth=2.0, label="BEMT - adjusted curve")
    ax2.plot(rpm, P, color=line_color, **marker_style, label="BEMT - experiments")
    # ax2.plot(rpm_fit, horn_P, color=reference_color, linestyle="--", linewidth=2.0, label="Horn, 2019")
    ax2.set_xlabel(r"$\omega$ [rpm]")
    ax2.set_xlim(250, 750)
    ax2.set_ylabel(r"$P$ [W]")
    ax2.xaxis.set_major_formatter(kfmt)

    # ax2.text(
    #     0.02, 0.98, txt_P,
    #     transform=ax2.transAxes, ha="left", va="top",
    #     fontsize=9,
    #     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85)
    # )

    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Geometria principal
    diameter     = 0.177800   # m  (≈ 7 in)
    radius_hub   = 0.016772   # m  (≈ 0.6603 in)
    n_blades     = 2

    # Estações radiais (m)
    stations_m = [
        0.016772, 0.017788, 0.018804, 0.019820, 0.020836, 0.021854, 0.022870, 0.024016,
        0.025900, 0.027912, 0.029926, 0.031938, 0.033952, 0.035966, 0.037978, 0.039992,
        0.042004, 0.044018, 0.046030, 0.048044, 0.050058, 0.052070, 0.054084, 0.056096,
        0.058110, 0.060124, 0.062136, 0.064150, 0.066162, 0.068176, 0.070190, 0.072202,
        0.074216, 0.076228, 0.078242, 0.080256, 0.082268, 0.084277, 0.085794, 0.086848,
        0.087653, 0.088367, 0.088900,
    ]

    # Corda (m)
    chord_list = [
        0.015588, 0.016187, 0.016769, 0.017333, 0.017879, 0.018407, 0.018920, 0.019477,
        0.020343, 0.021196, 0.021981, 0.022692, 0.023332, 0.023896, 0.024387, 0.024801,
        0.025136, 0.025395, 0.025575, 0.025677, 0.025697, 0.025634, 0.025491, 0.025263,
        0.024953, 0.024557, 0.024074, 0.023505, 0.022850, 0.022103, 0.021267, 0.020343,
        0.019324, 0.018217, 0.017013, 0.015718, 0.014308, 0.012377, 0.010300, 0.008263,
        0.005776, 0.002733, 0.000452,
    ]

    # Pitch PRATHER (polegadas) -> ângulo local (graus) por seção
    pitch_pr_in = [
        3.800, 3.800, 3.795, 3.790, 3.785, 3.780, 3.775, 3.770,
        3.760, 3.750, 3.740, 3.730, 3.720, 3.710, 3.700, 3.690,
        3.680, 3.670, 3.660, 3.650, 3.640, 3.630, 3.620, 3.610,
        3.600, 3.590, 3.580, 3.570, 3.560, 3.550, 3.540, 3.530,
        3.520, 3.510, 3.500, 3.490, 3.480, 3.470, 3.460, 3.450,
        3.440, 3.430, 3.420,
    ]

    inch = 0.0254
    pitch_list = [
        math.degrees(math.atan2(p*inch, 2*math.pi*r))
        for p, r in zip(pitch_pr_in, stations_m)
    ]

    # Aerofólio por seção (APC12 mapeado para NACA4412)
    foil_list = [
        'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63',
        'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63',
        'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63',
        'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'E63', 'NACA4412',
        'NACA4412', 'NACA4412', 'NACA4412',
    ]

    number_of_sections = len(stations_m)

    rotor = Rotor(
        n_blades=n_blades,
        diameter=diameter,
        radius_hub=radius_hub,
        number_of_sections=number_of_sections,
        foil_list=foil_list,
        chord_list=chord_list,
        pitch_list=pitch_list,
    )

    # ------- Evaluate in air ------------------------------

    list_of_scenarios = [
        Scenario(rpm=500.0, v_inf=0.0),
        Scenario(rpm=1000.0, v_inf=0.0),
        Scenario(rpm=1500.0, v_inf=0.0),
        Scenario(rpm=2000.0, v_inf=0.0),
        Scenario(rpm=2500.0, v_inf=0.0),
        Scenario(rpm=3000.0, v_inf=0.0),
        Scenario(rpm=3500.0, v_inf=0.0),
        Scenario(rpm=4000.0, v_inf=0.0),
        Scenario(rpm=4500.0, v_inf=0.0),
        Scenario(rpm=5000.0, v_inf=0.0),
        Scenario(rpm=5500.0, v_inf=0.0),
        Scenario(rpm=6000.0, v_inf=0.0),
        Scenario(rpm=6500.0, v_inf=0.0),
        Scenario(rpm=7000.0, v_inf=0.0),
        Scenario(rpm=7500.0, v_inf=0.0),
        Scenario(rpm=8000.0, v_inf=0.0),
        Scenario(rpm=8500.0, v_inf=0.0)
    ]
    
    aerial_results = aerial_bemt_evaluations(rotor, list_of_scenarios)

    plot_aerial_results(aerial_results)

    # ---------------------------------------------------

    print()

    # ------------ Evaluate in water --------------------
    
    list_of_scenarios_water = [
        Scenario(rpm=300.0, v_inf=0.0),
        Scenario(rpm=350.0, v_inf=0.0),
        Scenario(rpm=400.0, v_inf=0.0),
        Scenario(rpm=450.0, v_inf=0.0),
        Scenario(rpm=500.0, v_inf=0.0),
        Scenario(rpm=550.0, v_inf=0.0),
        Scenario(rpm=600.0, v_inf=0.0),
        Scenario(rpm=650.0, v_inf=0.0)
    ]

    # water_results = aquatic_bemt_evaluations(rotor, list_of_scenarios_water)

    # plot_aquatic_results(water_results)