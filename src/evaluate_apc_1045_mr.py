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

        T, Q, P, J, CT, CQ, CP, eta, cavitating_proportion, QI = water_solver.evaluate(rotor)     
        print("Cavitating Proportion", cavitating_proportion)
        print("Quality Index (QI): ", QI)

        results.append((scenario.rpm, T, Q, P, J, CT, CQ, CP, eta, cavitating_proportion))
    
    return results

def plot_aerial_results(results):
    # Converte para arrays
    rpm       = np.array([r[0] for r in results], dtype=float)
    T_single  = np.array([r[1] for r in results], dtype=float)
    Q_single  = np.array([r[2] for r in results], dtype=float)  # N·m
    P_single  = np.array([r[3] for r in results], dtype=float)  # W

    # dobra para 2 hélices em paralelo
    T = 2.0 * T_single
    Q = 2.0 * Q_single
    P = 2.0 * P_single

    rpm_fit = np.linspace(0, 6000, 300)

    # Horn,2019 data
    df_horn_T = pd.read_csv(base_dir / "analysis" / "horn2019" / "propeller_10_data_T.csv")
    df_horn_P = pd.read_csv(base_dir / "analysis" / "horn2019" / "propeller_10_data_P.csv")

    horn_T = np.poly1d(np.polyfit(df_horn_T['rpm'].to_numpy(), df_horn_T['T'].to_numpy(), 3))(rpm_fit)
    horn_P = np.poly1d(np.polyfit(df_horn_P['rpm'].to_numpy(), df_horn_P['P'].to_numpy(), 3))(rpm_fit)

    # Curvas suavizadas (polinômio grau 3)
    T_fit = np.poly1d(np.polyfit(rpm, T, 3))(rpm_fit)
    Q_fit = np.poly1d(np.polyfit(rpm, Q, 3))(rpm_fit)
    P_fit = np.poly1d(np.polyfit(rpm, P, 3))(rpm_fit)

    # --- diferenças ponto a ponto (BEMT - Horn) ---
    mask = (rpm_fit >= 1000) & (rpm_fit <= 6000)

    T_ref, T_hat = horn_T[mask], T_fit[mask]
    P_ref, P_hat = horn_P[mask], P_fit[mask]

    eT = T_hat - T_ref
    eP = P_hat - P_ref

    MAE_T  = np.mean(np.abs(eT))
    RMSE_T = np.sqrt(np.mean(eT**2))
    MAE_P  = np.mean(np.abs(eP))
    RMSE_P = np.sqrt(np.mean(eP**2))

    # Normalização opcional
    NRMSE_T = RMSE_T / (T_ref.max() - T_ref.min())
    NRMSE_P = RMSE_P / (P_ref.max() - P_ref.min())

    # Texto compacto para cada subplot
    txt_T = f"MAE = {MAE_T:.2f} N\nRMSE = {RMSE_T:.2f} N\nNRMSE = {NRMSE_T*100:.1f}%"
    txt_P = f"MAE = {MAE_P:.2f} W\nRMSE = {RMSE_P:.2f} W\nNRMSE = {NRMSE_P*100:.1f}%"

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
    ax1.plot(rpm_fit, horn_T, color=reference_color, linestyle="--", linewidth=2.0, label="Horn, 2019")
    ax1.set_xlabel(r"$\omega$ [rpm]")
    ax1.set_xlim(1000, 6000)
    ax1.set_ylabel(r"$T$ [N]")
    ax1.set_ylim(0, 15)
    ax1.xaxis.set_major_formatter(kfmt)

    ax1.text(
        0.02, 0.98, txt_T,
        transform=ax1.transAxes, ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85)
    )
    
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(frameon=True, fontsize=9)

    # --- Subplot Q ---
    ax2.plot(rpm_fit, P_fit, color=line_color, linewidth=2.0, label="BEMT - adjusted curve")
    ax2.plot(rpm, P, color=line_color, **marker_style, label="BEMT - experiments")
    ax2.plot(rpm_fit, horn_P, color=reference_color, linestyle="--", linewidth=2.0, label="Horn, 2019")
    ax2.set_xlabel(r"$\omega$ [rpm]")
    ax2.set_xlim(1000, 6000)
    ax2.set_ylabel(r"$P$ [W]")
    ax2.xaxis.set_major_formatter(kfmt)

    ax2.text(
        0.02, 0.98, txt_P,
        transform=ax2.transAxes, ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85)
    )

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

    # Geometria principal (do arquivo do 10x4.5 MR)
    diameter     = 0.254      # 10 in
    radius_hub   = 0.02032    # 0.8 in  (HUBTRA)
    n_blades     = 2

    # Estações radiais (m) — do arquivo (0.80" ... 5.00")
    stations_m = [
        0.020320, 0.021844, 0.023368, 0.024892, 0.026416, 0.027940, 0.029464, 0.031118,
        0.034011, 0.037033, 0.040056, 0.043078, 0.046100, 0.049122, 0.052144, 0.055166,
        0.058188, 0.061210, 0.064233, 0.067255, 0.070277, 0.073299, 0.076321, 0.079343,
        0.082365, 0.085387, 0.088409, 0.091431, 0.094454, 0.097476, 0.100498, 0.103520,
        0.106542, 0.109564, 0.112586, 0.115608, 0.118630, 0.121653, 0.124675, 0.127000
    ]

    # Corda (m) — do arquivo
    chord_list = [
        0.018674, 0.019776, 0.020813, 0.021781, 0.022682, 0.023515, 0.02428, 0.025037, 
        0.02617, 0.027099, 0.02777, 0.028184, 0.028344, 0.02825, 0.028011, 0.027711, 
        0.027356, 0.026944, 0.026479, 0.025964, 0.025397, 0.02478, 0.02412, 0.023414, 
        0.022664, 0.021877, 0.021049, 0.020183, 0.019284, 0.018349, 0.017386, 0.016391, 
        0.01537, 0.014323, 0.013251, 0.012156, 0.011044, 0.009728, 0.007127, 0.000003
    ]

    # Pitch PRATHER (polegadas) -> ângulo local (graus) por seção
    pitch_pr_in = [
        3.1671, 3.5392, 3.8486, 4.0903, 4.2613, 4.3597, 4.3918, 4.4057,
        4.4226, 4.4329, 4.4370, 4.4374, 4.4365, 4.4352, 4.4332, 4.4309,
        4.4283, 4.4254, 4.4224, 4.4192, 4.4156, 4.4116, 4.4071, 4.4019,
        4.3960, 4.3891, 4.3811, 4.3719, 4.3612, 4.3488, 4.3347, 4.3185,
        4.3003, 4.2797, 4.2567, 4.2310, 4.2022, 4.1701, 4.1348, 4.5147
    ]

    inch = 0.0254
    pitch_list = [
        math.degrees(math.atan2(p*inch, 2*math.pi*r))
        for p, r in zip(pitch_pr_in, stations_m)
    ]

    # Aerofólios por seção (E63 até 2.85"; NACA4412 a partir de 4.65"; zona de transição tratada como NACA4412)
    r_e63_end    = 2.85 * inch
    r_apc12_beg  = 4.65 * inch
    foil_list = [
        'E63' if r <= r_e63_end else ('NACA4412' if r >= r_apc12_beg else 'NACA4412')
        for r in stations_m
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
        Scenario(rpm=1000.0, v_inf=0.0),
        Scenario(rpm=1500.0, v_inf=0.0),
        Scenario(rpm=2000.0, v_inf=0.0),
        Scenario(rpm=2500.0, v_inf=0.0),
        Scenario(rpm=3000.0, v_inf=0.0),
        Scenario(rpm=3500.0, v_inf=0.0),
        Scenario(rpm=4000.0, v_inf=0.0),
        Scenario(rpm=4500.0, v_inf=0.0),
        Scenario(rpm=5000.0, v_inf=0.0),
        Scenario(rpm=5500.0, v_inf=0.0)
    ]
    
    # aerial_results = aerial_bemt_evaluations(rotor, list_of_scenarios)

    # plot_aerial_results(aerial_results)

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
        Scenario(rpm=650.0, v_inf=0.0),
        Scenario(rpm=700.0, v_inf=0.0)
    ]

    water_results = aquatic_bemt_evaluations(rotor, list_of_scenarios_water)

    plot_aquatic_results(water_results)