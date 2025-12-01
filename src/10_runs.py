# runner.py
import sys
import json
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = BASE_DIR / "main.py"


def run_nsga_instance(run_id, elitism_fraction=None, mutation_rate=None, pop_size=None, generations=None, seed=None):
    """
    Executa main.py como um processo separado, passando parâmetros via CLI.
    Retorna um dicionário com id da execução e o pareto lido do stdout.
    """

    time.sleep(run_id * 2)

    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        f"--elitism_fraction={elitism_fraction}",
        f"--mutation_rate={mutation_rate}",
        f"--pop_size={pop_size}",
        f"--generations={generations}",
        f"--seed={seed}"  
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        # main.py imprime APENAS o JSON da frente de Pareto
        pareto_front = json.loads(result.stdout)

        return {
            "run_id": run_id,
            "success": True,
            "pareto_front": pareto_front,
        }

    except subprocess.CalledProcessError as e:
        # Se der erro, retornamos info útil
        return {
            "run_id": run_id,
            "success": False,
            "error": f"Process failed with code {e.returncode}",
            "stderr": e.stderr,
        }
    except json.JSONDecodeError as e:
        return {
            "run_id": run_id,
            "success": False,
            "error": f"Failed to parse JSON from stdout: {e}",
            "raw_stdout": result.stdout if "result" in locals() else "",
        }


def main():
    runs_config = []
    for i in range(10):
        runs_config.append({
            "run_id": i,
            "elitism_fraction": 0.5,
            "mutation_rate": 0.2,
            "pop_size": 120,
            "generations": 80,
            "seed": i * 100
        })

    results = []

    workers = 5
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for cfg in runs_config:
            futures.append(
                executor.submit(
                    run_nsga_instance,
                    cfg["run_id"],
                    cfg["elitism_fraction"],
                    cfg["mutation_rate"],
                    cfg["pop_size"],
                    cfg["generations"],
                    cfg["seed"]
                )
            )

        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            if res["success"]:
                print(f"[RUN {res['run_id']}] finished with {len(res['pareto_front'])} points")
            else:
                print(f"[RUN {res['run_id']}] FAILED: {res['error']}")

    # Se quiser salvar tudo em um JSON único:
    with open("all_runs_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
