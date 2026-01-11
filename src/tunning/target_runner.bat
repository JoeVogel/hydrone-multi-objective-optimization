@echo off
setlocal

REM ==========================================
REM 1) Descobrir caminhos base de forma robusta
REM ==========================================

REM %~dp0 = diretório onde está este .bat, ex:
REM D:\Mestrado\Dissertação\hydrone-multi-objective-optimization\src\tunning\
set "THIS_DIR=%~dp0"

REM Repo root = dois níveis acima de src\tunning -> ...\hydrone-multi-objective-optimization
set "ROOT_DIR=%THIS_DIR%..\.."

REM Diretório do ambiente virtual (.env)
set "VENV_DIR=%ROOT_DIR%\.env"

REM Diretório src
set "SRC_DIR=%ROOT_DIR%\src"

REM Normalizar caminhos (resolve ..\..)
for %%I in ("%ROOT_DIR%") do set "ROOT_DIR=%%~fI"
for %%I in ("%VENV_DIR%") do set "VENV_DIR=%%~fI"
for %%I in ("%SRC_DIR%") do set "SRC_DIR=%%~fI"

REM ==========================================
REM 2) Ativar o ambiente virtual (.env)
REM ==========================================

call "%VENV_DIR%\Scripts\activate.bat" 1>nul 2>nul

IF ERRORLEVEL 1 (
    REM Se der erro ao ativar o venv, devolve custo ruim e sai
    echo 1000000000
    endlocal
    goto :eof
)

REM ==========================================
REM 3) Parse dos parâmetros vindos do irace
REM ==========================================

set "ELITISM="
set "MUTATION="
set "POPSIZE="
set "GENERATIONS="

:parse
if "%~1"=="" goto afterparse

if "%~1"=="--elitism_fraction" (
    set "ELITISM=%~2"
    shift
) else if "%~1"=="--mutation_rate" (
    set "MUTATION=%~2"
    shift
) else if "%~1"=="--pop_size" (
    set "POPSIZE=%~2"
    shift
) else if "%~1"=="--generations" (
    set "GENERATIONS=%~2"
    shift
)
shift
goto parse

:afterparse

REM Nada de echo aqui! Se quiser debugar, joga pra arquivo:
REM echo elitism=%ELITISM% mutation=%MUTATION% pop=%POPSIZE% gen=%GENERATIONS%  >> "%THIS_DIR%runner_debug.log"

REM ==========================================
REM 4) Rodar main.py -> JSON -> compute_hv_direct.py
REM ==========================================

cd /d "%SRC_DIR%"

python main.py ^
    --elitism_fraction %ELITISM% ^
    --mutation_rate %MUTATION% ^
    --pop_size %POPSIZE% ^
    --generations %GENERATIONS% ^
    --write_log_file False ^
    2> NUL | python "%THIS_DIR%compute_hv_direct.py"

endlocal
