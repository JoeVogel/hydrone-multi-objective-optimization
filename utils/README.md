## Polars Generate

The current directory contains some files (.inp format) that contains the instructions to XFOIL (also included). You can generate polars using (Windows only):

```
xfoil.exe < <.inp file>
```

## Convert to csv file for airfoil.py

You can use the xfoil_polar_to_airfoiltools.py to convert to the format accepted by airfoil.py, for example:

```
python xfoil_polar_to_airfoiltools.py NACA0018_10000.txt NACA0018_10000.csv --airfoil naca0018-il --re 10000 --ncrit 5 --mach 0 --stall-mode drop --stall-drop-pct 0.05

or
    
python xfoil_polar_to_airfoiltools.py NACA_0018_10000.txt NACA_0018_10000.csv --re 10000
    
```

## Generate Cp_min file

After generate a polar file (txt) ou can use the ```xfoil_polar_to_airfoiltools.py``` module to generate the Cp_min file. Two ways are allowed:

1. When XFOIL has the dat loaded by default, like the NACA airfoils

```
    python make_cpmin_from_pol.py \
        --pol E63_200000.txt \
        --naca 0018 \
        --re 200000 \
        --out-dir run_naca0018_re200k \
        --vacc 0 \
        --iter 800 \
        --n-pan 320 \
        --stall-alpha 10.5
```

2. XFOIL has not the airfoil coordinates, in these cases you need a coordinates file (Airfoil Tools site has the coordinates for many airfoils), usually a .dat file

```
    python make_cpmin_from_pol.py \
        --pol NACA0018_10000.txt \
        --foil-file NACA0018.dat \
        --re 10000 \
        --out-dir run_naca0018_re10000 \
        --vacc 0 \
        --iter 800 \
        --n-pan 320 \
        --stall-alpha 10.5
```

## Generate Cp_min files (after fail of previous section)

Sometimes, make_cpmin_from_pol stucks generating cpwr for a alpha. If it's the case and the program did generate the txt file for enough angles, you can stop (ctrl + c) the execution and run the ```make_cpmin_from_cp_folder``` to generate the cp_min file for the angles that the previous module did generate de CPWR output. For example: 

```
python make_cpmin_from_cp_folder.py \
    --cp-dir run_e63_re500000\cp \
    --airfoil-name e63 \
    --re 500000 \
    --out-dir run_e63_re500000
```

For a folder called ```run_e63_re500000```.