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