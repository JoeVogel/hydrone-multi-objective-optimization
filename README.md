# hydrone-multi-objective-optimization
Repository of artifacts generated for Master's dissertation.

## Modules

### Airfoil Module

This module provides tools for loading airfoil data and retrieving aerodynamic coefficients (lift and drag) for use in solvers such as Blade Element Momentum Theory (BEMT). It also supports visualization of the aerodynamic polars for different airfoils.

#### Features

- Load airfoil polar data in **AeroDyn format** (single polar table).
- Data must be available from -180° to 180°.
- Provides interpolated lift (Cl) and drag (Cd) coefficients using quadratic interpolation.
- Normalize angles automatically for use in solvers.
- Command-line execution to quickly plot Cl and Cd curves.

#### Repository Structure

- airfoils/    - Folder containing airfoil polar data in .dat format
- airfoil.py   - Module implementation

#### Usage from the Command Line

You can run the module directly to visualize the polar data of an airfoil:

```
python airfoil.py NACA_4412
```
This will plot the lift and drag coefficients across the full range of attack angles [−180°, 180°]

You can also restrict the x-axis range:

```
python airfoil.py NACA_4412 -20 20
```

This will plot only the range from −20° to 20°.

#### API Reference

##### class Airfoil

Represents an airfoil with lift and drag data.

- Attributes:

    - name: Airfoil name.
    - alpha_: Angle of attack samples (degrees).
    - Cl_: Lift coefficient samples.
    - Cd_: Drag coefficient samples.
    - Cl_func, Cd_func: Quadratic interpolation functions.
    - zero_lift: Reference angle offset.

- Methods:

    - Cl(alpha: float) -> float: Returns lift coefficient for angle alpha in radians.
    - Cd(alpha: float) -> float: Returns drag coefficient for angle alpha in radians.
    - plot(color='k'): Plots Cl and Cd vs angle of attack.

##### function load_airfoil(name: str) -> Airfoil

Loads an airfoil from the airfoils/ folder given its name (without extension).