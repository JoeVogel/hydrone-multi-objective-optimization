import argparse

import airfoil

"""
Script to check airfoil aerodynamic coefficients at a given angle of attack and Reynolds number.

Usage example:

  python src/check_airfoil.py --airfoil NACA0012 --re 100000 --alpha 0.1

"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--airfoil", help="Airfoil name")
    ap.add_argument("--re", type=float, help="Reynolds number")
    ap.add_argument("--alpha", type=float, help="Angle in radians to check")
    args = ap.parse_args()
    
    af = airfoil.load_airfoil(args.airfoil)
    
    af.out_of_range_policy = "clamp_drag_penalty"
    af.enable_oor_logging = True
    
    cl = af.Cl(args.alpha, Re=args.re)
    cd = af.Cd(args.alpha, Re=args.re)
    cp_min = af.Cp_min(args.alpha, Re=args.re)
    stall_a = af.stall_alpha(Re=args.re)
    
    print(f"Airfoil: {args.airfoil}, Re={args.re}, alpha rad={args.alpha}, alpha deg={args.alpha*180.0/3.141592653589793}")
    print(f"  Cl = {cl:.4f}")   
    print(f"  Cd = {cd:.5f}")
    print(f"  Cp_min = {cp_min:.4f}")
    if stall_a is not None:
        print(f"  Stall alpha = {stall_a:.3f} deg")
    else:
        print("  Stall alpha = N/A")