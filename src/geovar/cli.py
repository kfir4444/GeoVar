import argparse
import sys
from geovar.geom import read_xyz, write_xyz, identify_active_indices, align_product_to_reactant
from geovar.opt import optimize_path

def main():
    parser = argparse.ArgumentParser(description="GeoVar: Variational Geodesic Initialization for Transition States")
    parser.add_argument("reactant", help="Path to Reactant XYZ file")
    parser.add_argument("product", help="Path to Product XYZ file")
    parser.add_argument("output", help="Path to Output TS XYZ file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # 1. Read
    if args.verbose:
        print(f"Reading {args.reactant}...")
    try:
        atoms_r, coords_r = read_xyz(args.reactant)
    except Exception as e:
        print(f"Error reading reactant: {e}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Reading {args.product}...")
    try:
        atoms_p, coords_p = read_xyz(args.product)
    except Exception as e:
        print(f"Error reading product: {e}")
        sys.exit(1)
    
    if atoms_r != atoms_p:
        print("Error: Atom lists do not match between Reactant and Product.")
        sys.exit(1)
        
    atoms = atoms_r
    
    # 2. Identify Active Set
    if args.verbose:
        print("Identifying Active Set...")
    try:
        active_indices, spectator_indices = identify_active_indices(atoms, coords_r, coords_p)
    except ImportError as e:
        print(f"Error: {e}")
        print("Please ensure rdkit is installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error identifying active indices: {e}")
        sys.exit(1)
        
    active_indices = list(active_indices)
    spectator_indices = list(spectator_indices)

    if args.verbose:
        print(f"Active Atoms: {len(active_indices)} indices: {sorted(active_indices)}")
        print(f"Spectator Atoms: {len(spectator_indices)}")

    if len(active_indices) == 0:
        print("Warning: No active atoms detected (Geometries are identical or bond orders didn't change).")
        write_xyz(args.output, atoms, coords_r, comment="GeoVar: Identical Reactant/Product")
        sys.exit(0)

    # 3. Align
    if args.verbose:
        print("Aligning Product to Reactant...")
    coords_p_aligned = align_product_to_reactant(coords_r, coords_p, spectator_indices)
    
    # 4. Optimize
    if args.verbose:
        print("Optimizing Path...")
    ts_coords, res = optimize_path(atoms, coords_r, coords_p_aligned, active_indices, spectator_indices, verbose=args.verbose)
    
    if args.verbose:
        print(f"Optimization Success: {res.success}")
        print(f"Final Loss: {res.fun}")

    # 5. Output
    write_xyz(args.output, atoms, ts_coords, comment=f"GeoVar TS Guess | Loss: {res.fun:.4f}")
    if args.verbose:
        print(f"Wrote TS guess to {args.output}")

if __name__ == "__main__":
    main()
