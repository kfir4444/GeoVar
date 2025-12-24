import matplotlib.pyplot as plt
import numpy as np
import os

def plot_reaction_path(objective, genes, output_path="reaction_path.png"):
    """
    Generates a single surveillance image containing a grid of subplots.
    Rows: Active Atoms
    Cols: X, Y, Z dimensions
    """
    curves = objective.decode_genes(genes)
    rc_grid = np.linspace(-1, 1, 100)
    dims = ['x', 'y', 'z']
    
    n_active = len(objective.active_indices)
    if n_active == 0:
        print("No active atoms to plot. The system is static.")
        return

    # Setup Grid: One row per atom, 3 columns (x,y,z)
    # Squeeze=False ensures we always address axes as [row, col]
    fig, axes = plt.subplots(nrows=n_active, ncols=3, figsize=(15, 3 * n_active), squeeze=False)
    
    fig.suptitle(f"Variational Reaction Path (Total Active Atoms: {n_active})", fontsize=16)

    for i in range(n_active):
        atom_global_idx = objective.active_indices[i]
        atom_symbol = objective.atoms[atom_global_idx]
        
        for j in range(3): # x, y, z
            curve_idx = i * 3 + j
            curve = curves[curve_idx]
            dim_label = dims[j]
            
            ax = axes[i, j]
            
            # Calculate trajectory
            trajectory = np.array([curve.value(rc) for rc in rc_grid])
            
            # Plot
            ax.plot(rc_grid, trajectory, color='tab:blue', linewidth=2)
            
            # Aesthetics
            ax.set_title(f"{atom_symbol}{atom_global_idx} - {dim_label.upper()}")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if i == n_active - 1:
                ax.set_xlabel("RC")
            
            # Highlight Start/End
            ax.scatter([-1], [trajectory[0]], color='green', s=30, label='R')
            ax.scatter([1], [trajectory[-1]], color='red', s=30, label='P')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
    
    # Ensure directory exists for the FILE
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Surveillance dashboard saved to: {output_path}")
