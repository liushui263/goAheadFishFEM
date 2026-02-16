import json
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_field_slices(filename, output_file=None):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)

    # Extract field data
    if "solution" not in data:
        print("Error: 'solution' field not found in JSON.")
        sys.exit(1)
        
    real = np.array(data["solution"]["real"])
    imag = np.array(data["solution"]["imag"])
    magnitude = np.abs(real + 1j * imag)

    # Extract coordinates
    if "mesh" not in data or "x" not in data["mesh"]:
        print("Error: Mesh coordinates not found in JSON.")
        sys.exit(1)

    x_idx = np.array(data["mesh"]["x"], dtype=int)
    y_idx = np.array(data["mesh"]["y"], dtype=int)
    z_idx = np.array(data["mesh"]["z"], dtype=int)

    # Determine grid dimensions
    nx = np.max(x_idx) + 1
    ny = np.max(y_idx) + 1
    nz = np.max(z_idx) + 1

    print(f"Grid dimensions: {nx}x{ny}x{nz}")

    # Populate 3D grid with (X, Y, Z) layout
    grid = np.full((nx, ny, nz), np.nan)
    grid[x_idx, y_idx, z_idx] = magnitude

    # Define slice indices (middle of the domain)
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    # Create plots
    fig = plt.figure(figsize=(16, 8))
    
    # 1. 3D Orthogonal Slices
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.viridis

    # Slice X (YZ plane)
    y = np.arange(ny)
    z = np.arange(nz)
    Y, Z = np.meshgrid(y, z)
    X = np.full_like(Y, cx)
    data_x = grid[cx, :, :].T
    colors_x = cmap(norm(data_x))
    ax0.plot_surface(X, Y, Z, facecolors=colors_x, shade=False, rstride=1, cstride=1, alpha=0.5)

    # Slice Y (XZ plane)
    x = np.arange(nx)
    z = np.arange(nz)
    X, Z = np.meshgrid(x, z)
    Y = np.full_like(X, cy)
    data_y = grid[:, cy, :].T
    colors_y = cmap(norm(data_y))
    ax0.plot_surface(X, Y, Z, facecolors=colors_y, shade=False, rstride=1, cstride=1, alpha=0.5)

    # Slice Z (XY plane)
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, cz)
    data_z = grid[:, :, cz].T
    colors_z = cmap(norm(data_z))
    ax0.plot_surface(X, Y, Z, facecolors=colors_z, shade=False, rstride=1, cstride=1, alpha=0.5)

    ax0.set_title('3D Orthogonal Slices')
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_zlabel('Z')
    ax0.set_xlim(0, nx)
    ax0.set_ylim(0, ny)
    ax0.set_zlim(0, nz)

    # 2. 3D Isosurface (Scatter plot of high intensity field)
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    max_val = np.nanmax(grid)
    threshold = max_val * 0.35
    
    mask = (grid > threshold)
    xs, ys, zs = np.where(mask)
    vals = grid[mask]
    
    if len(xs) > 10000:
        skip = len(xs) // 10000
        xs, ys, zs, vals = xs[::skip], ys[::skip], zs[::skip], vals[::skip]
        
    ax1.scatter(xs, ys, zs, c=vals, cmap='viridis', alpha=0.5, s=3)
    ax1.set_title(f'3D Field > {threshold:.2e}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(0, nx)
    ax1.set_ylim(0, ny)
    ax1.set_zlim(0, nz)

    plt.suptitle(f'3D Field Magnitude Slices: {filename}', fontsize=16)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3D slices of FEM field solution")
    parser.add_argument("filename", help="Path to JSON result file")
    parser.add_argument("--output", "-o", help="Output image file (e.g. slices.png)")
    args = parser.parse_args()

    plot_field_slices(args.filename, args.output)