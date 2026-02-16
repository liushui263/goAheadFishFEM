import json
import sys
import math
import argparse
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def load_results(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        sys.exit(1)
    
    real = data["solution"]["real"]
    imag = data["solution"]["imag"]
    solver = data["metadata"].get("solver_type", "Unknown")
    
    # Reconstruct complex numbers
    sol = [complex(r, i) for r, i in zip(real, imag)]
    return solver, sol

def compare(file1, file2):
    name1, sol1 = load_results(file1)
    name2, sol2 = load_results(file2)
    
    print(f"========================================")
    print(f"Comparing Results")
    print(f"  File 1: {file1} ({name1})")
    print(f"  File 2: {file2} ({name2})")
    print(f"========================================")
    
    if len(sol1) != len(sol2):
        print(f"Error: Dimension mismatch {len(sol1)} vs {len(sol2)}")
        return

    diff_norm_sq = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    
    for v1, v2 in zip(sol1, sol2):
        diff = v1 - v2
        diff_norm_sq += abs(diff)**2
        norm1_sq += abs(v1)**2
        norm2_sq += abs(v2)**2
        
    diff_norm = math.sqrt(diff_norm_sq)
    norm1 = math.sqrt(norm1_sq)
    norm2 = math.sqrt(norm2_sq)
    
    rel_error = diff_norm / norm1 if norm1 > 0 else 0.0
    
    print(f"L2 Norm {name1}: {norm1:.6e}")
    print(f"L2 Norm {name2}: {norm2:.6e}")
    print(f"Absolute Diff (L2): {diff_norm:.6e}")
    print(f"Relative Error:     {rel_error:.6e}")
    print(f"----------------------------------------")
    
    if HAS_MATPLOTLIB:
        plot_results(name1, sol1, name2, sol2)
    else:
        print("Warning: matplotlib not found, skipping plots.")

def plot_results(name1, sol1, name2, sol2):
    idx = np.arange(len(sol1))
    mag1 = np.array([abs(v) for v in sol1])
    mag2 = np.array([abs(v) for v in sol2])
    diff_mag = np.array([abs(v1 - v2) for v1, v2 in zip(sol1, sol2)])
    
    plt.figure(figsize=(14, 10))

    # 1. Magnitude Comparison
    plt.subplot(2, 2, 1)
    plt.plot(idx, mag1, label=name1, alpha=0.8)
    plt.plot(idx, mag2, label=name2, alpha=0.8, linestyle='--')
    plt.title('Field Magnitude |E|')
    plt.xlabel('Index (Flattened Grid)')
    plt.ylabel('|E|')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Difference (Error)
    plt.subplot(2, 2, 2)
    plt.plot(idx, diff_mag, label='|E1 - E2|', color='red', linewidth=0.5)
    plt.title('Absolute Difference')
    plt.xlabel('Index')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # 3. Parity Plot (Real Part)
    real1 = np.array([v.real for v in sol1])
    real2 = np.array([v.real for v in sol2])
    plt.subplot(2, 2, 3)
    plt.scatter(real1, real2, alpha=0.5, s=2)
    plt.plot([real1.min(), real1.max()], [real1.min(), real1.max()], 'r--', linewidth=1)
    plt.title(f'Real Part: {name1} vs {name2}')
    plt.grid(True, alpha=0.3)

    # 4. Parity Plot (Imaginary Part)
    imag1 = np.array([v.imag for v in sol1])
    imag2 = np.array([v.imag for v in sol2])
    plt.subplot(2, 2, 4)
    plt.scatter(imag1, imag2, alpha=0.5, s=2)
    plt.plot([imag1.min(), imag1.max()], [imag1.min(), imag1.max()], 'r--', linewidth=1)
    plt.title(f'Imag Part: {name1} vs {name2}')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_plots.png')
    print("Plots saved to 'comparison_plots.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FEM Solver JSON results")
    parser.add_argument("file1", help="First JSON result file (e.g., cpu_results.json)")
    parser.add_argument("file2", help="Second JSON result file (e.g., gpu_results.json)")
    args = parser.parse_args()
    
    compare(args.file1, args.file2)