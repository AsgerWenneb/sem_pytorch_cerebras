from data_export import export_etov, export_data_header, export_solution
from global_assembly import global_assembly_poisson
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def estimate_error(sol, exact_sol):
    # L2 Error in nodes
    error = sol - exact_sol
    L2_error = 1/len(error)*np.sqrt(np.sum(error**2))
    return L2_error


if __name__ == "__main__":
    # Define source term and boundary condition functions
    def q(x, y): 
        return -5*np.pi**2*np.sin(np.pi*x)*np.sin(2*np.pi*y)  
    
    def bc(x, y): 
        return 0
    
    def exact_solution(x, y):
        return np.sin(np.pi*x)*np.sin(2*np.pi*y)


    # Problem parameters
    elemsx = 5  # Number of elements in x
    elemsy = 5  # Number of elements in y
    errors = []
    P_range = range(4, 17)
    for P in P_range:
    # Assemble global system
        A, B, C, xv, yv = global_assembly_poisson(elemsx, elemsy, P, q, bc)
        sol = sparse.linalg.spsolve(A, B)
        err = estimate_error(sol, exact_solution(xv, yv))
        errors.append(err)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(P_range, errors, c='C1')
    ax.plot(P_range, errors, c='C1', alpha=0.6)
    ax.set_xlabel('Polynomial order P')
    ax.set_ylabel('Nodal L2 error')
    ax.set_yscale('log')
    ax.set_title('Nodal L2 error vs P (5x5 grid)')
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=50))
    ax.grid(which='major', axis='y', linestyle='--', alpha=0.9)
    ax.grid(which='major', axis='x', linestyle='--', alpha=0.9)
    ax.grid(which='minor', axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('errors_vs_P.png', dpi=150)
    plt.show()


    P = 1
    elems = range(5, 300, 5)
    errors = []
    for elems_count in elems:
        A, B, C, xv, yv = global_assembly_poisson(elems_count, elems_count, P, q, bc)
        sol = sparse.linalg.spsolve(A, B)
        err = estimate_error(sol, exact_solution(xv, yv))
        errors.append(err)
    
    
    elem_counts = list(elems)
    elem_counts = (4/np.array(elem_counts))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(elem_counts, errors, c='C1')
    ax.plot(elem_counts, errors, c='C1', alpha=0.6)
    ax.set_xlabel('element length h')
    ax.set_ylabel('Nodal L2 error')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(f'Nodal L2 error vs element length h (P = {P})')
    ax.grid(which='major', axis='y', linestyle='--', alpha=0.9)
    ax.grid(which='minor', axis='y', linestyle=':', alpha=0.6)
    ax.grid(which='major', axis='x', linestyle='--', alpha=0.9)
    ax.grid(which='minor', axis='x', linestyle=':', alpha=0.6)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig('errors_vs_elements.png', dpi=150)
    plt.show()


    # # Export data
    # n_corner_nodes = (elemsx + 1) * (elemsy + 1)
    # export_data_header("output_data.dat", n_corner_nodes, xv, yv)
    # export_solution("output_data.dat", sol)

    # export_etov("output_etov.csv", C)

