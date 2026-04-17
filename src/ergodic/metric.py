import numpy as np

def generate_k_vectors(num_k_per_dim):
    ks_dim1, ks_dim2 = np.meshgrid(
        np.arange(num_k_per_dim),
        np.arange(num_k_per_dim)
    )
    ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T
    return ks


def create_grid(L_list, grid_size):
    grids_x, grids_y = np.meshgrid(
        np.linspace(0, L_list[0], grid_size),
        np.linspace(0, L_list[1], grid_size)
    )
    grids = np.array([grids_x.ravel(), grids_y.ravel()]).T

    dx = L_list[0] / (grid_size - 1)
    dy = L_list[1] / (grid_size - 1)

    return grids, dx, dy


def compute_phi(pdf, ks, grids, dx, dy, L_list):
    """
    pdf: function (N,2) -> (N,)
    ks: (K,2)
    grids: (N,2)
    """

    coefficients = np.zeros(ks.shape[0])

    pdf_vals = pdf(grids)

    for i, k_vec in enumerate(ks):
        fk_vals = np.prod(
            np.cos(np.pi * k_vec / L_list * grids),
            axis=1
        )

        hk = np.sqrt(np.sum(fk_vals**2) * dx * dy)
        fk_vals /= hk

        phik = np.sum(fk_vals * pdf_vals) * dx * dy
        coefficients[i] = phik

    return coefficients