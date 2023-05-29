import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx, solar_field):
    """
    Calculate the acceleration on each particle due to electric field,
    including the influence of solar electric fields.

    pos           is an Nx1 matrix of particle positions
    Nx            is the number of mesh cells
    boxsize       is the domain [0, boxsize]
    n0            is the electron number density
    Gmtx          is an Nx x Nx matrix for calculating the gradient on the grid
    Lmtx          is an Nx x Nx matrix for calculating the laplacian on the grid
    solar_field   is an Nx1 matrix representing the solar electric field at each grid point
    a             is an Nx1 matrix of accelerations
    """
    N = pos.shape[0]
    dx = boxsize / Nx
    j = np.floor(pos / dx).astype(int)
    jp1 = j + 1
    weight_j = (jp1 * dx - pos) / dx
    weight_jp1 = (pos - j * dx) / dx
    jp1 = np.mod(jp1, Nx)
    n = np.bincount(j[:, 0], weights=weight_j[:, 0], minlength=Nx)
    n += np.bincount(jp1[:, 0], weights=weight_jp1[:, 0], minlength=Nx)
    n *= n0 * boxsize / N / dx

    phi_grid = spsolve(Lmtx, n - n0, permc_spec="MMD_AT_PLUS_A")
    E_grid = -Gmtx @ phi_grid + solar_field

    E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]
    a = -E

    return a


def main():
    """Plasma PIC simulation"""

    N = 40000
    Nx = 400
    t = 0
    tEnd = 50
    dt = 1
    boxsize = 50
    n0 = 1
    vb = 3
    vth = 1
    A = 0.1
    plotRealTime = True

    np.random.seed(42)
    pos = np.random.rand(N, 1) * boxsize
    vel = vth * np.random.randn(N, 1) + vb
    Nh = int(N / 2)
    vel[Nh:] *= -1
    vel *= (1 + A * np.sin(2 * np.pi * pos / boxsize))

    dx = boxsize / Nx
    e = np.ones(Nx)
    diags = np.array([-1, 1])
    vals = np.vstack((-e, e))
    Gmtx = sp.spdiags(vals, diags, Nx, Nx)
    Gmtx = sp.lil_matrix(Gmtx)
    Gmtx[0, Nx - 1] = -1
    Gmtx[Nx - 1, 0] = 1
    Gmtx /= (2 * dx)
    Gmtx = sp.csr_matrix(Gmtx)

    diags = np.array([-1, 0, 1])
    vals = np.vstack((e, -2 * e, e))
    Lmtx = sp.spdiags(vals, diags, Nx, Nx)
    Lmtx = sp.lil_matrix(Lmtx)
    Lmtx[0, Nx - 1] = 1
    Lmtx[Nx - 1, 0] = 1
    Lmtx /= dx ** 2
    Lmtx = sp.csr_matrix(Lmtx)

    # Generate solar electric field
    k = 2 * np.pi / boxsize  # Wave number of the solar field
    omega = 0.2             # Frequency of the solar field
    x_grid = np.linspace(0, boxsize, Nx)
    solar_field = np.sin(k * x_grid) * np.sin(omega * t)

    acc = getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx, solar_field)

    Nt = int(np.ceil(tEnd / dt))

    fig = plt.figure(figsize=(5, 4), dpi=80)

    for i in range(Nt):
        #vel += acc * dt / 2.0
        pos += vel * dt
        pos = np.mod(pos, boxsize)
        acc = getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx, solar_field)
        vel += acc * dt / 2.0
        t += dt

        # Update solar electric field
        #solar_field = np.sin(k * x_grid) * np.sin(omega * t)

        if plotRealTime or (i == Nt - 1):
            plt.cla()
            plt.scatter(pos[0:Nh], vel[0:Nh], s=.4, color='blue', alpha=0.5)
            plt.scatter(pos[Nh:], vel[Nh:], s=.4, color='red', alpha=0.5)
            plt.axis([0, boxsize, -6, 6])

            plt.pause(0.001)

    plt.xlabel('x')
    plt.ylabel('v')
    plt.savefig('pic.png', dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
