# ======Dependecies=======
# python 3.10.19
# matplotlib 3.10.8
# numpy 2.2.6
# scikit-image 0.25.2
# tqdm 4.67.1
# ========================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple, Union
import numpy.typing as npt
from skimage import measure
from skimage.filters import gaussian
from tqdm import tqdm
import os


# =============================
#     Simulation Parameters
# =============================
MASK_TYPE = 3
MASK_LENGTH = 0.6

N = 200
TARGET_TIME = 0.5

X_MIN, X_MAX = -3.0, 3.0
Y_MIN, Y_MAX = -3.0, 3.0
Z_MIN, Z_MAX = -1.0, 0.5

IMAGE_DIR = "simulation_images"
# =============================

def createStoreImage() -> None:
    """
        Description: create directory for simulation images
    """
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Directory '{IMAGE_DIR}' created.")

def _setMask(X:npt.NDArray, Y:npt.NDArray, maskType:int, maskSize:float) -> npt.NDArray:
    """
        Description: Returns different mask for etching depending on type value
        Inputs:
            X: X meshgrid value
            Y: Y meshgrid value
            maskType: 1(Circle), 2(Square), 3(Trench), 4(Cross)
            maskSize: mask size
        Output: mask
    """
    mask=None
    if maskType == 1:
        mask = (X**2 + Y**2 >= maskSize**2)
    elif maskType== 2:
        mask = ~((np.abs(X) < maskSize) & (np.abs(Y) < maskSize))
    elif maskType == 3:
        mask = ~((np.abs(X) < maskSize))
    elif maskType == 4:
        mask = ~((np.abs(X) < maskSize) | (np.abs(Y) < maskSize))
    return mask

def _convertCoord(verts:npt.NDArray) -> npt.NDArray:
    """
        Description: Converts node numbers to cartesian coordinates
        Inputs:
            verts: node numbers
        Outputs:
            verts: cartesian coordinates
    """
    dx, dy, dz = _returnDelta()
    verts[:, 0] = X_MIN + verts[:, 0] * dx
    verts[:, 1] = Y_MIN + verts[:, 1] * dy
    verts[:, 2] = Z_MIN + verts[:, 2] * dz
    return verts

def _returnDelta() -> Tuple[float]:
    """
        Description: calculates delta value
        Inputs:
        Outputs:
            (dx, dy, dz): delta x, delta y, delta z
    """
    dx = (X_MAX - X_MIN) / (N - 1)
    dy = (Y_MAX - Y_MIN) / (N - 1)
    dz = (Z_MAX - Z_MIN) / (N - 1)
    return (dx, dy, dz)

def print_etch_depth(phi: npt.NDArray, title: str) -> None:
    """
        Description: Calculates and prints the maximum etch depth.
        Inputs:
            phi: level set
            title: 
        Output: simulation etch depth
    """
    dx, dy, dz = _returnDelta()
    try:
        verts, faces, normals, values = measure.marching_cubes(phi, 0)
        z_coords = Z_MIN + verts[:, 2] * dz
        min_z = z_coords.min()
        print(f"[{title}] Max Etch Depth: {min_z:.4f} (Target was ~{TARGET_TIME if TARGET_TIME else 'N/A'})")
    except ValueError:
        print(f"[{title}] Surface not found (Completely etched away or error).")


def godunovUpwindScheme(phi:npt.NDArray, dx:float, dy:float, dz:float) -> npt.NDArray:
    """
        Description: Implementaion of godunow upwind difference scheme
        Input:
            phi: domain
            dx: delta x
            dy: delta y
            dz: delta z
        Output: gradient magnitude
    """
    P = np.pad(phi, 1, mode='edge')     # Padding for keeping numpy array size same

    BD_x_minus = (P[1:-1, 1:-1, 1:-1] - P[0:-2, 1:-1, 1:-1]) / dx
    FD_x_plus  = (P[2:,   1:-1, 1:-1] - P[1:-1, 1:-1, 1:-1]) / dx
    BD_y_minus = (P[1:-1, 1:-1, 1:-1] - P[1:-1, 0:-2, 1:-1]) / dy
    FD_y_plus  = (P[1:-1, 2:,   1:-1] - P[1:-1, 1:-1, 1:-1]) / dy
    BD_z_minus = (P[1:-1, 1:-1, 1:-1] - P[1:-1, 1:-1, 0:-2]) / dz
    FD_z_plus  = (P[1:-1, 1:-1, 2:]   - P[1:-1, 1:-1, 1:-1]) / dz

    # --------------- Riemann Solver -------------------
    # Etcing equation is a convex function which simplfy
    # godunov hamilton in a simple closed from solution.
    gradX = np.maximum(np.maximum(BD_x_minus, 0)**2, np.minimum(FD_x_plus, 0)**2)
    gradY = np.maximum(np.maximum(BD_y_minus, 0)**2, np.minimum(FD_y_plus, 0)**2)
    gradZ = np.maximum(np.maximum(BD_z_minus, 0)**2, np.minimum(FD_z_plus, 0)**2)

    norm_grad = np.sqrt(gradX + gradY + gradZ+ 1e-20)
    return norm_grad

def godunovVerticalOnly(phi:npt.NDArray, dz:float) -> npt.NDArray:
    """
    Description: Calculates gradient ONLY in Z direction (Fixed: keeps array size same)
    """
    P = np.pad(phi, 1, mode='edge') 

    BD_z_minus = (P[1:-1, 1:-1, 1:-1] - P[1:-1, 1:-1, 0:-2]) / dz
    FD_z_plus  = (P[1:-1, 1:-1, 2:]   - P[1:-1, 1:-1, 1:-1]) / dz
    gradZ = np.maximum(np.maximum(BD_z_minus, 0)**2, np.minimum(FD_z_plus, 0)**2)
    
    # Return purely vertical gradient magnitude (Shape: N, N, N)
    return np.sqrt(gradZ + 1e-20)

def simulator(Viso:float, Vaniso:float) -> npt.NDArray:
    """
        Description: Main algorithm for etching simulation
        Inputs:
            steps: total iteration
            Viso: Phenomenological model velocity for isotropic etching
            Vaniso: Phenomenological model velocity for anisotropic etching
        Outputs:
            Phi: level set
    """
    dx, dy, dz = _returnDelta()
    x = np.linspace(X_MIN, X_MAX, N)
    y = np.linspace(Y_MIN, Y_MAX, N)
    z = np.linspace(Z_MIN, Z_MAX, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    Phi = -Z 
    Is_Under_Mask = _setMask(X,Y,MASK_TYPE,MASK_LENGTH)

    cfl_number = 0.25
    max_velocity = Viso + Vaniso
    if max_velocity == 0: max_velocity = 1.0

    dt = cfl_number * min(dx,dy,dz) / max_velocity
    steps = int(TARGET_TIME / dt)

    print(f"\n[Simulation Config: Grid N={N}, dt={dt:.5f}, Total Steps={steps}]")

    for t in tqdm(range(steps), desc='PROGRESS', mininterval=0.001):

        grad_Mag_All = godunovUpwindScheme(Phi, dx, dy, dz)
        grad_Mag_Z = godunovVerticalOnly(Phi, dz)

        grad = np.gradient(Phi, dx, dy, dz)
        norm = np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2 + 1e-10)
        nz = np.abs(grad[2] / norm)

        Is_Shadow_Zone = Is_Under_Mask               
        Is_Solid_Mask = (Is_Under_Mask) & (Z >= 0.0)
        
        sidewall_threshold = 0.5
        is_dry_blocked = (Is_Shadow_Zone) | (nz < sidewall_threshold)
        
        dry_rate = np.where(is_dry_blocked, 0.0, Vaniso)
        dry_change = dry_rate * grad_Mag_Z

        nz_upper = 0.995 
        nz_lower = 0.900
        ramp_factor = (nz_upper - nz) / (nz_upper - nz_lower)
        wet_scale = np.clip(ramp_factor, 0.0, 1.0)
        
        wet_rate = np.where(Is_Shadow_Zone, Viso * wet_scale, Viso)
        wet_change = wet_rate * grad_Mag_All

        total_change = dry_change + wet_change
        final_change = np.where(Is_Solid_Mask, 0.0, total_change)
        
        # Level Set Update
        Phi = Phi - final_change * dt
        
    return Phi

def simulator1(Viso:float, Vaniso:float) -> npt.NDArray:
    """
        Description: Main algorithm for etching simulation
        Inputs:
            steps: total iteration
            Viso: Phenomenological model velocity for isotropic etching
            Vaniso: Phenomenological model velocity for anisotropic etching
        Outputs:
            Phi: level set
    """
    dx, dy, dz = _returnDelta()
    x = np.linspace(X_MIN, X_MAX, N)
    y = np.linspace(Y_MIN, Y_MAX, N)
    z = np.linspace(Z_MIN, Z_MAX, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


    Phi = -Z 
    Is_Under_Mask = _setMask(X,Y,MASK_TYPE,MASK_LENGTH)

    cfl_number = 0.25
    max_velocity = Viso + Vaniso
    if max_velocity == 0: max_velocity = 1.0

    dt = cfl_number * min(dx,dy,dz) / max_velocity
    steps = int(TARGET_TIME / dt)

    print(f"\n[Simulation Config: Grid N={N}, dt={dt:.5f}, Total Steps={steps}]")

    for t in tqdm(range(steps), desc='PROGRESS', mininterval=0.001):
        gradUpwind = godunovUpwindScheme(Phi, dx, dy, dz)
        
        grad = np.gradient(Phi, dx, dy, dz)
        norm_grad_central = np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2 + 1e-10)
        nz = np.abs(grad[2] / norm_grad_central)

        # Phenomenological model velocity: z direction
        V_physics = Viso + (Vaniso * nz)
        V_protected = Viso * (1.0 - nz)
        V_effective = np.where(Is_Under_Mask, V_protected, V_physics)
        V_final = V_effective

        # Level Set Update
        Phi = Phi - V_final * gradUpwind * dt
    return Phi


def plotResult(levelSet:npt.NDArray, elevation:int, azimuth:int, title) -> None:
    """
        Description: save simulated image files to specific directory
        Inputs:
            levelSet: surface data
            elevation: elevation angle
            azimuth: azimuth angle
            title: title for image
    """
    save_path = os.path.join(IMAGE_DIR, f'{title}.svg')
    fig,ax = plt.subplots(figsize=(8, 8),subplot_kw={"projection":"3d"})
    try:
        verts, faces, normals, values = measure.marching_cubes(levelSet, 0)
        verts = _convertCoord(verts=verts)
        mesh = Poly3DCollection(verts[faces], alpha=0.8)
        mesh.set_facecolor('dodgerblue')
        mesh.set_edgecolor('k')
        mesh.set_linewidth(0.05)
        ax.add_collection3d(mesh)
        
    except ValueError:
        print("Surface not found within volume.")
    
    mask_N = 200
    mask_x_lin = np.linspace(X_MIN, X_MAX, mask_N)
    mask_y_lin = np.linspace(Y_MIN, Y_MAX, mask_N)
    Mask_X, Mask_Y = np.meshgrid(mask_x_lin, mask_y_lin)

    Mask_Z_pos = 0.001      # z axis position of mask
    Mask_Z = np.full_like(Mask_X, Mask_Z_pos)
    
    mask_boolean = _setMask(Mask_X,Mask_Y,MASK_TYPE,MASK_LENGTH)
    Mask_Z[~mask_boolean] = np.nan

    surf = ax.plot_surface(Mask_X, Mask_Y, Mask_Z, color='gray', alpha=0.4, 
                           linewidth=0, antialiased=False, rstride=2, cstride=2, shade=True)

    range_x = X_MAX - X_MIN
    range_y = Y_MAX - Y_MIN
    range_z = Z_MAX - Z_MIN
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_zlim(Z_MIN, Z_MAX)
    ax.set_box_aspect((range_x, range_y, range_z))
    ax.grid('off')
    ax.axis(False)
    ax.view_init(elev=elevation, azim=azimuth)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"{title} saved")
    plt.close(fig)

if __name__ == "__main__":

    createStoreImage()

    # Case1: Isotropic etching
    V_iso_1 = 1.0
    V_aniso_1 = 0.0
    result_1 = simulator(Viso=V_iso_1,Vaniso=V_aniso_1)
    print_etch_depth(result_1,"Isotropic")
    plotResult(levelSet=result_1, elevation=20, azimuth=45, title="Isotropic etching_1")
    plotResult(levelSet=result_1, elevation=0, azimuth=90, title="Isotropic etching_2")
    plotResult(levelSet=result_1, elevation=-150, azimuth=45, title="Isotropic etching_3")

    # Case2: Anisotropic etching
    V_iso_2 = 0.0
    V_aniso_2 = 1.0
    result_2 = simulator(Viso=V_iso_2,Vaniso=V_aniso_2)
    print_etch_depth(result_2,"Anisotropic")
    plotResult(levelSet=result_2, elevation=20, azimuth=45, title="Anisotropic etching_1")
    plotResult(levelSet=result_2, elevation=0, azimuth=90, title="Anisotropic etching_2")
    plotResult(levelSet=result_2, elevation=-150, azimuth=45, title="Anisotropic etching_3")
    