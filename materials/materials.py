# =======Dependecies=======
# python 3.10.19
# matplotlib 3.10.8
# numpy 2.2.6
# =========================


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet
import numpy.typing as npt
from typing import List, Tuple, Union
import os

IMAGE_DIR = "levelset_images"

def createStoreImage() -> None:
    """
        Description: create directory for simulation images
    """
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Directory '{IMAGE_DIR}' created.")

def _setup_parabolic() -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
        Description: generate data for 3D parabolic surface
        Inputs:
        Outputs: surface data         
    """
    # prepare data(polar coordinates)
    r = np.linspace(0, np.sqrt(20), 50)
    theta = np.linspace(0, 2*np.pi, 100)
    r, theta = np.meshgrid(r, theta)
    # transforming polar coordiantes to cartesian coordinates
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    Surface = 10 - X**2 - Y**2
    return (X, Y, Surface)

def _setup_cubiSurface() -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
        Description: generate data for 3D cubic surface
        Inputs:
        Outputs: surface data         
    """
    X = np.arange(-4,4,0.01)
    Y = np.arange(-4,4,0.01)
    X, Y = np.meshgrid(X, Y, indexing='ij')
    Surface = X**3 + Y**2 - 6*X
    return (X, Y, Surface)

def _create_Plane(xyList:npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
        Description: generate data for inserting plane
        Inputs: domain range
        Outputs: plane data        
    """
    return tuple(np.meshgrid(xyList, xyList))

def _create_levelset(Xmesh:npt.NDArray[np.float64], Ymesh:npt.NDArray[np.float64], 
                  Surface:npt.NDArray[np.float64], zList:List[Union[int,float]]) -> QuadContourSet:
    """
        Description: generate level set data of correspoding z values
        Inputs: 
            Xmesh: X meshgrid
            Ymesh: Y meshgrid
            Surface: Surface of Xmesh and Ymesh
            zList: target z coordinates
        Outputs: level set data         
    """
    fig,ax = plt.subplots(figsize=(8,8),subplot_kw={"projection":"3d"})
    curves_set = ax.contour(Xmesh, Ymesh, Surface, levels=zList, alpha=0)
    plt.close(fig)
    return curves_set

def create_3D_parabolic(elevation: int, azimuth: int) -> None:
    """
        Description: generate image of 3D parabolic surface
        Inputs: 
            elevation: angle
            azimuth: angle
        Outputs: image.svg         
    """
    save_path = os.path.join(IMAGE_DIR, '3D_parabolic.svg')
    surface = _setup_parabolic()
    fig,ax = plt.subplots(figsize=(8,8),subplot_kw={"projection":"3d"})
    ax.plot_surface(surface[0],surface[1],surface[2],cmap='viridis',alpha=0.9)

    # transparent background
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    ax.set_xticks(range(-6,8,2))
    ax.set_yticks(range(-6,8,2))
    ax.set_zticks(range(-10,11,2))
    ax.view_init(elev=elevation, azim=azimuth)
    plt.savefig(save_path)
    print("Image saved")
    plt.close(fig)

def create_3D_parabolic_insert(elevation:int, azimuth: int) -> None:
    """
        Description: generate image of 3D parabolic surface with plane
        Inputs: 
            elevation: angle
            azimuth: angle
        Outputs: image.svg         
    """
    surface = _setup_parabolic()
    fig,ax = plt.subplots(figsize=(8,8),subplot_kw={"projection":"3d"})
    ax.plot_surface(surface[0],surface[1],surface[2],cmap='viridis',alpha=0.9)

    zList = [-9,-4,0,4,9]
    colorList = ['red','black','blue','orange','brown']
    level_sets = _create_levelset(surface[0],surface[1],surface[2],zList)
    for i, level_segments in enumerate(level_sets.allsegs):
        z_level = zList[i] 
        for seg in level_segments:
            x_line = seg[:, 0]
            y_line = seg[:, 1]
            z_line = np.full_like(x_line, z_level)
            ax.plot(x_line, y_line, z_line, color=colorList[i], linewidth=2, zorder=10)

    if elevation != 90:
        domain = np.arange(-6, 6, 0.5)
        plane_X, plane_Y = _create_Plane(domain)
        for z in zList:
            ax.plot_surface(plane_X, plane_Y, np.full_like(plane_X, z), color='black', alpha=0.1)

    # transparent background
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    ax.set_xticks(range(-6,8,2))
    ax.set_yticks(range(-6,8,2))
    ax.set_zticks(range(-10,11,2))
    ax.view_init(elev=elevation, azim=azimuth)
    if elevation == 90:
        ax.axis('off')
        save_path = os.path.join(IMAGE_DIR, '3D_parabolic_insert_upper.svg')
        plt.savefig(save_path)
    else:
        save_path = os.path.join(IMAGE_DIR, '3D_parabolic_insert.svg')
        plt.savefig(save_path)
    print("Image saved")
    plt.close(fig)

def create_cubicsurface(elevation:int, azimuth:int) -> None:
    """
        Description: generate image of 3D cubic surface
        Inputs: 
            elevation: angle
            azimuth: angle
        Outputs: image.svg         
    """
    save_path = os.path.join(IMAGE_DIR, '3D_cubicSurface.svg')
    surface = _setup_cubiSurface()

    fig,ax = plt.subplots(figsize=(8,8),subplot_kw={"projection":"3d"})
    ax.plot_surface(surface[0],surface[1],surface[2],cmap='viridis',alpha=0.9)

    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    ax.set_xticks(range(-6,8,2))
    ax.set_yticks(range(-6,8,2))
    ax.set_zticks(range(-50, 61, 10))
    ax.view_init(elev=elevation, azim=azimuth)
    plt.savefig(save_path)
    print('Image saved')
    plt.close(fig)

def create_cubicsurface_insert(elevation:int, azimuth:int) -> None:
    """
        Description: generate image of 3D cubic surface with plane
        Inputs: 
            elevation: angle
            azimuth: angle
        Outputs: image.svg         
    """
    surface = _setup_cubiSurface()

    fig,ax = plt.subplots(figsize=(8,8),subplot_kw={"projection":"3d"})
    ax.plot_surface(surface[0],surface[1],surface[2],cmap='viridis',alpha=0.9)

    zList = [-25,-8, 0, 8, 25]
    colorList = ['red','black','blue','orange','brown']
    level_sets = _create_levelset(surface[0],surface[1],surface[2],zList)
    for i, level_segments in enumerate(level_sets.allsegs):
        z_level = zList[i] 
        for seg in level_segments:
            x_line = seg[:, 0]
            y_line = seg[:, 1]
            z_line = np.full_like(x_line, z_level)
            ax.plot(x_line, y_line, z_line, color=colorList[i], linewidth=2, zorder=10)

    if elevation != 90:
        domain = np.arange(-6, 6, 0.5)
        plane_X, plane_Y = _create_Plane(domain)
        for z in zList:
            ax.plot_surface(plane_X, plane_Y, np.full_like(plane_X, z), color='black', alpha=0.1)

    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    ax.set_xticks(range(-6,8,2))
    ax.set_yticks(range(-6,8,2))
    ax.set_zticks(range(-40, 61, 10))
    ax.view_init(elev=elevation, azim=azimuth)
    if elevation == 90:
        ax.axis('off')
        save_path = os.path.join(IMAGE_DIR, '3D_cubicSurface_insert_upper.svg')
        plt.savefig(save_path)
    else:
        save_path = os.path.join(IMAGE_DIR, '3D_cubicSurface_insert.svg')
        plt.savefig(save_path)
    print('Image saved')
    plt.close(fig)
    

if __name__ == "__main__":
    createStoreImage()

    # Create 3 image of 3D parabolic surface
    create_3D_parabolic(elevation=20,azimuth=315)
    create_3D_parabolic_insert(elevation=20,azimuth=315)
    create_3D_parabolic_insert(elevation=90,azimuth=0)

    # Create 3 image of 3D cubic surface
    create_cubicsurface(elevation=20, azimuth=225)
    create_cubicsurface_insert(elevation=20, azimuth=225)
    create_cubicsurface_insert(elevation=90, azimuth=270)