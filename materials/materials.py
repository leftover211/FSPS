# =======Dependencies=======
# python 3.10.19
# matplotlib 3.10.8
# numpy 2.2.6
# ==========================


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import QuadContourSet
import numpy.typing as npt
from typing import List, Tuple, Union
import os

IMAGE_DIR = "levelset_images"

def _create_2D_circle(radius:Union[int,float]) -> Tuple[npt.NDArray,npt.NDArray,npt.NDArray]:
    """
        Description: Create a 2D circle with a given radius
        Inputs:
            radius: radius of the circle
        Outputs:
            x_circle: x-coordinates of the circle
            y_circle: y-coordinates of the circle
            z_circle: z-coordinates of the circle (all zeros)
    """
    domain = np.linspace(0, 2 * np.pi, 100)
    x_circle = radius * np.cos(domain)
    y_circle = radius * np.sin(domain)
    z_circle = np.zeros_like(domain)
    return x_circle, y_circle, z_circle

def _generate_points(Flag:bool, Number:int) -> Tuple[npt.NDArray,npt.NDArray,npt.NDArray]:
    """
        Description: Generate random points in 2D space
        Inputs:
            Flag: True for random points, False for fixed points
            Number: Number of points to generate
        Outputs:
            random_x: x-coordinates of the points
            random_y: y-coordinates of the points
            random_z: z-coordinates of the points (all zeros)
    """
    if Flag:
        points = Number
        random_x = np.random.uniform(-6, 6, points)
        random_y = np.random.uniform(-6, 6, points)
        random_z = np.zeros(points)
    else:
        random_x = np.array([5, -4, 3, 2, 1, -3, -1])
        random_y = np.array([3, 2, -1, 1, 0, -2, 4])
        random_z = np.zeros_like(random_x)
    return random_x, random_y, random_z

def _set_alpha(elevation: int) -> float:
    """
        Description: Set the transparency of the points based on the elevation angle
    """
    return 1.0 if elevation == 90 else 0.6

def _set_size(Flag: bool) -> float:
    """
        Description: Set the size of the points based on the Flag
    """
    return 0.1 if Flag else 20.0


def level_set_vis(elevation: int, azimuth: int, arrow: bool, normal: bool, lev: bool) -> None:
    """
        Description: Visualize the level set of a 2D circle with random points
        Inputs:
            elevation: elevation angle of the camera
            azimuth: azimuth angle of the camera
            arrow: True to show the arrows, False otherwise
            normal: True to show the normals, False otherwise
            lev: True to show the level set, False otherwise
        Outputs:
            None
    """
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})

    RADIUS = 3.0
    x_circle, y_circle, z_circle = _create_2D_circle(radius=RADIUS)
    
    num_points = 20000 if lev else 7 
    rnd_x, rnd_y, rnd_z = _generate_points(Flag=lev, Number=num_points)
    
    alpha = _set_alpha(elevation=elevation)
    size = _set_size(Flag=lev)

    dist = np.sqrt(rnd_x**2 + rnd_y**2)
    
    inside = dist < RADIUS
    outside = ~inside
    
    sdf = np.zeros_like(dist)
    sdf[inside] = np.abs(RADIUS - dist[inside])
    sdf[outside] = -np.abs(dist[outside] - RADIUS)

    with np.errstate(divide='ignore', invalid='ignore'):
        dir_x = rnd_x / dist
        dir_y = rnd_y / dist
    dir_x = np.nan_to_num(dir_x)
    dir_y = np.nan_to_num(dir_y)

    if not lev:
        ax.scatter(rnd_x[inside], rnd_y[inside], rnd_z[inside], 
                   color='blue', s=size, marker='x', alpha=alpha)
        ax.scatter(rnd_x[outside], rnd_y[outside], rnd_z[outside], 
                   color='blue', s=size, marker='o', alpha=alpha)

    if arrow:
        if not lev:
            len_in = np.abs(RADIUS - dist[inside])
            ax.quiver(rnd_x[inside], rnd_y[inside], rnd_z[inside], 
                      dir_x[inside] * len_in, dir_y[inside] * len_in, 0, 
                      color='cyan', arrow_length_ratio=0.3)
            len_out = np.abs(dist[outside] - RADIUS)
            ax.quiver(rnd_x[outside], rnd_y[outside], rnd_z[outside], 
                      -dir_x[outside] * len_out, -dir_y[outside] * len_out, 0, 
                      color='coral', arrow_length_ratio=0.3)

        if normal:
            if lev:
                ax.scatter(rnd_x[inside], rnd_y[inside], sdf[inside], 
                           color='cyan', s=size, alpha=alpha)
                ax.scatter(rnd_x[outside], rnd_y[outside], sdf[outside], 
                           color='coral', s=size, alpha=alpha)
            else:
                len_in = np.abs(RADIUS - dist[inside])
                ax.quiver(rnd_x[inside], rnd_y[inside], rnd_z[inside], 
                          0, 0, len_in, color='cyan')
                len_out = np.abs(dist[outside] - RADIUS)
                ax.quiver(rnd_x[outside], rnd_y[outside], rnd_z[outside], 
                          0, 0, -len_out, color='coral')

    ax.plot(x_circle, y_circle, z_circle, color='red', linewidth=2)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-4, 4)
    ax.view_init(elev=elevation, azim=azimuth)
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    ax.axis('off')


    if arrow and normal and lev:
        save_path = os.path.join(IMAGE_DIR, 'Level_set.svg')
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    elif arrow and normal and not lev:
        save_path = os.path.join(IMAGE_DIR, '3D_circle_normal.svg')
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    elif arrow and not normal and not lev:
        if elevation == 90:
            save_path = os.path.join(IMAGE_DIR, '2D_circle_arrow_upper.svg')
        else:
            save_path = os.path.join(IMAGE_DIR, '2D_circle_arrow.svg')
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    elif not arrow and not normal and not lev:
        save_path = os.path.join(IMAGE_DIR, '2D_circle.svg')
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        
    plt.close(fig)

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
        Description: generate level set data of corresponding z values
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
    # create_3D_parabolic(elevation=20,azimuth=315)
    # create_3D_parabolic_insert(elevation=20,azimuth=315)
    # create_3D_parabolic_insert(elevation=90,azimuth=0)

    # Create 3 image of 3D cubic surface
    # create_cubicsurface(elevation=20, azimuth=225)
    # create_cubicsurface_insert(elevation=20, azimuth=225)
    # create_cubicsurface_insert(elevation=90, azimuth=270)


    level_set_vis(elevation=90, azimuth=70, arrow=False, normal=False, lev=False)
    level_set_vis(elevation=90, azimuth=70, arrow=True, normal=False, lev=False)
    level_set_vis(elevation=25, azimuth=70, arrow=True, normal=False, lev=False)
    level_set_vis(elevation=20, azimuth=70, arrow=True, normal=True, lev=False)
    level_set_vis(elevation=20, azimuth=150, arrow=True, normal=True, lev=True)