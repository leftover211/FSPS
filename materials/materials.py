# ======Denpendecies======
# python 3.10.19
# matplotlib 3.10.8
# numpy 2.2.6
# pandas 2.3.3
# ========================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

def _setup_surface() -> Tuple:
    """
        Discription: generate data for 3D parabolic surface
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
    Surface = 20 - X**2 - Y**2
    return (X, Y, Surface)

def _create_plane(domain: List[Union[int, float]], height:Union[int, float], radius:Union[int,float]) -> Tuple:
    """
        Discription: return plane which has normal vector k
        Inputs:
            domain: [domain start, domain end, step]
            height: z postion
            radius: radius of circle
        Outputs: tuple
            [0] plane data
            [1] circle data
    """
    X = np.arange(domain[0],domain[1],domain[2])
    Y = np.arange(domain[0],domain[1],domain[2])
    X,Y = np.meshgrid(X,Y,indexing='ij')
    plane = np.full_like(X, height)
    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    z_circle = np.full_like(theta, height)
    return (X,Y,plane), (x_circle, y_circle, z_circle)

def create_3D_parabolic(elevation: int, azimuth: int) -> None:
    
    surface = _setup_surface()
    fig,ax = plt.subplots(figsize=(8,8),subplot_kw={"projection":"3d"})
    ax.plot_surface(surface[0],surface[1],surface[2],cmap='viridis',alpha=0.9)

    # transparent background
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    ax.set_xticks(range(-6,8,2))
    ax.set_yticks(range(-6,8,2))
    ax.set_zticks(range(0,21,2))
    ax.view_init(elev=elevation, azim=azimuth)
    plt.savefig('3D_parabolic.svg')
    print("Image saved")

def create_3D_parabolic_insert(elevation:int, azimuth: int) -> None:
    
    surface = _setup_surface()
    domain = [-6,6,0.05]
    plane1, circle1 = _create_plane(domain=domain, height=16, radius=2)
    plane2, circle2 = _create_plane(domain=domain, height=11, radius=3)
    plane3, circle3 = _create_plane(domain=domain, height=4, radius=4)

    fig,ax = plt.subplots(figsize=(8,8),subplot_kw={"projection":"3d"})

    ax.plot_surface(surface[0],surface[1],surface[2],cmap='viridis',alpha=0.9)

    if elevation != 90:
        ax.plot_surface(plane1[0],plane1[1],plane1[2],color='black',alpha=0.2)
        ax.plot_surface(plane2[0],plane2[1],plane2[2],color='black',alpha=0.2)
        ax.plot_surface(plane3[0],plane3[1],plane3[2],color='black',alpha=0.2)

    ax.plot(circle1[0], circle1[1], circle1[2], color='red', linewidth=2,zorder=5)
    ax.plot(circle2[0], circle2[1], circle2[2], color='red', linewidth=2,zorder=5)
    ax.plot(circle3[0], circle3[1], circle3[2], color='red', linewidth=2,zorder=5)

    # transparent background
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    ax.set_xticks(range(-6,8,2))
    ax.set_yticks(range(-6,8,2))
    ax.set_zticks(range(0,21,2))
    ax.view_init(elev=elevation, azim=azimuth)
    if elevation == 90:
        ax.axis('off')
        plt.savefig('3D_parabolic_insert_upper.svg')
    else:
        plt.savefig('3D_parabolic_insert.svg')
    print("Image saved")

if __name__ == "__main__":
    create_3D_parabolic(elevation=20,azimuth=315)
    create_3D_parabolic_insert(elevation=20,azimuth=315)
    create_3D_parabolic_insert(elevation=90,azimuth=0)