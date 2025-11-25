# Denpendecy
# - Python 3.10.19
# 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_3D_parabolic() -> None:
    # prepare data(polar coordinates)
    r = np.linspace(0, 4, 50)
    theta = np.linspace(0, 2*np.pi, 100)
    r, theta = np.meshgrid(r, theta)
    # transforming polar coordiantes to cartesian coordinates
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    Surface = 20 - X**2 - Y**2


    fig,ax = plt.subplots(figsize=(12,12),subplot_kw={"projection":"3d"})
    ax.plot_surface(X,Y,Surface,cmap='viridis',alpha=0.8)

    # transparent background
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.set_xticks(range(-6,8,2))
    ax.set_yticks(range(-6,8,2))
    ax.zaxis.grid(False)
    ax.zaxis.set_visible(False)
    plt.savefig('3D_parabolic.svg')

def create_3D_parabolic_insert() -> None:
    # prepare data(polar coordinates)
    r = np.linspace(0, 4, 50)
    theta = np.linspace(0, 2*np.pi, 100)
    r, theta = np.meshgrid(r, theta)
    # transforming polar coordiantes to cartesian coordinates
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    Surface = 20 - X**2 - Y**2

    # InsertPlane1
    insertX_1 = np.arange(-6,6,0.05)
    insertY_1 = np.arange(-6,6,0.05)
    insertX_1, insertY_1 = np.meshgrid(insertX_1,insertY_1,indexing='ij')
    InsertPlane1 = np.full_like(insertX_1, 16)

    # InsertPlane2
    insertX_2 = np.arange(-6,6,0.05)
    insertY_2 = np.arange(-6,6,0.05)
    insertX_2, insertY_2 = np.meshgrid(insertX_2,insertY_2,indexing='ij')
    InsertPlane2 = np.full_like(insertX_2, 11)

    # InsertPlane3
    insertX_3 = np.arange(-6,6,0.05)
    insertY_3 = np.arange(-6,6,0.05)
    insertX_3, insertY_3 = np.meshgrid(insertX_3,insertY_3,indexing='ij')
    InsertPlane3 = np.full_like(insertX_3, 16)

    fig,ax = plt.subplots(figsize=(12,12),subplot_kw={"projection":"3d"})
    ax.plot_surface(X,Y,Surface,cmap='viridis',alpha=0.8)
    ax.plot_surface(insertX_1,insertY_1,InsertPlane1,color='black',alpha=0.4)
    ax.plot_surface(insertX_1,insertY_1,InsertPlane1,color='black',alpha=0.4)
    ax.plot_surface(insertX_1,insertY_1,InsertPlane1,color='black',alpha=0.4)
    ax.plot_surface(insertX_1,insertY_1,InsertPlane1,color='black',alpha=0.4)
    

    # transparent background
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.set_xticks(range(-6,8,2))
    ax.set_yticks(range(-6,8,2))
    ax.zaxis.grid(False)
    ax.zaxis.set_visible(False)
    plt.savefig('3D_parabolic_insert.svg')