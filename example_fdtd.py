import fdtd
import numpy as np
from math import sqrt, ceil, floor
from fdtd.boundaries import DomainBorderPML

fdtd.set_backend("numpy")

#Constants
dielectric_const = 4
WAVELENGTH = (3e8) / (180e9 * sqrt(4))
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light

grid_sp = 0.005 * WAVELENGTH

start_vert = 50e-6
vertical_len = 80e-6 + 2 * start_vert 

grid = fdtd.Grid(
    shape = (ceil(500e-6/grid_sp), ceil(500e-6/grid_sp), ceil(vertical_len/grid_sp)), # 500um x 500um x 80um (grid_spacing) --> 3D FDTD
    grid_spacing=grid_sp,
    permittivity=1,
)

#Add dielectric
grid[5:-5,5:-5,start_vert+5e-6: start_vert + 75e-6] = fdtd.Object(permittivity=4, name="diel")

#Add microstrip line 
grid[230e-6:270e-6, 5:-5, start_vert+75e-6:start_vert +80e-6] = fdtd.AbsorbingObject(permittivity=1, conductivity=5.8e8, name="conductor")

#Add ground plane
grid[5:-5,5:-5,start_vert: start_vert + 5e-6] = fdtd.AbsorbingObject(permittivity=1, conductivity=5.8e8, name="gnd")


#Add source at far left side (Sx1
grid[60, 6, start_vert:start_vert+80e-6] = fdtd.LineSource(
    period = 200e9, name="source"
)

#Add detector at far right side (Port 1)
grid[60, 6, start_vert+75e-6:start_vert +80e-6] = fdtd.LineDetector(name="port1")

#Add detector at far right side (Port 2)
grid[60, -5, start_vert+75e-6:start_vert +80e-6] = fdtd.LineDetector(name="port2")

#Box of size 2
DomainBorderPML(grid, 2)  

grid.run(total_time=500)

grid.visualize(x=0, show=True)
grid.visualize(z=12, show=True)
grid.visualize(y=0, show=True)
