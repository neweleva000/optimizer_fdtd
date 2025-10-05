import fdtd
import numpy as np
from math import sqrt, ceil, floor
from fdtd.boundaries import DomainBorderPML
from fdtd.conversions import simE_to_worldE, simH_to_worldH, const
from matplotlib import pyplot as plt

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
grid[60, 6, start_vert:start_vert+79e-6] = fdtd.LineSource(
    period = 100e9, name="source"
)

#Add detector at far right side (Port 1)
grid[60, 6, start_vert+76e-6:start_vert +79e-6] = fdtd.LineDetector(name="port1")

#Add detector at far right side (Port 2)
grid[60, -5, start_vert+76e-6:start_vert +79e-6] = fdtd.LineDetector(name="port2")

#Box of size 2
DomainBorderPML(grid, 2)  

grid.run(total_time=1000)

grid.visualize(z=12, show=True)
grid.visualize(x=0, show=True)
grid.visualize(y=0, show=True)

#print(sum(grid.detectors[1].detector_values()["E"]))
#print(sum(grid.detectors[1].detector_values()["H"]))


E_2_t_unitless = grid.detectors[1].detector_values()["E"]
H_2_t_unitless = grid.detectors[1].detector_values()["H"]

print(sum(E_2_t_unitless))
print(sum(H_2_t_unitless))

E_2_t = [simE_to_worldE(x) for x in E_2_t_unitless]
H_2_t = [simH_to_worldH(x) for x in H_2_t_unitless]

print(sum(E_2_t))
print(sum(H_2_t))

#print(simE_to_worldE(sum(E_2_t_unitless)))
#print(simH_to_worldH(sum(H_2_t_unitless)))

E_2_t_x =[x[0] for x in E_2_t]
plt.plot(range(len(E_2_t_x)), E_2_t_x)
plt.show()

#E_2_t_y =[y[1] for y in E_2_t]
#plt.plot(range(len(E_2_t_y)), E_2_t_y)
#plt.show()
#
#E_2_t_y =[z[2] for z in E_2_t]
#plt.plot(range(len(E_2_t_z)), E_2_t_z)
#plt.show()




E_1_t_unitless = grid.detectors[0].detector_values()["E"]
H_1_t_unitless = grid.detectors[0].detector_values()["H"]

print(sum(E_1_t_unitless))
print(sum(H_1_t_unitless))

E_1_t = [simE_to_worldE(x) for x in E_1_t_unitless]
H_1_t = [simH_to_worldH(x) for x in H_1_t_unitless]

print(sum(E_1_t))
print(sum(H_1_t))

#print(simE_to_worldE(sum(E_2_t_unitless)))
#print(simH_to_worldH(sum(H_2_t_unitless)))

E_1_t_x =[x[0] for x in E_1_t]
plt.plot(range(len(E_1_t_x)), E_1_t_x)
plt.show()
