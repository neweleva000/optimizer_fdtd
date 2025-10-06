import fdtd
import numpy as np
from math import sqrt, ceil, floor
from fdtd.boundaries import DomainBorderPML
from fdtd.conversions import simE_to_worldE, simH_to_worldH, const
from matplotlib import pyplot as plt

fdtd.set_backend("numpy")

#TODO scale wavelength up by e6 and make everything in meters?

#Constants
dielectric_const = 4
max_freq = 180e9
WAVELENGTH = (3e8) / (max_freq * sqrt(4))
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light
grid_sp = 0.005 * WAVELENGTH

#Grid definition
outer_dim_x = 500e-6
outer_dim_y = 500e-6
dielectric_thickness = 70e-6
start_vert = 50e-6
air_size = grid_sp * 5  #Amount of air material between boundaries and conductor in vertical dimension
conductor_thickness = 0.01 * WAVELENGTH
vertical_len = 2 * air_size + dielectric_thickness + 2 * conductor_thickness #2 layer stackup
pml_size = 12
microstrip_width = 40e-6

grid = fdtd.Grid(
    shape = (ceil(outer_dim_x/grid_sp), ceil(outer_dim_y/grid_sp), ceil(vertical_len/grid_sp)), 
    grid_spacing=grid_sp,
    permittivity=1.0,
)

#Add dielectric
#grid[5:-5,5:-5,start_vert+5e-6: start_vert + 75e-6] = fdtd.Object(permittivity=dielectric_const, name="dielectric")
grid[pml_size:-pml_size,\
        pml_size:-pml_size,\
        air_size + conductor_thickness: air_size + conductor_thickness + dielectric_thickness]\
        = fdtd.Object(permittivity=dielectric_const, name="dielectric")

#Add microstrip line 
#grid[230e-6:270e-6, 5:-5, start_vert+75e-6:start_vert +80e-6] = fdtd.AbsorbingObject(permittivity=1, conductivity=5.8e8, name="conductor")
microstrip_vertical_start = air_size + dielectric_thickness + conductor_thickness 
grid[outer_dim_x/2 - microstrip_width/2:outer_dim_x/2 + microstrip_width/2,\
        pml_size:-pml_size,\
        microstrip_vertical_start:microstrip_vertical_start + conductor_thickness]\
        = fdtd.AbsorbingObject(permittivity=1, conductivity=5.8e8, name="conductor")

#Add ground plane
grid[pml_size:-pml_size,\
        pml_size:-pml_size,\
        air_size : air_size + conductor_thickness]\
        = fdtd.AbsorbingObject(permittivity=1, conductivity=5.8e8, name="gnd")


#Add source at far left side (Sx1
grid[outer_dim_x / 2\
        , pml_size + 1,\
        air_size:microstrip_vertical_start + conductor_thickness]\
        = fdtd.LineSource(period = 1/max_freq, name="source")

#Add detector at far right side (Port 1)
grid[outer_dim_x/2,\
        pml_size +1,\
        microstrip_vertical_start + conductor_thickness]\
        = fdtd.LineDetector(name="port1")

#Add detector at far right side (Port 2)
grid[outer_dim_x /2,\
        -pml_size -1,\
        microstrip_vertical_start + conductor_thickness]\
        = fdtd.LineDetector(name="port2")

#Box of size 2
DomainBorderPML(grid, pml_size)  

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
