import pudb
#import fdtd #Error on grid placement for source

import sys
sys.path.append('../fdtd')
import fdtd_local as fdtd


import numpy as np
from math import sqrt, ceil, floor
from fdtd.boundaries import DomainBorderPML
from fdtd.boundaries import PML
from fdtd.conversions import simE_to_worldE, simH_to_worldH, const
from matplotlib import pyplot as plt

def CustomDomainBorder(grid, border_cells, stability=1e-8, extra_z=0):
    #top and bottom
    grid[:, : ,0:border_cells] = PML(a=stability)
    grid[:, : ,-border_cells-extra_z:] = PML(a=stability)

    #left and right
    grid[0:border_cells, :, border_cells:-border_cells] = PML(a=stability)
    grid[-border_cells:, :, border_cells:-border_cells] = PML(a=stability)

    #front and back
    grid[border_cells:-border_cells, 0:border_cells, border_cells:-border_cells] = PML(a=stability)
    grid[border_cells:-border_cells, -border_cells:, border_cells:-border_cells] = PML(a=stability)

#Calculates the poynting vector of E and H phasors as:
#S= 1/2 Re{E x H*}
def get_poynting_freq(E_f_x, E_f_y, E_f_z, H_f_x, H_f_y, H_f_z,):
    # Create complex vector fields
    E = np.stack((E_f_x, E_f_y, E_f_z), axis=-1)
    H_conj = np.conj(np.stack((H_f_x, H_f_y, H_f_z), axis=-1))
    
    # Compute complex Poynting vector: E × H*
    S_complex = np.cross(E, H_conj)
    
    # Compute time-averaged (real) Poynting vector: (1/2) Re{E × H*}
    #return np.abs(0.5 * np.real(S_complex))
    return 0.5 * np.abs(S_complex)

#Takes time domain E and H field arrays. Returns spectral power.
#Assuems that fields are sampled evenly across area
#Assumes field arrays are time x spatial x (dimensional X,Y,Z)
def calc_power(E_field_t, H_field_t, Area):
    #TODO could we possibly do this in opposite order (fft last) to save fft cycles?
    spatial_steps = np.shape(E_field_t)[1]
    area_step = Area / spatial_steps

    #Seperate out vectors
    E_t_x = np.array(E_field_t)[:,:,0]
    E_t_y = np.array(E_field_t)[:,:,1]
    E_t_z = np.array(E_field_t)[:,:,2]
                                    
    H_t_x = np.array(H_field_t)[:,:,0]
    H_t_y = np.array(H_field_t)[:,:,1]
    H_t_z = np.array(H_field_t)[:,:,2]

    P = 0
    #Shift to frequency domain and calculate 
    #power via integral of poynting vector
    for i in range(spatial_steps):
        E_f_x = np.fft.rfft(E_t_x[:,i]) / E_t_x.shape[0]
        E_f_y = np.fft.rfft(E_t_y[:,i]) / E_t_y.shape[0]
        E_f_z = np.fft.rfft(E_t_z[:,i]) / E_t_z.shape[0]

        H_f_x = np.fft.rfft(H_t_x[:,i]) / H_t_x.shape[0]
        H_f_y = np.fft.rfft(H_t_y[:,i]) / H_t_y.shape[0]
        H_f_z = np.fft.rfft(H_t_z[:,i]) / H_t_z.shape[0]

        S = get_poynting_freq(E_f_x, E_f_y, E_f_z,\
                H_f_x, H_f_y, H_f_z)

        #TODO does this normal need to be dynamic?
        #Only add the power in the normal direction of the 
        #port area. (In this case y)
        #TODO (negative normal?)
        P += S[:,1] * area_step

    #Return power as a funciton of frequency
    return P

fdtd.set_backend("numpy")

#Constants
dielectric_const = 4
max_freq = 180e9
WAVELENGTH = (3e8) / (max_freq * sqrt(4))
SPEED_LIGHT: float = 299_792_458.0  # [m/s] speed of light
grid_sp = 0.03 * WAVELENGTH

#Grid definition
outer_dim_x_conductor = 500e-6
outer_dim_y_conductor = 500e-6
dielectric_thickness = 70e-6
air_grid_pt = 2 * int(dielectric_thickness / grid_sp)
air_size = grid_sp * air_grid_pt  #Amount of air material between boundaries and conductor in vertical dimension
conductor_thickness = grid_sp
vertical_len = air_size + dielectric_thickness + 2 * conductor_thickness #2 layer stackup
pml_size = 6
microstrip_width = 40e-6
outer_dim_x = outer_dim_x_conductor + 2 * pml_size * grid_sp
outer_dim_y = outer_dim_y_conductor + 2 * pml_size * grid_sp
conductor_start = pml_size * grid_sp
extra_z_pml = 0
horizontal_port_offset = 5 #TODO

#Simulation duration parameters
min_run_time = 5000 
run_time_step = 500
thresh = 0.005 #Level at which E field must drop to end simulation
num_avg = run_time_step
max_iterations = 100000
max_run_calls = (max_iterations - 1000) // run_time_step

grid = fdtd.Grid(
    shape = (ceil(outer_dim_x_conductor/grid_sp) + 2*pml_size,\
            ceil(outer_dim_y_conductor/grid_sp) + 2*pml_size + 2 * horizontal_port_offset,\
            ceil(vertical_len/grid_sp) + 2*pml_size + extra_z_pml),\
    grid_spacing=grid_sp,
    permittivity=1.0)


print(grid)

#Add air TODO -- is this needed? -- only for conducitivity if so. Only needed for non zero
#grid[pml_size:grid.shape[0]//2 - int(microstrip_width / grid_sp),\
#        pml_size:-pml_size,\
#        vertical_len - air_size +(pml_size-1) * grid_sp:vertical_len +pml_size * grid_sp]\
#        = fdtd.AbsorbingObject(permittivity=1, conductivity=0, name="air1")
#
#grid[grid.shape[0]//2 + int(microstrip_width / grid_sp):-pml_size,\
#        pml_size:-pml_size,\
#        vertical_len - air_size +(pml_size-1) * grid_sp:vertical_len +pml_size * grid_sp]\
#        = fdtd.AbsorbingObject(permittivity=1, conductivity=0, name="air2")
#
#grid[outer_dim_x/2 - microstrip_width/2:outer_dim_x/2 + microstrip_width/2,\
#        pml_size:-pml_size,\
#        vertical_len - air_size +pml_size * grid_sp:vertical_len +pml_size * grid_sp]\
#        = fdtd.AbsorbingObject(permittivity=1, conductivity=0, name="air3")

#Add ground plane
grid[pml_size:-pml_size,\
        pml_size:-pml_size,\
        conductor_start : conductor_start + conductor_thickness]\
        = fdtd.AbsorbingObject(permittivity=1, conductivity=1e20, name="gnd")
        #= fdtd.AbsorbingObject(permittivity=1, conductivity=float('inf'), name="gnd")

#Add dielectric. TODO non zero conductivity to model leakage current
grid[pml_size:-pml_size,\
        pml_size:-pml_size,\
        conductor_start + conductor_thickness: conductor_start + conductor_thickness + dielectric_thickness]\
        = fdtd.AbsorbingObject(permittivity=dielectric_const, conductivity=0, name="dielectric")

#Add microstrip line 
microstrip_vertical_start = conductor_start + dielectric_thickness + conductor_thickness 
grid[outer_dim_x/2 - microstrip_width/2:outer_dim_x/2 + microstrip_width/2,\
        pml_size:-pml_size,\
        microstrip_vertical_start:microstrip_vertical_start + conductor_thickness]\
        = fdtd.AbsorbingObject(permittivity=1, conductivity=5.8e8, name="conductor")

#Add source at port 1
#pulse size = t1 = int(2 * pi / (frequency * hanning_dt / cycle)); where frequency is in step size
grid[outer_dim_x/2,\
        pml_size + horizontal_port_offset,\
        microstrip_vertical_start + conductor_thickness + grid_sp]\
        = fdtd.PointSource(period = 3/max_freq, name="source", pulse=True, cycle=10, hanning_dt=10.0)

#Add detector at far left side (Port 1)
grid[outer_dim_x/2 - microstrip_width/2:outer_dim_x/2 + microstrip_width/2,\
        pml_size + horizontal_port_offset,\
        microstrip_vertical_start + conductor_thickness + grid_sp]\
        = fdtd.LineDetector(name="port1")

#Add detector at far right side (Port 2)
grid[outer_dim_x/2 - microstrip_width/2:outer_dim_x/2 + microstrip_width/2,\
        -pml_size - horizontal_port_offset,\
        microstrip_vertical_start + conductor_thickness + grid_sp]\
        = fdtd.LineDetector(name="port2")

#Box of size 2
#CustomDomainBorder(grid, pml_size, 1e-8, extra_z_pml)  
DomainBorderPML(grid, pml_size)

#Show setup 
grid.visualize(z=0, show=True)
grid.visualize(x=0, show=True)
grid.visualize(y=0, show=True)

#Run while field quantities are above certain threshold 
#Minimum run time
grid.run(total_time=min_run_time)

#Get E and H field from port 2
E_2_t_unitless = np.array(grid.detectors[1].detector_values()["E"])

recent_E2 = np.mean(np.abs(E_2_t_unitless[-num_avg:, 0, 0]))
count = 0
while(recent_E2 > thresh * np.max(E_2_t_unitless[:,0,0])\
        and count < max_run_calls):

    #Run simulation additional steps
    grid.run(total_time=run_time_step)

    #Recompute new max
    E_2_t_unitless = np.array(grid.detectors[1].detector_values()["E"])
    recent_E2 = np.mean(np.abs(E_2_t_unitless[-num_avg:, 0, 0]))
    count += 1

#Fetch port 2 H field
H_2_t_unitless = grid.detectors[1].detector_values()["H"]

#Show setup 
grid.visualize(z=0, show=True)
grid.visualize(x=0, show=True)
grid.visualize(y=0, show=True)

#Normalize field quantities
E_2_t = np.array([simE_to_worldE(x) for x in E_2_t_unitless])
H_2_t = np.array([simH_to_worldH(x) for x in H_2_t_unitless])

#Plot E field at port 2
E_2_t_x = E_2_t[:, 0, 0] #At first spatial step
plt.plot(range(len(E_2_t_x)), E_2_t_x)
plt.title('Port 2 E field.')
plt.show()

#Calculate power through port
P2 = calc_power(E_2_t, H_2_t,\
        microstrip_width * conductor_thickness)

#Calculate frequency array
freq_array = np.fft.rfftfreq(2 * len(P2) - 1, grid.time_step) / 1e9

#Plot P2 as a function of frequency
plt.plot(freq_array, P2)
plt.title('Port 2 Power')
plt.xlim((0,250))
plt.show()

#Fetch field quantities from port 1
E_1_t_unitless = grid.detectors[0].detector_values()["E"]
H_1_t_unitless = grid.detectors[0].detector_values()["H"]

#Normalize port 1 field quantities
E_1_t = np.array([simE_to_worldE(x) for x in E_1_t_unitless])
H_1_t = np.array([simH_to_worldH(x) for x in H_1_t_unitless])

P1 = calc_power(E_1_t, H_1_t,\
        microstrip_width * conductor_thickness)
plt.plot(freq_array, P1)
plt.title('Port 1 Power')
plt.xlim((0,250))
plt.show()

mag_S21_sq = (np.sqrt(P2 / P1))
#mag_S21_sq = ((P2 / P1))
plt.plot(freq_array, mag_S21_sq)
plt.title('|S21|')
plt.xlim((0,250))
plt.show()

#Plot E field at port 1
E_1_t_x = E_1_t[:, 0, 2] #At first spatial step
plt.plot(range(len(E_1_t_x)), E_1_t_x)
plt.title('Port 1 E field.')
plt.show()

#Ratio of field quantities
E_1_t_y = E_1_t[:, 0, 1] #At first spatial step
E_2_t_y = E_2_t[:, 0, 1] #At first spatial step

E_1_f_y = np.fft.rfft(E_1_t_y)
E_2_f_y = np.fft.rfft(E_2_t_y)

S21B = E_2_f_y / E_1_f_y
plt.plot(freq_array, np.abs(S21B))
plt.xlim((0,250))
plt.title("Field based S21")
plt.show()

#Derive impedance from normal fields
E_2_t_z = E_2_t[:, 0, 2] 
E_2_f_z = np.fft.rfft(E_2_t_z)
H_2_t_x = H_2_t[:, 0, 0] 
H_2_f_x = np.fft.rfft(H_2_t_x)
Zc = E_2_f_z / H_2_f_x 
plt.plot(freq_array, np.real(Zc))
plt.plot(freq_array, np.imag(Zc))
plt.title('Input impedance as ratio of normal fields.')
plt.show()

#Frequency domain port 2 fields
E_2_t_x = E_2_t[:, 0, 0] 
E_2_f_x = np.fft.rfft(E_2_t_x)
E_2_t_y = E_2_t[:, 0, 1] 
E_2_f_y = np.fft.rfft(E_2_t_y)
E_2_t_z = E_2_t[:, 0, 2] 
E_2_f_z = np.fft.rfft(E_2_t_z)

H_2_t_x = H_2_t[:, 0, 0] 
H_2_f_x = np.fft.rfft(H_2_t_x)
H_2_t_y = H_2_t[:, 0, 1] 
H_2_f_y = np.fft.rfft(H_2_t_y)
H_2_t_z = E_2_t[:, 0, 2] 
H_2_f_z = np.fft.rfft(H_2_t_z)

plt.plot(freq_array, E_2_f_x, label="Ex")
plt.plot(freq_array, E_2_f_y, label="Ey")
plt.plot(freq_array, E_2_f_z, label="Ez")
plt.plot(freq_array, H_2_f_x, label="Hx")
plt.plot(freq_array, H_2_f_y, label="Hy")
plt.plot(freq_array, H_2_f_z, label="Hz")
plt.legend()
plt.xlim((0,250))
plt.show()

#Compute input impedance from field magnitudes
E_2_mag_f = np.sqrt(E_2_f_x **2 + E_2_f_y **2 + E_2_f_z ** 2) 
H_2_mag_f = np.sqrt(H_2_f_x **2 + H_2_f_y **2 + H_2_f_z ** 2) 

Zc2 = E_2_mag_f / H_2_mag_f
plt.plot(freq_array, np.real(Zc2))
plt.plot(freq_array, np.imag(Zc2))
plt.title("Input impedance")
plt.show()

#Scale derived s parameters by input impedance
S21B_50 = S21B * np.sqrt(np.abs(np.real(Zc)) / 50)
plt.plot(freq_array, S21B_50)
plt.xlim((0,250))
plt.show()

S21C_50 = mag_S21_sq * np.sqrt(np.abs(np.real(Zc)) / 50)
plt.plot(freq_array, S21C_50)
plt.xlim((0,250))
plt.title('S21 scaled by field quantity')
plt.show()

S21C_50 = mag_S21_sq * np.sqrt(np.abs(np.real(Zc2)) / 50)
plt.plot(freq_array, S21C_50)
plt.xlim((0,250))
plt.title('S21 scaled by magnitude field quantity')
plt.show()
