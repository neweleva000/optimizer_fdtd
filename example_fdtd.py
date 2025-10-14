import pudb
#import fdtd

import sys
sys.path.append('../fdtd')
import fdtd_local as fdtd


import numpy as np
from math import sqrt, ceil, floor
from fdtd.boundaries import DomainBorderPML
from fdtd.boundaries import PML
from fdtd.conversions import simE_to_worldE, simH_to_worldH, const
from matplotlib import pyplot as plt

#import sys
#sys.path.append('../fdtd/fdtd/')
#from fdtd.sources import SoftArbitraryPointSource

def bound_check_avg(field, num_avg):
    avg = 0
    for i in range(num_avg):
        avg += np.abs(field[i])
    return avg/num_avg


def gaussian_pulse(period: float, width: float, num_steps: int, amplitude: float = 1.0) -> np.ndarray:
    """
    Generate a Gaussian pulse centered in the time window.

    Parameters:
    - period (float): Time between peaks (used to center the pulse).
    - width (float): Width (standard deviation) of the Gaussian envelope.
    - num_steps (int): Number of discrete time steps.
    - amplitude (float): Maximum amplitude of the pulse.

    Returns:
    - np.ndarray: 1D array representing the Gaussian pulse over time.
    """
    t = np.arange(num_steps)
    t0 = period / 2.0  # Center the pulse around half the period
    pulse = amplitude * np.exp(-((t - t0) ** 2) / (2 * width ** 2))
    return pulse


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
    #pudb.set_trace()
    
    # Compute time-averaged (real) Poynting vector: (1/2) Re{E × H*}
    return 0.5 * np.real(S_complex)
    
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
        E_f_x = np.fft.rfft(E_t_x[:,i])
        E_f_y = np.fft.rfft(E_t_y[:,i])
        E_f_z = np.fft.rfft(E_t_z[:,i])

        H_f_x = np.fft.rfft(H_t_x[:,i])
        H_f_y = np.fft.rfft(H_t_y[:,i])
        H_f_z = np.fft.rfft(H_t_z[:,i])

        #E_f_x = (E_t_x[:,i])
        #E_f_y = (E_t_y[:,i])
        #E_f_z = (E_t_z[:,i])

        #H_f_x = (H_t_x[:,i])
        #H_f_y = (H_t_y[:,i])
        #H_f_z = (H_t_z[:,i])

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

#TODO scale wavelength up by e6 and make everything in meters?

min_run_time = 1000 #This is where it appears to level off
run_time_step = 100

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
start_vert = 50e-6
air_grid_pt = 10
air_size = grid_sp * air_grid_pt  #Amount of air material between boundaries and conductor in vertical dimension
conductor_thickness = grid_sp
vertical_len = air_size + dielectric_thickness + 2 * conductor_thickness #2 layer stackup
pml_size = 5
microstrip_width = 40e-6
outer_dim_x = outer_dim_x_conductor + 2 * pml_size * grid_sp
outer_dim_y = outer_dim_y_conductor + 2 * pml_size * grid_sp
conductor_start = pml_size * grid_sp
extra_z_pml = 10

grid = fdtd.Grid(
    shape = (ceil(outer_dim_x_conductor/grid_sp) + 2*pml_size,\
            ceil(outer_dim_y_conductor/grid_sp) + 2*pml_size,\
            ceil(vertical_len/grid_sp) + 2*pml_size + extra_z_pml),\
    grid_spacing=grid_sp,
    permittivity=1.0)

#Add air
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
        = fdtd.AbsorbingObject(permittivity=1, conductivity=5.8e8, name="gnd")

#Add dielectric. TODO non zero conductivity
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


#Add source at far left side (Sx1
#grid[outer_dim_x/2 - microstrip_width/2:outer_dim_x/2 + microstrip_width/2,\
#        pml_size,\
#        microstrip_vertical_start:microstrip_vertical_start + conductor_thickness]\
#        = fdtd.LineSource(period = 1/max_freq, name="source", pulse=True, cycle=10, hanning_dt=1e14/max_freq)
grid[outer_dim_x/2 - microstrip_width/2:outer_dim_x/2 + microstrip_width/2,\
        #grid[pml_size:-pml_size,\
        #grid[pml_size:pml_size+20,\
        pml_size,\
        #conductor_start:microstrip_vertical_start + conductor_thickness]\
        microstrip_vertical_start:microstrip_vertical_start + conductor_thickness]\
        = fdtd.LineSource(period = 1/max_freq, name="source", pulse=True, cycle=10, hanning_dt=10.0)#/max_freq)

        #pulse size = t1 = int(2 * pi / (self.frequency * self.hanning_dt / self.cycle)) ; where frequency is in step size

#Add detector at far left side (Port 1)
grid[outer_dim_x/2 - microstrip_width/2:outer_dim_x/2 + microstrip_width/2,\
        pml_size +1,\
        microstrip_vertical_start:microstrip_vertical_start + conductor_thickness]\
        = fdtd.LineDetector(name="port1")

#Add detector at far right side (Port 2)
grid[outer_dim_x/2 - microstrip_width/2:outer_dim_x/2 + microstrip_width/2,\
        -pml_size -3,\
        microstrip_vertical_start:microstrip_vertical_start + conductor_thickness]\
        = fdtd.LineDetector(name="port2")

#Box of size 2
CustomDomainBorder(grid, pml_size, 1e-6, extra_z_pml)  

#Show setup 
grid.visualize(z=0, show=True)
grid.visualize(x=0, show=True)
grid.visualize(y=0, show=True)

#Run while field quantities are above certain threshold 
#Minimum run time
grid.run(total_time=min_run_time)

#Get E and H field from port 2
E_2_t_unitless = np.array(grid.detectors[1].detector_values()["E"])

thresh = 0.005
num_avg = run_time_step // 2
recent_E2 = max(np.abs(E_2_t_unitless[-num_avg:, 0, 0]))
print(recent_E2)
print(np.max(E_2_t_unitless[:,0,0]))
count = 0
while(recent_E2 > thresh * np.max(E_2_t_unitless[:,0,0]) and count < 100):
    count += 1
    grid.run(total_time=run_time_step)
    E_2_t_unitless = np.array(grid.detectors[1].detector_values()["E"])
    recent_E2 = max(np.abs(E_2_t_unitless[-num_avg:, 0, 0]))
    print(recent_E2)
    print(np.max(E_2_t_unitless[:,0,0]))

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

plt.plot(range(len(P2)), P2)
plt.title('P2')
plt.show()

#Fetch field quantities from port 1
E_1_t_unitless = grid.detectors[0].detector_values()["E"]
H_1_t_unitless = grid.detectors[0].detector_values()["H"]

#Normalize port 1 field quantities
E_1_t = np.array([simE_to_worldE(x) for x in E_1_t_unitless])
H_1_t = np.array([simH_to_worldH(x) for x in H_1_t_unitless])

P1 = calc_power(E_1_t, H_1_t,\
        microstrip_width * conductor_thickness)
plt.plot(range(len(P1)), P1)
plt.title('P1')
plt.show()

mag_S21_sq = np.abs(P2 / P1)
plt.plot(range(len(mag_S21_sq)), mag_S21_sq)
plt.title('S21 ish')
plt.show()

#Plot E field at port 1
E_1_t_x = E_1_t[:, 0, 2] #At first spatial step
plt.plot(range(len(E_1_t_x)), E_1_t_x)
plt.title('Port 1 E field.')
plt.show()
