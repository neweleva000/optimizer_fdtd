#import fdtd
import sys
sys.path.append('../fdtd')
import fdtd_local as fdtd
import numpy as np


import autograd.numpy as np  # Use autograd's numpy wrapper
from autograd import grad, jacobian

# Setup grid
grid = fdtd.Grid(shape=(10, 10, 10), grid_spacing=1e-3)

def forward(cond):
    grid[:,:,:]  = fdtd.AbsorbingObject(permittivity=1, conductivity=cond, name="gnd")

    # Add a pulse source
    grid[5, 8, 1:5] = fdtd.LineSource(
        period=1e-9,  # 1 GHz
        name="source",
        pulse=True,
        cycle=20,
    )

    # Add a detector
    grid[1, 8, 8] = fdtd.LineDetector(name="detector")

    # Run
    grid.run(total_time=5)

    return np.array(grid.detectors[0].detector_values()["E"])

cond = np.ones((10, 10, 10)) * 5.8e8

#gradient_fn = grad(forward)
gradient_fn = jacobian(forward)

gradient = gradient_fn(cond)

print("Gradient:", gradient)




