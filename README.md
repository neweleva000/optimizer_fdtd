# Install Dependencies Ubuntu 
python3 -m venv build\_env <br>
source build\_env/bin/activate <br>
pip install -r requirements.txt <br>

# Run
#Example microstrip -- Parameters to adjust grid dimensions are at the top of the python script
<br>python3 example\_fdtd.py

#Plot EMX
cd emx\_comparison
python3 plot\_touchstone.py results/emx\_comparison.s2p

