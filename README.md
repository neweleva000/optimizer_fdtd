# Install Dependencies Ubuntu 
python3 -m venv build\_env
source build\_env/bin/activate
pip install -r requirements.txt

# Run
#Example microstrip -- Parameters to adjust grid dimensions are at the top of the python script
python3 example\_fdtd.py

#Plot EMX
cd emx\_comparison
python3 plot\_touchstone.py results/emx\_comparison.s2p

