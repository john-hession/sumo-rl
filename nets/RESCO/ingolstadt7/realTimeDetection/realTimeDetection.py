import traci
import os
import subprocess
import time

# Set SUMO_HOME environment variable (replace with your SUMO installation path)
os.environ['SUMO_HOME'] = '/opt/homebrew/bin/sumo'

# Define SUMO command
sumo_binary = "sumo"  # Use "sumo-gui" for the GUI version
sumo_config = "ingolstadt7.sumocfg"  # Replace with your config file path
sumo_cmd = [sumo_binary, "-c", sumo_config]

# Start SUMO with TraCI
traci.start(sumo_cmd)

try:
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        # Check for vehicle presence on the induction loop
        vehicle_count = traci.inductionloop.getLastStepVehicleNumber("testSensor")
        vehicle_present = vehicle_count > 0

        # Print the boolean status
        print(f"Vehicle present on testSensor: {vehicle_present}")

    traci.close()
except Exception as e:
    print(f"Error during simulation: {e}")
finally:
    if traci.isConnected():
        traci.close()