import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pythia8
import geant4
import uproot
import sys

if len(sys.argv) != 2:
    sys.exit('\nUsage: python GenerateSignalCcbar.py file_number\n')

file_number = int(sys.argv[1])

# Output filename
output_filename = f"./DstSignal_{file_number}.root"

# Decay file (this is just a placeholder, pythia8 handles decays internally)
decfile = './2610030001.dec'

# Number of events to be generated
num_events = 7500

# Initialize Pythia
pythia = pythia8.Pythia()
pythia.readString("Beams:eCM = 1003.0")
pythia.readString("HardQCD:all = on")
pythia.init()

# Event generation
events = []
for i in range(num_events):
    if not pythia.next(): continue
    event = [(p.id(), p.px(), p.py(), p.pz(), p.e()) for p in pythia.event]
    events.append(event)

# Detector simulation with Geant4 (simplified, real implementation would be more complex)
def simulate_event(event):
    g4_event = geant4.G4Event()
    for particle in event:
        g4_particle = geant4.G4PrimaryParticle(particle[0], particle[1], particle[2], particle[3], particle[4])
        g4_event.add_primary(g4_particle)
    geant4.run_event(g4_event)
    return g4_event

simulated_events = [simulate_event(event) for event in events]

# Reconstruction (placeholder for actual reconstruction code)
def reconstruct_event(event):
    # Placeholder for reconstruction logic
    return event

reconstructed_events = [reconstruct_event(event) for event in simulated_events]

# Save to ROOT file using uproot
with uproot.recreate(output_filename) as f:
    f["Events"] = {"Particles": reconstructed_events}

print(f"Events saved to {output_filename}")
