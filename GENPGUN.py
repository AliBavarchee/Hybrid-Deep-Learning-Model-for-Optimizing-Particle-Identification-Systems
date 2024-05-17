#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Descriptor: B_tau_nu

import numpy as np
import pandas as pd
import sys

# Check for correct usage
if len(sys.argv) != 2:
    sys.exit('\nUsage: python3 GenSimAndRecoPG.py \'file_number\'\n')

file_number = int(sys.argv[1])

# Output filename
output_filename = f"./PGtest_{file_number}.csv"

# Number of events to generate
num_events = 100000

# Particle codes (example includes electron)
particle_codes = {
    'electron': 11,
    'positron': -11,
    'muon-': 13,
    'muon+': -13,
    'pi+': 211,
    'pi-': -211,
    'K+': 321,
    'K-': -321,
    'proton': 2212,
    'anti-proton': -2212,
    'deuteron': 1000010020,
    'anti-deuteron': -1000010020
}

# Define particle gun parameters
pdg_code = particle_codes['electron']
n_tracks = 1
momentum_range = [0.2, 5.0]
theta_range = [10.0, 170.0]
phi_range = [-180.0, 180.0]
vertex_position = [0, 0, 0]

# Generate random events
events = []
for _ in range(num_events):
    momentum = np.random.uniform(*momentum_range)
    theta = np.deg2rad(np.random.uniform(*theta_range))
    phi = np.deg2rad(np.random.uniform(*phi_range))

    px = momentum * np.sin(theta) * np.cos(phi)
    py = momentum * np.sin(theta) * np.sin(phi)
    pz = momentum * np.cos(theta)

    event = {
        'pdg_code': pdg_code,
        'px': px,
        'py': py,
        'pz': pz,
        'x': vertex_position[0],
        'y': vertex_position[1],
        'z': vertex_position[2]
    }
    events.append(event)

# Create DataFrame
df = pd.DataFrame(events)

# Save to CSV
df.to_csv(output_filename, index=False)

print(f"Generated {num_events} events and saved to {output_filename}")
