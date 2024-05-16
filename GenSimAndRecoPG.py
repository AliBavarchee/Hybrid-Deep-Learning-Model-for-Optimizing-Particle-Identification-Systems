#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Descriptor: B_tau_nu

#############################################################
# 
# Generate ParticleGun events
#
# A. Gaz, June 2022
#
#############################################################


import basf2 as b2
from simulation import add_simulation
from L1trigger import add_tsim
from reconstruction import add_reconstruction
from ROOT import Belle2
import mdst as mdst

import sys
if len(sys.argv) != 2:
    sys.exit('\n Usage: basf2 GenSimAndRecoPG.py \'file_number\' \n\n')

file_number = int(sys.argv[1])

#: output filename, can be overriden with -o
output_filename = "./PGtest_"
output_filename += str(file_number)
output_filename += ".root"

#: number of events to generate, can be overriden with -n
num_events = 100000
#num_events = 1000

# create path
main = b2.create_path()

# specify number of events to be generated
main.add_module("EventInfoSetter", expList=0, runList=0, evtNumList=num_events)


################################################
# pdg codes:
#
# 11 (-11):     electron (positron)
# 13 (-13):     mu- (mu+)
# 211 (-211):   pi+ (pi-)
# 321 (-321):   K+ (K-)
# 2212 (-2212): proton (anti-proton)
# 1000010020 (-1000010020): deuteron (anti-deuteron)
#
################################################


# generate PG tracks within the TOP acceptance
particlegun = b2.register_module('ParticleGun')
particlegun.param('pdgCodes', [-11])
particlegun.param('nTracks', 1)
particlegun.param('momentumGeneration', 'uniform')
particlegun.param('momentumParams', [.2, 5.])
particlegun.param('thetaGeneration', 'uniformCos')
particlegun.param('thetaParams', [10., 170.])
particlegun.param('phiGeneration', 'uniform')
particlegun.param('phiParams', [-180., 180.])
particlegun.param('vertexGeneration', 'fixed')
particlegun.param('xVertexParams', [0])
particlegun.param('yVertexParams', [0])
particlegun.param('zVertexParams', [0])
main.add_module(particlegun)


# detector simulation
add_simulation(main)

# reconstruction
add_reconstruction(main)

# Finally add mdst output
mdst.add_mdst_output(main, filename=output_filename)

# process events and print call statistics
b2.process(main)
print(statistics)
