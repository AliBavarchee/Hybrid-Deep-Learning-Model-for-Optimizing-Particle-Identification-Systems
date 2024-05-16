#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Descriptor: ccbar Dstarm2D0barPim D0bar2PipPimPi0 PHSP

#############################################################
# Steering file for official MC production of signal samples
#
# October 2020 - Belle II Collaboration
#############################################################

import basf2 as b2
import generators as ge
import simulation as si
import L1trigger as l1
import reconstruction as re
import mdst as mdst
import glob as glob


import sys
if len(sys.argv) != 2:
    sys.exit('\n Usage: basf2 GenerateSignalCcbar.py \'file_number\' \n\n')

file_number = int(sys.argv[1])

#: output filename, can be overriden with -o
output_filename = "./DstSignal_"
output_filename += str(file_number)
output_filename += ".root"


# decay file
decfile = './2610030001.dec'

# create path
main = b2.create_path()

# specify number of events to be generated
main.add_module("EventInfoSetter", expList=1003, runList=0, evtNumList=100000)

# generate ccbar events
ge.add_inclusive_continuum_generator(main, finalstate='ccbar', particles=['D*+'], userdecfile=decfile, include_conjugates=True)

# detector simulation
si.add_simulation(main)

# reconstruction
re.add_reconstruction(main)

# Finally add mdst output
mdst.add_mdst_output(main, filename=output_filename)

# process events and print call statistics
b2.process(main)
print(b2.statistics)
