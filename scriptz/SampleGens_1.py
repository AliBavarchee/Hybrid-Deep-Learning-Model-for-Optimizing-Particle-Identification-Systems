import pythia8
import ROOT

# Initialize Pythia
pythia = pythia8.Pythia()

# Set the energies and particles for the simulation
beamEnergy = 7000  # Center-of-mass energy in GeV
pythia.readString("Beams:eCM = {}".format(beamEnergy))
pythia.readString("HardQCD:all = on")  # Enable hard QCD processes

# Particle filter for D*+
dstar_filter = pythia8.ParticleFilter(423)  # D*+ has PDG ID 423

# Particle filter for D0
d0_filter = pythia8.ParticleFilter(421)  # D0 has PDG ID 421

# Create ROOT file and tree
output_file = ROOT.TFile("output.root", "RECREATE")
tree = ROOT.TTree("DecayTree", "Particle Decays")

# Variables to store particle information
particle_id = ROOT.std.vector[int]()
particle_pt = ROOT.std.vector[float]()

# Branches for the tree
tree.Branch("ParticleID", particle_id)
tree.Branch("ParticlePT", particle_pt)

# Event loop
for event in pythia(events=100):  # Simulate 100 events
    slow_pions = []
    
    # Loop through particles in the event
    for particle in event:
        # Check if the particle is a D*+
        if dstar_filter.isMatch(particle):
            # Store D*+ information
            particle_id.push_back(particle.id())
            particle_pt.push_back(particle.pT())
        
        # Check if the particle is a D0
        if d0_filter.isMatch(particle):
            # Store D0 information
            particle_id.push_back(particle.id())
            particle_pt.push_back(particle.pT())

        # Check if the particle is a pion from D* decay (momentum < 0.5 GeV/c)
        if particle.id() == 211 and particle.pT() < 0.5 and particle.isFinal():
            slow_pions.append(particle)

        # Check if the particle is a pion from D0 decay (momentum between 0.5 and 5 GeV/c)
        if particle.id() == 211 and 0.5 < particle.pT() < 5 and particle.isFinal():
            slow_pions.append(particle)
    
    # Fill the tree with slow pion information for this event
    tree.Fill()
    
    # Clear the vectors for the next event
    particle_id.clear()
    particle_pt.clear()

# Write the tree and close the ROOT file
output_file.Write()
output_file.Close()
