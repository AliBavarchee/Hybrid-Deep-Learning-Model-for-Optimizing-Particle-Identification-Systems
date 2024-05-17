#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"

void Dstr_Sim_Vv() {
    // Initialize ROOT output file and trees
    TFile outputFile("Shabbou_002.root", "RECREATE");
    TTree decayTree("DecayTree", "Particle Decays");
    TTree dstTree("Dst", "Dst Tree");

    // Variables for DecayTree
    int particleID;
    float particlePT;

    // Variables for Dst tree
    int __experiment__, __run__, __event__;
    float DST_E, DST_p, DST_px, DST_py, DST_pz, DST_pt, DST_cosTheta, DST_theta, DST_phi;
    float DST_D0_M, DST_D0_InvM, DST_D0_chiProb;

    // Branches for DecayTree
    decayTree.Branch("ParticleID", &particleID, "ParticleID/I");
    decayTree.Branch("ParticlePT", &particlePT, "ParticlePT/F");

    // Branches for Dst tree
    dstTree.Branch("__experiment__", &__experiment__, "__experiment__/I");
    dstTree.Branch("__run__", &__run__, "__run__/I");
    dstTree.Branch("__event__", &__event__, "__event__/I");
    dstTree.Branch("DST_E", &DST_E, "DST_E/F");
    dstTree.Branch("DST_p", &DST_p, "DST_p/F");
    // Add other branches similarly...

    // Random number generator
    TRandom3 randomGenerator;

    // Simulate decays and fill the trees
    for (int event = 0; event < 100; ++event) { // Simulate 100 events
        // Simulate D* decays producing slow pions and D0
        if (randomGenerator.Uniform() < 0.6) { // Assuming 60% branching ratio for D*+ -> D0 pi+ decay
            // Generate slow pion from D* decay with momentum < 0.5 GeV/c
            particleID = 211; // Pion PDG ID
            particlePT = randomGenerator.Uniform(0, 0.5);
            decayTree.Fill();

            // Generate D0 particle
            particleID = 421; // D0 PDG ID
            particlePT = randomGenerator.Uniform(0, 10); // Momentum for D0 (arbitrary range)
            decayTree.Fill();
        }

        // Simulate D0 decays producing pions and kaons
        if (randomGenerator.Uniform() < 0.4) { // Assuming 40% branching ratio for D0 -> K- pi+ decay
            // Generate pion with momentum between 0.5 and 5 GeV/c
            particleID = 211; // Pion PDG ID
            particlePT = randomGenerator.Uniform(0.5, 5);
            decayTree.Fill();

            // Generate kaon with momentum between 0.5 and 5 GeV/c
            particleID = 321; // Kaon PDG ID
            particlePT = randomGenerator.Uniform(0.5, 5);
            decayTree.Fill();
        }

        // Fill example values for Dst tree
        __experiment__ = event;
        __run__ = event * 10;
        __event__ = event * 100;
        DST_E = randomGenerator.Uniform(1, 10);
        DST_p = randomGenerator.Uniform(0.1, 5);
        DST_px = randomGenerator.Uniform(-2, 2);
        DST_py = randomGenerator.Uniform(-2, 2);
        DST_pz = randomGenerator.Uniform(-2, 2);
        DST_pt = randomGenerator.Uniform(0.1, 5);
        DST_cosTheta = randomGenerator.Uniform(-1, 1);
        DST_theta = std::acos(DST_cosTheta);
        DST_phi = randomGenerator.Uniform(0, 2 * TMath::Pi());
        DST_D0_M = randomGenerator.Uniform(1.8, 1.9);
        DST_D0_InvM = 1.9 - DST_D0_M;
        DST_D0_chiProb = randomGenerator.Uniform(0, 1);

        dstTree.Fill();
    }

    // Write the trees and close the ROOT file
    outputFile.Write();
    outputFile.Close();
    std::cout << "Simulation data saved in output.root" << std::endl;
}
