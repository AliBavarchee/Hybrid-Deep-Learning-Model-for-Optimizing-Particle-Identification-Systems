#include <TFile.h>
#include <TTree.h>
#include <TLorentzVector.h>
#include <TRandom3.h>

void GenerateDecayChain(const char* fileName, int numberOfEvents) {
    TFile outputFile(fileName, "RECREATE");
    TTree *tree = new TTree("Dst", "Dst Decay Simulation");

    // Define variables for the branches
    TLorentzVector *slowPion = new TLorentzVector();
    TLorentzVector *pion = new TLorentzVector();
    TLorentzVector *kaon = new TLorentzVector();

    // Set branch addresses
    tree->Branch("SlowPion", &slowPion);
    tree->Branch("Pion", &pion);
    tree->Branch("Kaon", &kaon);

    // Initialize random number generator
    TRandom3 randomGenerator;

    // Simulation loop
    for (int i = 0; i < numberOfEvents; ++i) {
        // Simulate D*+ decay to slow pion and D0
        slowPion->SetPxPyPzE(randomGenerator.Uniform(-0.5, 0.5),
                             randomGenerator.Uniform(-0.5, 0.5),
                             randomGenerator.Uniform(-0.5, 0.5),
                             randomGenerator.Uniform(0.2, 0.3));

        // Simulate D0 decay to pion and kaon
        pion->SetPxPyPzE(randomGenerator.Uniform(-2.5, 2.5),
                        randomGenerator.Uniform(-2.5, 2.5),
                        randomGenerator.Uniform(-2.5, 2.5),
                        randomGenerator.Uniform(0.2, 5.0));

        kaon->SetPxPyPzE(randomGenerator.Uniform(-2.5, 2.5),
                         randomGenerator.Uniform(-2.5, 2.5),
                         randomGenerator.Uniform(-2.5, 2.5),
                         randomGenerator.Uniform(0.2, 5.0));

        // Fill the tree
        tree->Fill();
    }

    // Write the tree to the output file
    tree->Write();

    // Clean up
    outputFile.Close();
}

void DstD0DecaySimulation() {
    const char* fileName = "Dst_D0_decay_sim.root";
    int numberOfEvents = 10000;
    GenerateDecayChain(fileName, numberOfEvents);
}
