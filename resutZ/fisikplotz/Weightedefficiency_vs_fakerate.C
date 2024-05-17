#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TH2F.h>
#include <TString.h>

void Weightedefficiency_vs_fakerate() {
    // Open the ROOT file
    TFile* file = new TFile("mc_Dstar_D0_K_pi_mdst02.root");
    if (!file->IsOpen()) {
        cout << "Error: Cannot open the ROOT file." << endl;
        return;
    }

    // Get the tree from the ROOT file
    TTree* tree = (TTree*)file->Get("Dst");
    if (!tree) {
        cout << "Error: Cannot find the tree in the ROOT file." << endl;
        file->Close();
        return;
    }

    // Variables to hold tree branches
    Double_t Ture_K_SVD, Ture_K_CDC, False_K_SVD, False_K_CDC, False_K_TOP, False_pi_SVD, False_pi_CDC, True_pi_SVD, True_pi_CDC;

    // Set the branch addresses
    tree->SetBranchAddress("DST_D0_K_pidLogLikelihoodOf321FromSVD", &Ture_K_SVD);
    tree->SetBranchAddress("DST_D0_K_pidLogLikelihoodOf321FromCDC", &Ture_K_CDC);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf321FromSVD", &False_K_SVD);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf321FromCDC", &False_K_CDC);
    tree->SetBranchAddress("DST_D0_K_pidLogLikelihoodOf211FromSVD", &False_pi_SVD);
    tree->SetBranchAddress("DST_D0_K_pidLogLikelihoodOf211FromCDC", &False_pi_CDC);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf211FromSVD", &True_pi_SVD);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf211FromCDC", &True_pi_CDC);

    // Create histograms to hold efficiency and fake rate
    TH2F* histEfficiencyVsFakeRate = new TH2F("EfficiencyVsFakeRate", "KaonID Efficiency vs Pion Mis-ID Rate;Fake Rate (Pions);Efficiency (Kaons)",
        100, 0, 1, 100, 0, 1);

    // Loop over entries in the tree
    Long64_t nEntries = tree->GetEntries();
    for (Long64_t iEntry = 0; iEntry < nEntries; iEntry++) {
        tree->GetEntry(iEntry);

        // Calculate efficiency and fake rate for Kaons
        Double_t efficiency = (Ture_K_SVD * 0.6535989 + Ture_K_CDC * 1.219908) /
            (Ture_K_SVD * 0.6535989  + Ture_K_CDC * 1.219908 + False_pi_SVD * 0.6535989  + False_pi_CDC * 1.219908);

        Double_t fakeRate = (False_K_SVD * 0.59864897 + False_K_CDC * 1.219908) /
            (False_K_SVD * 0.59864897 + False_K_CDC * 1.219908 + True_pi_SVD * 0.59864897 + True_pi_CDC * 1.0344862);

        // Fill the histogram
        histEfficiencyVsFakeRate->Fill(fakeRate, efficiency);
    }

    // Create a canvas to plot and save the histogram
    TCanvas* canvas = new TCanvas("canvas", "KaonID Efficiency vs Pion Mis-ID Rate", 800, 600);
    canvas->cd();

    // Plot efficiency vs fake rate
    histEfficiencyVsFakeRate->SetTitle("KaonID Efficiency vs Pion Mis-ID Rate;Fake Rate (Pions);Efficiency (Kaons)");
    histEfficiencyVsFakeRate->Draw("colz");

    // Save the canvas as a .C file
    canvas->SaveAs("EfficiencyVsFakeRate_Weighted.root");

    // Close the ROOT file
    file->Close();
}

