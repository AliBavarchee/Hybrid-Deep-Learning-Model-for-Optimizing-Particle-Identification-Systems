#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <Rtypes.h>

void plotHistograms_mc_rf()
{
    // Open the root file
    TFile* file = TFile::Open("mc_Dstar_D0_K_pi_mdst.root");
    if (!file || file->IsZombie())
    {
        printf("Error: Failed to open the root file.\n");
        return;
    }

    // Get the tree from the file
    TTree* tree = dynamic_cast<TTree*>(file->Get("Dst;1"));

    // Variables for storing the values
    Double_t var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12;

    // Set the branch addresses
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf211FromSVD", &var1);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf211FromCDC", &var2);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf211FromTOP", &var3);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf211FromARICH", &var4);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf211FromECL", &var5);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf211FromKLM", &var6);


    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf321FromSVD", &var7);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf321FromCDC", &var8);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf321FromTOP", &var9);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf321FromARICH", &var10);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf321FromECL", &var11);
    tree->SetBranchAddress("DST_D0_pi_pidLogLikelihoodOf321FromKLM", &var12);


    // Create histograms
    TH1F* histLR = new TH1F("histLR", "Likelihood Ratios", 100, 0, 1);
    TH1F* histWLR = new TH1F("histWLR", "Weighted Likelihood Ratios", 100, 0, 1);

    // Loop over the entries in the tree
    Int_t nEntries = tree->GetEntries();
    for (Int_t i = 0; i < nEntries; i++)
    {
        tree->GetEntry(i);

        // Calculate the likelihood ratios
        Double_t LR = (var1 + var2 + var3 + var4 + var5 + var6) / (var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12);

        // Apply weights
        Double_t weightSVD = 0.59864897; 
        Double_t weightCDC = 1.0344862;
        Double_t weightTOP = 1.1731281; 
        Double_t weightARICH = 1.2320882;
        Double_t weightECL = 1.0824207; 
        Double_t weightKLM = 1.201249;

        Double_t weight321SVD = 0.6535989; 
        Double_t weight321CDC = 1.219908; 
        Double_t weight321TOP = 0.9566189; 
        Double_t weight321ARICH = 0.7264329;
        Double_t weight321ECL = 1.0372746; 
        Double_t weight321KLM = 0.63789237;

        Double_t WLR = (weightSVD * var1 + weightCDC * var2 + weightTOP * var3 + weightARICH * var4 + weightECL * var5 + weightKLM * var6) /
                       (weightSVD * var1 + weightCDC * var2 + weightTOP * var3 + weightARICH * var4 + weightECL * var5 + weightKLM * var6 +
                        weight321SVD * var7 + weight321CDC * var8 + weight321TOP * var9 + weight321ARICH * var10 + weight321ECL * var11 + weight321KLM * var12);


        // Fill the histograms
        histLR->Fill(LR);
        histWLR->Fill(WLR);
    }

    // Create a canvas
    TCanvas* canvas = new TCanvas("canvas", "Histograms", 800, 600);

    // Set histogram styles
    histLR->SetLineColor(kBlue);
    histWLR->SetLineColor(kRed);

    // Draw histograms on the same canvas
    histLR->Draw();
    histWLR->Draw("same");

    // Add legend
    TLegend* legend = new TLegend(0.68, 0.6, 0.99, 0.76);
    legend->AddEntry(histLR, "Likelihood Ratios", "l");
    legend->AddEntry(histWLR, "Weighted Likelihood Ratios", "l");
    legend->SetTextFont(72);
    legend->SetTextSize(0.027);
    legend->Draw();

    // Save the plot as an image file
    canvas->SaveAs("ANEW_mc_histograms_likelihood_ratios_Deep_rf.root");

    // Clean up
    delete histLR;
    delete histWLR;
    delete legend;
    delete canvas;
    file->Close();
}
