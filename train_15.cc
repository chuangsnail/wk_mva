#include <string>
#include "string.h"
#include <iostream>


using namespace TMVA;

void train_15()
{
    TChain* cor = new TChain("correct");
    TChain* incor = new TChain("incorrect");
    
    cor->Add( "/wk_cms2/cychuang/trained_files/st08_*.root" );
    incor->Add( "/wk_cms2/cychuang/trained_files/st08_*.root" );
    
    TFile* fout = new TFile("train_15.root", "recreate");

    TMVA::Factory* factory = new TMVA::Factory("TMVAClassification",fout,"V:!Silent:Color:Transformations=I:DrawProgressBar:AnalysisType=Classification");
    
    TMVA::DataLoader* dataloader = new TMVA::DataLoader("train_15");
   
   /*	
    dataloader->AddVariable("hadb_Pt",'D');
    dataloader->AddVariable("hadb_Phi",'D');
    dataloader->AddVariable("hadb_Eta",'D');
    dataloader->AddVariable("lepb_Pt",'D');
    dataloader->AddVariable("lepb_Eta",'D');
    dataloader->AddVariable("lepb_Phi",'D');
    dataloader->AddVariable("j1j2_sumPt",'D');
    dataloader->AddVariable("j1j2_absdelEta",'D');
    dataloader->AddVariable("j1j2_absdelPt",'D');
    dataloader->AddVariable("j1j2_delPhi",'D');
    dataloader->AddVariable("lepblep_absdelPt",'D');
    dataloader->AddVariable("lepblep_absdelEta",'D');
    dataloader->AddVariable("lepblep_delPhi",'D');
    dataloader->AddVariable("lepblep_sumPt",'D');
    dataloader->AddVariable("hadb_deepcsv_v",'D');
    dataloader->AddVariable("lepb_deepcsv_v",'D');

	dataloader->AddVariable("hadblepb_delPt",'D');
	dataloader->AddVariable("hadb_probb",'D');
	dataloader->AddVariable("hadb_probbb",'D');
	*/
	dataloader->AddVariable("top_mass",'D');
	dataloader->AddVariable("w_mass",'D');
	//dataloader->AddVariable("top_mass_dev",'D');
	//dataloader->AddVariable("w_mass_dev",'D');
    
    dataloader->AddVariable("j1j2_sumPt",'D');
    dataloader->AddVariable("j1j2_absdelEta",'D');
    dataloader->AddVariable("j1j2_delPhi",'D');

    //dataloader->AddVariable("lepblep_sumPt",'D');
    //dataloader->AddVariable("lepblep_absdelEta",'D');
    //dataloader->AddVariable("lepblep_delPhi",'D');

    dataloader->SetWeightExpression("evt_weight");
    
    dataloader->AddSignalTree(cor,1.);
    dataloader->AddBackgroundTree(incor,1.);
    
    dataloader->PrepareTrainingAndTestTree("","","random");     //prepare testing/training sample randomly
    
    factory->BookMethod(dataloader,TMVA::Types::kBDT,"BDT","NTrees=800:MaxDepth=3");
	factory->BookMethod(dataloader,TMVA::Types::kBDT,"BDTG","!H:!V:NTrees=800:MinNodeSize=2.5%:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=3");
    factory->BookMethod(dataloader,TMVA::Types::kMLP,"MLP","H:!V:NeuronType=sigmoid:VarTransform=N:NCycles=550:HiddenLayers=N,N+5,3:TestRate=5:!UseRegulator");
    
    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();
    
    fout->Write();
    fout->Close();
    
}
