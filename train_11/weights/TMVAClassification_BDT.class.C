// Class: ReadBDT
// Automatically generated by MethodBase::MakeClass
//

/* configuration options =====================================================

#GEN -*-*-*-*-*-*-*-*-*-*-*- general info -*-*-*-*-*-*-*-*-*-*-*-

Method         : BDT::BDT
TMVA Release   : 4.2.1         [262657]
ROOT Release   : 6.10/09       [395785]
Creator        : cychuang
Date           : Sun Oct 13 17:04:55 2019
Host           : Linux cmsbuild49.cern.ch 2.6.32-696.10.2.el6.x86_64 #1 SMP Thu Sep 14 16:35:02 CEST 2017 x86_64 x86_64 x86_64 GNU/Linux
Dir            : /wk_cms2/cychuang/CMSSW_9_4_2/src/wk_mva
Training events: 16233843
Analysis type  : [Classification]


#OPT -*-*-*-*-*-*-*-*-*-*-*-*- options -*-*-*-*-*-*-*-*-*-*-*-*-

# Set by User:
NTrees: "800" [Number of trees in the forest]
MaxDepth: "3" [Max depth of the decision tree allowed]
# Default:
V: "False" [Verbose output (short form of "VerbosityLevel" below - overrides the latter one)]
VerbosityLevel: "Default" [Verbosity level]
VarTransform: "None" [List of variable transformations performed before training, e.g., "D_Background,P_Signal,G,N_AllClasses" for: "Decorrelation, PCA-transformation, Gaussianisation, Normalisation, each for the given class of events ('AllClasses' denotes all events of all classes, if no class indication is given, 'All' is assumed)"]
H: "False" [Print method-specific help message]
CreateMVAPdfs: "False" [Create PDFs for classifier outputs (signal and background)]
IgnoreNegWeightsInTraining: "False" [Events with negative weights are ignored in the training (but are included for testing and performance evaluation)]
MinNodeSize: "5%" [Minimum percentage of training events required in a leaf node (default: Classification: 5%, Regression: 0.2%)]
nCuts: "20" [Number of grid points in variable range used in finding optimal cut in node splitting]
BoostType: "AdaBoost" [Boosting type for the trees in the forest (note: AdaCost is still experimental)]
AdaBoostR2Loss: "quadratic" [Type of Loss function in AdaBoostR2]
UseBaggedBoost: "False" [Use only a random subsample of all events for growing the trees in each boost iteration.]
Shrinkage: "1.000000e+00" [Learning rate for GradBoost algorithm]
AdaBoostBeta: "5.000000e-01" [Learning rate  for AdaBoost algorithm]
UseRandomisedTrees: "False" [Determine at each node splitting the cut variable only as the best out of a random subset of variables (like in RandomForests)]
UseNvars: "2" [Size of the subset of variables used with RandomisedTree option]
UsePoissonNvars: "True" [Interpret "UseNvars" not as fixed number but as mean of a Poisson distribution in each split with RandomisedTree option]
BaggedSampleFraction: "6.000000e-01" [Relative size of bagged event sample to original size of the data sample (used whenever bagging is used (i.e. UseBaggedBoost, Bagging,)]
UseYesNoLeaf: "True" [Use Sig or Bkg categories, or the purity=S/(S+B) as classification of the leaf node -> Real-AdaBoost]
NegWeightTreatment: "inverseboostnegweights" [How to treat events with negative weights in the BDT training (particular the boosting) : IgnoreInTraining;  Boost With inverse boostweight; Pair events with negative and positive weights in training sample and *annihilate* them (experimental!)]
Css: "1.000000e+00" [AdaCost: cost of true signal selected signal]
Cts_sb: "1.000000e+00" [AdaCost: cost of true signal selected bkg]
Ctb_ss: "1.000000e+00" [AdaCost: cost of true bkg    selected signal]
Cbb: "1.000000e+00" [AdaCost: cost of true bkg    selected bkg ]
NodePurityLimit: "5.000000e-01" [In boosting/pruning, nodes with purity > NodePurityLimit are signal; background otherwise.]
SeparationType: "giniindex" [Separation criterion for node splitting]
RegressionLossFunctionBDTG: "huber" [Loss function for BDTG regression.]
HuberQuantile: "7.000000e-01" [In the Huber loss function this is the quantile that separates the core from the tails in the residuals distribution.]
DoBoostMonitor: "False" [Create control plot with ROC integral vs tree number]
UseFisherCuts: "False" [Use multivariate splits using the Fisher criterion]
MinLinCorrForFisher: "8.000000e-01" [The minimum linear correlation between two variables demanded for use in Fisher criterion in node splitting]
UseExclusiveVars: "False" [Variables already used in fisher criterion are not anymore analysed individually for node splitting]
DoPreselection: "False" [and and apply automatic pre-selection for 100% efficient signal (bkg) cuts prior to training]
SigToBkgFraction: "1.000000e+00" [Sig to Bkg ratio used in Training (similar to NodePurityLimit, which cannot be used in real adaboost]
PruneMethod: "nopruning" [Note: for BDTs use small trees (e.g.MaxDepth=3) and NoPruning:  Pruning: Method used for pruning (removal) of statistically insignificant branches ]
PruneStrength: "0.000000e+00" [Pruning strength]
PruningValFraction: "5.000000e-01" [Fraction of events to use for optimizing automatic pruning.]
SkipNormalization: "False" [Skip normalization at initialization, to keep expectation value of BDT output according to the fraction of events]
nEventsMin: "0" [deprecated: Use MinNodeSize (in % of training events) instead]
UseBaggedGrad: "False" [deprecated: Use *UseBaggedBoost* instead:  Use only a random subsample of all events for growing the trees in each iteration.]
GradBaggingFraction: "6.000000e-01" [deprecated: Use *BaggedSampleFraction* instead: Defines the fraction of events to be used in each iteration, e.g. when UseBaggedGrad=kTRUE. ]
UseNTrainEvents: "0" [deprecated: Use *BaggedSampleFraction* instead: Number of randomly picked training events used in randomised (and bagged) trees]
NNodesMax: "0" [deprecated: Use MaxDepth instead to limit the tree size]
##


#VAR -*-*-*-*-*-*-*-*-*-*-*-* variables *-*-*-*-*-*-*-*-*-*-*-*-

NVar 2
top_mass                      top_mass                      top_mass                      top_mass                                                        'D'    [30.9399604797,6283.00878906]
w_mass                        w_mass                        w_mass                        w_mass                                                          'D'    [13.6341571808,4887.98876953]
NSpec 0


============================================================================ */

#include <vector>
#include <cmath>
#include <string>
#include <iostream>

#define NN new BDTNode
   
#ifndef BDTNode__def
#define BDTNode__def
   
class BDTNode {
   
public:
   
   // constructor of an essentially "empty" node floating in space
   BDTNode ( BDTNode* left,BDTNode* right,
                          int selector, double cutValue, bool cutType, 
                          int nodeType, double purity, double response ) :
   fLeft         ( left         ),
   fRight        ( right        ),
   fSelector     ( selector     ),
   fCutValue     ( cutValue     ),
   fCutType      ( cutType      ),
   fNodeType     ( nodeType     ),
   fPurity       ( purity       ),
   fResponse     ( response     ){
   }

   virtual ~BDTNode();

   // test event if it descends the tree at this node to the right
   virtual bool GoesRight( const std::vector<double>& inputValues ) const;
   BDTNode* GetRight( void )  {return fRight; };

   // test event if it descends the tree at this node to the left 
   virtual bool GoesLeft ( const std::vector<double>& inputValues ) const;
   BDTNode* GetLeft( void ) { return fLeft; };   

   // return  S/(S+B) (purity) at this node (from  training)

   double GetPurity( void ) const { return fPurity; } 
   // return the node type
   int    GetNodeType( void ) const { return fNodeType; }
   double GetResponse(void) const {return fResponse;}

private:

   BDTNode*   fLeft;     // pointer to the left daughter node
   BDTNode*   fRight;    // pointer to the right daughter node
   int                     fSelector; // index of variable used in node selection (decision tree)   
   double                  fCutValue; // cut value applied on this node to discriminate bkg against sig
   bool                    fCutType;  // true: if event variable > cutValue ==> signal , false otherwise
   int                     fNodeType; // Type of node: -1 == Bkg-leaf, 1 == Signal-leaf, 0 = internal 
   double                  fPurity;   // Purity of node from training
   double                  fResponse; // Regression response value of node
}; 
   
//_______________________________________________________________________
   BDTNode::~BDTNode()
{
   if (fLeft  != NULL) delete fLeft;
   if (fRight != NULL) delete fRight;
}; 
   
//_______________________________________________________________________
bool BDTNode::GoesRight( const std::vector<double>& inputValues ) const
{
   // test event if it descends the tree at this node to the right
   bool result;
     result = (inputValues[fSelector] > fCutValue );
   if (fCutType == true) return result; //the cuts are selecting Signal ;
   else return !result;
}
   
//_______________________________________________________________________
bool BDTNode::GoesLeft( const std::vector<double>& inputValues ) const
{
   // test event if it descends the tree at this node to the left
   if (!this->GoesRight(inputValues)) return true;
   else return false;
}
   
#endif
   
#ifndef IClassifierReader__def
#define IClassifierReader__def

class IClassifierReader {

 public:

   // constructor
   IClassifierReader() : fStatusIsClean( true ) {}
   virtual ~IClassifierReader() {}

   // return classifier response
   virtual double GetMvaValue( const std::vector<double>& inputValues ) const = 0;

   // returns classifier status
   bool IsStatusClean() const { return fStatusIsClean; }

 protected:

   bool fStatusIsClean;
};

#endif

class ReadBDT : public IClassifierReader {

 public:

   // constructor
   ReadBDT( std::vector<std::string>& theInputVars ) 
      : IClassifierReader(),
        fClassName( "ReadBDT" ),
        fNvars( 2 ),
        fIsNormalised( false )
   {      
      // the training input variables
      const char* inputVars[] = { "top_mass", "w_mass" };

      // sanity checks
      if (theInputVars.size() <= 0) {
         std::cout << "Problem in class \"" << fClassName << "\": empty input vector" << std::endl;
         fStatusIsClean = false;
      }

      if (theInputVars.size() != fNvars) {
         std::cout << "Problem in class \"" << fClassName << "\": mismatch in number of input values: "
                   << theInputVars.size() << " != " << fNvars << std::endl;
         fStatusIsClean = false;
      }

      // validate input variables
      for (size_t ivar = 0; ivar < theInputVars.size(); ivar++) {
         if (theInputVars[ivar] != inputVars[ivar]) {
            std::cout << "Problem in class \"" << fClassName << "\": mismatch in input variable names" << std::endl
                      << " for variable [" << ivar << "]: " << theInputVars[ivar].c_str() << " != " << inputVars[ivar] << std::endl;
            fStatusIsClean = false;
         }
      }

      // initialize min and max vectors (for normalisation)
      fVmin[0] = 0;
      fVmax[0] = 0;
      fVmin[1] = 0;
      fVmax[1] = 0;

      // initialize input variable types
      fType[0] = 'D';
      fType[1] = 'D';

      // initialize constants
      Initialize();

   }

   // destructor
   virtual ~ReadBDT() {
      Clear(); // method-specific
   }

   // the classifier response
   // "inputValues" is a vector of input values in the same order as the 
   // variables given to the constructor
   double GetMvaValue( const std::vector<double>& inputValues ) const;

 private:

   // method-specific destructor
   void Clear();

   // common member variables
   const char* fClassName;

   const size_t fNvars;
   size_t GetNvar()           const { return fNvars; }
   char   GetType( int ivar ) const { return fType[ivar]; }

   // normalisation of input variables
   const bool fIsNormalised;
   bool IsNormalised() const { return fIsNormalised; }
   double fVmin[2];
   double fVmax[2];
   double NormVariable( double x, double xmin, double xmax ) const {
      // normalise to output range: [-1, 1]
      return 2*(x - xmin)/(xmax - xmin) - 1.0;
   }

   // type of input variable: 'F' or 'I'
   char   fType[2];

   // initialize internal variables
   void Initialize();
   double GetMvaValue__( const std::vector<double>& inputValues ) const;

   // private members (method specific)
   std::vector<BDTNode*> fForest;       // i.e. root nodes of decision trees
   std::vector<double>                fBoostWeights; // the weights applied in the individual boosts
};

double ReadBDT::GetMvaValue__( const std::vector<double>& inputValues ) const
{
   double myMVA = 0;
   double norm  = 0;
   for (unsigned int itree=0; itree<fForest.size(); itree++){
      BDTNode *current = fForest[itree];
      while (current->GetNodeType() == 0) { //intermediate node
         if (current->GoesRight(inputValues)) current=(BDTNode*)current->GetRight();
         else current=(BDTNode*)current->GetLeft();
      }
      myMVA += fBoostWeights[itree] *  current->GetNodeType();
      norm  += fBoostWeights[itree];
   }
   return myMVA /= norm;
};

void ReadBDT::Initialize()
{
  // itree = 0
  fBoostWeights.push_back(0.715891656020403);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.778206,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.434431,-99) , 
1, 96.9213, 1, 0, 0.736184,-99) , 
NN(
0, 
0, 
-1, 219.292, 1, -1, 0.159311,-99) , 
0, 201.064, 1, 0, 0.58944,-99) , 
NN(
0, 
0, 
-1, 246.515, 0, -1, 0.0178383,-99) , 
0, 328.658, 1, 0, 0.5,-99)    );
  // itree = 1
  fBoostWeights.push_back(0.536956);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.673817,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.193341,-99) , 
1, 58.0844, 0, 0, 0.625313,-99) , 
NN(
0, 
0, 
-1, 242.245, 1, -1, 0.174819,-99) , 
0, 215.241, 1, 0, 0.544727,-99) , 
NN(
0, 
0, 
-1, 246.515, 0, -1, 0.0358659,-99) , 
0, 328.658, 1, 0, 0.477272,-99)    );
  // itree = 2
  fBoostWeights.push_back(0.351134);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.582077,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.279946,-99) , 
0, 144.356, 0, 0, 0.527421,-99) , 
NN(
0, 
0, 
-1, 141.689, 1, -1, 0.225124,-99) , 
1, 113.233, 1, 0, 0.479957,-99) , 
NN(
0, 
0, 
-1, 246.515, 0, -1, 0.0598462,-99) , 
0, 328.658, 1, 0, 0.431614,-99)    );
  // itree = 3
  fBoostWeights.push_back(0.188541);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.406831,-99)    );
  // itree = 4
  fBoostWeights.push_back(0.229584);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.544162,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.26431,-99) , 
0, 132.204, 0, 0, 0.520926,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.217976,-99) , 
0, 243.595, 1, 0, 0.491295,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.0984545,-99) , 
0, 328.658, 1, 0, 0.453159,-99)    );
  // itree = 5
  fBoostWeights.push_back(0.155396);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.422922,-99)    );
  // itree = 6
  fBoostWeights.push_back(0.191365);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.53955,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.35185,-99) , 
1, 61.8198, 0, 0, 0.516548,-99) , 
NN(
0, 
0, 
-1, 267.224, 1, -1, 0.309243,-99) , 
0, 229.418, 1, 0, 0.491784,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.138296,-99) , 
0, 328.658, 1, 0, 0.461317,-99)    );
  // itree = 7
  fBoostWeights.push_back(0.119398);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.440583,-99)    );
  // itree = 8
  fBoostWeights.push_back(0.151669);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.534009,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.416139,-99) , 
1, 67.8374, 0, 0, 0.51111,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.349467,-99) , 
1, 127.461, 1, 0, 0.495154,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.179647,-99) , 
0, 328.658, 1, 0, 0.470235,-99)    );
  // itree = 9
  fBoostWeights.push_back(0.15864);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.53683,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.393252,-99) , 
1, 90.0514, 1, 0, 0.507136,-99) , 
NN(
0, 
0, 
-1, 85.3861, 0, -1, 0.421022,-99) , 
0, 186.887, 1, 0, 0.477845,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.203093,-99) , 
0, 328.658, 1, 0, 0.456961,-99)    );
  // itree = 10
  fBoostWeights.push_back(0.0847164);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.457743,-99)    );
  // itree = 11
  fBoostWeights.push_back(0.096662);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.516595,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.378464,-99) , 
0, 262.498, 1, 0, 0.508154,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.336123,-99) , 
0, 130.179, 0, 0, 0.496957,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.245323,-99) , 
0, 328.658, 1, 0, 0.478858,-99)    );
  // itree = 12
  fBoostWeights.push_back(0.0731588);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.463486,-99)    );
  // itree = 13
  fBoostWeights.push_back(0.081152);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.513564,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.374093,-99) , 
1, 56.8756, 0, 0, 0.505226,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.375013,-99) , 
0, 130.179, 0, 0, 0.496934,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.278103,-99) , 
0, 328.658, 1, 0, 0.481736,-99)    );
  // itree = 14
  fBoostWeights.push_back(0.062724);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.468679,-99)    );
  // itree = 15
  fBoostWeights.push_back(0.0608767);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.508962,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.40835,-99) , 
1, 56.8756, 0, 0, 0.503034,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.409297,-99) , 
0, 130.179, 0, 0, 0.497145,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.307887,-99) , 
0, 328.658, 1, 0, 0.484337,-99)    );
  // itree = 16
  fBoostWeights.push_back(0.0510897);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.474477,-99)    );
  // itree = 17
  fBoostWeights.push_back(0.0634333);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.511676,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.400598,-99) , 
1, 122.512, 1, 0, 0.50343,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.421397,-99) , 
0, 257.772, 1, 0, 0.498277,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.332245,-99) , 
0, 328.658, 1, 0, 0.48724,-99)    );
  // itree = 18
  fBoostWeights.push_back(0.13568);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.526732,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.312778,-99) , 
1, 71.073, 0, 0, 0.49958,-99) , 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.542159,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.320246,-99) , 
1, 76.5921, 1, 0, 0.454694,-99) , 
0, 158.533, 0, 0, 0.486601,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.346465,-99) , 
0, 328.658, 1, 0, 0.477365,-99)    );
  // itree = 19
  fBoostWeights.push_back(0.0791413);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.460512,-99)    );
  // itree = 20
  fBoostWeights.push_back(0.0457545);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.50242,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.457728,-99) , 
0, 196.339, 1, 0, 0.490353,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.42101,-99) , 
0, 130.179, 0, 0, 0.486025,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.396567,-99) , 
0, 328.658, 1, 0, 0.480237,-99)    );
  // itree = 21
  fBoostWeights.push_back(0.0840415);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.509701,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.3547,-99) , 
1, 71.073, 0, 0, 0.490583,-99) , 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.523053,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.360512,-99) , 
1, 76.5921, 1, 0, 0.460009,-99) , 
0, 158.533, 0, 0, 0.481792,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.407566,-99) , 
0, 328.658, 1, 0, 0.477007,-99)    );
  // itree = 22
  fBoostWeights.push_back(0.0672443);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.466428,-99)    );
  // itree = 23
  fBoostWeights.push_back(0.0336052);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.483204,-99)    );
  // itree = 24
  fBoostWeights.push_back(0.024619);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502403,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.437357,-99) , 
0, 229.418, 1, 0, 0.497909,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.442848,-99) , 
1, 141.689, 1, 0, 0.49425,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.452854,-99) , 
0, 328.658, 1, 0, 0.491603,-99)    );
  // itree = 25
  fBoostWeights.push_back(0.0245238);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.487741,-99)    );
  // itree = 26
  fBoostWeights.push_back(0.020143);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502387,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.449427,-99) , 
0, 262.498, 1, 0, 0.499215,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.444672,-99) , 
0, 130.179, 0, 0, 0.495837,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.465056,-99) , 
0, 328.658, 1, 0, 0.493872,-99)    );
  // itree = 27
  fBoostWeights.push_back(0.0188213);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.49059,-99)    );
  // itree = 28
  fBoostWeights.push_back(0.017664);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502527,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.45916,-99) , 
1, 142.06, 1, 0, 0.499489,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.454314,-99) , 
0, 130.179, 0, 0, 0.496696,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.474762,-99) , 
0, 328.658, 1, 0, 0.495297,-99)    );
  // itree = 29
  fBoostWeights.push_back(0.0150072);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 1, -1, 0.492497,-99)    );
  // itree = 30
  fBoostWeights.push_back(0.0145535);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502128,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.455092,-99) , 
1, 56.8756, 0, 0, 0.499444,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.462426,-99) , 
0, 130.179, 0, 0, 0.497158,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.482915,-99) , 
0, 328.658, 1, 0, 0.49625,-99)    );
  // itree = 31
  fBoostWeights.push_back(0.0872366);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.518869,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.427895,-99) , 
1, 90.0514, 1, 0, 0.500294,-99) , 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.5389,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.304556,-99) , 
1, 85.3861, 0, 0, 0.482782,-99) , 
0, 186.887, 1, 0, 0.494357,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.48655,-99) , 
0, 328.658, 1, 0, 0.493859,-99)    );
  // itree = 32
  fBoostWeights.push_back(0.0569308);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.508355,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502631,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.388641,-99) , 
1, 71.073, 0, 0, 0.488864,-99) , 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.53347,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.410909,-99) , 
1, 70.2963, 1, 0, 0.464616,-99) , 
0, 158.533, 0, 0, 0.481909,-99) , 
0, 328.658, 0, 0, 0.483599,-99)    );
  // itree = 33
  fBoostWeights.push_back(0.0482207);
  fForest.push_back( 
NN(
0, 
0, 
-1, 328.658, 0, -1, 0.475908,-99)    );
  // itree = 34
  fBoostWeights.push_back(0.0256514);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.506045,-99) , 
NN(
0, 
0, 
-1, 141.689, 1, -1, 0.486713,-99) , 
0, 328.658, 0, 0, 0.487951,-99)    );
  // itree = 35
  fBoostWeights.push_back(0.0131224);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.499577,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.500061,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.44662,-99) , 
0, 229.418, 1, 0, 0.4964,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.447433,-99) , 
1, 141.689, 1, 0, 0.493126,-99) , 
0, 328.658, 0, 0, 0.493539,-99)    );
  // itree = 36
  fBoostWeights.push_back(0.0177721);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502858,-99) , 
NN(
0, 
0, 
-1, 141.689, 1, -1, 0.490702,-99) , 
0, 328.658, 0, 0, 0.491481,-99)    );
  // itree = 37
  fBoostWeights.push_back(0.0132825);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.498378,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.501225,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.454268,-99) , 
0, 229.418, 1, 0, 0.498012,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.455083,-99) , 
1, 141.689, 1, 0, 0.495146,-99) , 
0, 328.658, 0, 0, 0.495353,-99)    );
  // itree = 38
  fBoostWeights.push_back(0.0138981);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.501699,-99) , 
NN(
0, 
0, 
-1, 141.689, 1, -1, 0.492692,-99) , 
0, 328.658, 0, 0, 0.493269,-99)    );
  // itree = 39
  fBoostWeights.push_back(0.0162497);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.498197,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502908,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.469868,-99) , 
1, 105.102, 1, 0, 0.498621,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.461831,-99) , 
1, 141.689, 1, 0, 0.496167,-99) , 
0, 328.658, 0, 0, 0.496297,-99)    );
  // itree = 40
  fBoostWeights.push_back(0.019114);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502259,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.50307,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.48359,-99) , 
0, 186.887, 1, 0, 0.496051,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.45685,-99) , 
0, 130.179, 0, 0, 0.493629,-99) , 
0, 328.658, 0, 0, 0.494182,-99)    );
  // itree = 41
  fBoostWeights.push_back(0.04655);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.497444,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.509038,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.416614,-99) , 
1, 71.073, 0, 0, 0.498,-99) , 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.516872,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.418297,-99) , 
1, 76.5921, 1, 0, 0.479385,-99) , 
0, 158.533, 0, 0, 0.492671,-99) , 
0, 328.658, 0, 0, 0.492977,-99)    );
  // itree = 42
  fBoostWeights.push_back(0.0282474);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.50908,-99) , 
NN(
0, 
0, 
-1, 130.179, 0, -1, 0.485534,-99) , 
0, 328.658, 0, 0, 0.487045,-99)    );
  // itree = 43
  fBoostWeights.push_back(0.014113);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.501967,-99) , 
NN(
0, 
0, 
-1, 130.179, 0, -1, 0.492595,-99) , 
0, 328.658, 0, 0, 0.493197,-99)    );
  // itree = 44
  fBoostWeights.push_back(0.00865017);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.498414,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.500365,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.469045,-99) , 
1, 142.06, 1, 0, 0.498159,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.465121,-99) , 
0, 130.179, 0, 0, 0.496124,-99) , 
0, 328.658, 0, 0, 0.496271,-99)    );
  // itree = 45
  fBoostWeights.push_back(0.0583501);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.500577,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.589554,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.395098,-99) , 
1, 113.766, 0, 0, 0.511924,-99) , 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.505732,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.297366,-99) , 
1, 111.425, 1, 0, 0.49134,-99) , 
0, 215.241, 0, 0, 0.494513,-99) , 
0, 328.658, 0, 0, 0.494902,-99)    );
  // itree = 46
  fBoostWeights.push_back(0.0589819);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.485897,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.575346,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.409125,-99) , 
1, 113.766, 0, 0, 0.509025,-99) , 
NN(
0, 
0, 
-1, 111.425, 1, -1, 0.478747,-99) , 
0, 215.241, 0, 0, 0.483397,-99) , 
0, 328.658, 0, 0, 0.483557,-99)    );
  // itree = 47
  fBoostWeights.push_back(0.0258476);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.500639,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.518482,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.491371,-99) , 
1, 92.9063, 0, 0, 0.49858,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.450818,-99) , 
1, 141.689, 1, 0, 0.495411,-99) , 
0, 328.658, 0, 0, 0.495748,-99)    );
  // itree = 48
  fBoostWeights.push_back(0.0212161);
  fForest.push_back( 
NN(
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.505899,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.472563,-99) , 
1, 105.102, 1, 0, 0.501607,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.457225,-99) , 
1, 141.689, 1, 0, 0.498665,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.494139,-99) , 
0, 328.658, 1, 0, 0.498373,-99)    );
  // itree = 49
  fBoostWeights.push_back(0.0181605);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.499442,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.503847,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.483486,-99) , 
1, 74.6128, 0, 0, 0.497668,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.462495,-99) , 
1, 141.689, 1, 0, 0.495338,-99) , 
0, 328.658, 0, 0, 0.495602,-99)    );
  // itree = 50
  fBoostWeights.push_back(0.0117944);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.503982,-99) , 
NN(
0, 
0, 
-1, 141.689, 1, -1, 0.493971,-99) , 
0, 328.658, 0, 0, 0.494616,-99)    );
  // itree = 51
  fBoostWeights.push_back(0.0124911);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.501017,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502167,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.476265,-99) , 
1, 105.102, 1, 0, 0.498832,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.469949,-99) , 
1, 141.689, 1, 0, 0.49692,-99) , 
0, 328.658, 0, 0, 0.497184,-99)    );
  // itree = 52
  fBoostWeights.push_back(0.0464982);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.497876,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.574624,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.419492,-99) , 
1, 113.766, 0, 0, 0.512775,-99) , 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.503726,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.325642,-99) , 
1, 111.425, 1, 0, 0.491735,-99) , 
0, 215.241, 0, 0, 0.494961,-99) , 
0, 328.658, 0, 0, 0.495149,-99)    );
  // itree = 53
  fBoostWeights.push_back(0.0490932);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.5095,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.563209,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.430856,-99) , 
1, 113.766, 0, 0, 0.510451,-99) , 
NN(
0, 
0, 
-1, 111.425, 1, -1, 0.481664,-99) , 
0, 215.241, 0, 0, 0.486068,-99) , 
0, 328.658, 0, 0, 0.487579,-99)    );
  // itree = 54
  fBoostWeights.push_back(0.0207593);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.497162,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.514099,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.493444,-99) , 
1, 92.9063, 0, 0, 0.498914,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.455873,-99) , 
1, 141.689, 1, 0, 0.496078,-99) , 
0, 328.658, 0, 0, 0.496148,-99)    );
  // itree = 55
  fBoostWeights.push_back(0.0170567);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502352,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.504697,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.478567,-99) , 
1, 105.102, 1, 0, 0.501356,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.461027,-99) , 
1, 141.689, 1, 0, 0.4987,-99) , 
0, 328.658, 0, 0, 0.498936,-99)    );
  // itree = 56
  fBoostWeights.push_back(0.00785436);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.498067,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.500046,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.471959,-99) , 
0, 130.179, 0, 0, 0.498182,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.465268,-99) , 
1, 141.689, 1, 0, 0.496015,-99) , 
0, 328.658, 0, 0, 0.496148,-99)    );
  // itree = 57
  fBoostWeights.push_back(0.010194);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.50003,-99) , 
NN(
0, 
0, 
-1, 141.689, 1, -1, 0.494553,-99) , 
0, 328.658, 0, 0, 0.494907,-99)    );
  // itree = 58
  fBoostWeights.push_back(0.00871383);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.497469,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.500909,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.472735,-99) , 
0, 229.418, 1, 0, 0.499028,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.469761,-99) , 
1, 141.689, 1, 0, 0.497102,-99) , 
0, 328.658, 0, 0, 0.497126,-99)    );
  // itree = 59
  fBoostWeights.push_back(0.0368527);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.499648,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.561168,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.442183,-99) , 
1, 113.766, 0, 0, 0.513785,-99) , 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502466,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.347171,-99) , 
1, 111.425, 1, 0, 0.492182,-99) , 
0, 215.241, 0, 0, 0.495482,-99) , 
0, 328.658, 0, 0, 0.495751,-99)    );
  // itree = 60
  fBoostWeights.push_back(0.0407196);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.50886,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.552066,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.451291,-99) , 
1, 113.766, 0, 0, 0.511932,-99) , 
NN(
0, 
0, 
-1, 111.425, 1, -1, 0.484183,-99) , 
0, 215.241, 0, 0, 0.488415,-99) , 
0, 328.658, 0, 0, 0.489736,-99)    );
  // itree = 61
  fBoostWeights.push_back(0.0167309);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.498634,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.511284,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.49521,-99) , 
1, 92.9063, 0, 0, 0.499457,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.45778,-99) , 
1, 141.689, 1, 0, 0.496724,-99) , 
0, 328.658, 0, 0, 0.496847,-99)    );
  // itree = 62
  fBoostWeights.push_back(0.0156089);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502817,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.504286,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.48186,-99) , 
1, 105.102, 1, 0, 0.50143,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.461935,-99) , 
1, 141.689, 1, 0, 0.498841,-99) , 
0, 328.658, 0, 0, 0.499098,-99)    );
  // itree = 63
  fBoostWeights.push_back(0.0139543);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, -1, 0.498898,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.50291,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.488683,-99) , 
0, 158.533, 0, 0, 0.498521,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.465817,-99) , 
1, 141.689, 1, 0, 0.496379,-99) , 
0, 328.658, 0, 0, 0.496542,-99)    );
  // itree = 64
  fBoostWeights.push_back(0.0154401);
  fForest.push_back( 
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.502386,-99) , 
NN(
NN(
NN(
0, 
0, 
-1, 0, 1, 1, 0.515151,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.494695,-99) , 
1, 62.417, 1, 0, 0.497185,-99) , 
NN(
0, 
0, 
-1, 0, 1, -1, 0.469291,-99) , 
1, 141.689, 1, 0, 0.495358,-99) , 
0, 328.658, 0, 0, 0.495813,-99)    );
  // itree = 65
  fBoostWeights.push_back(0.00329719);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.498351,-99)    );
  // itree = 66
  fBoostWeights.push_back(0.00164833);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.499176,-99)    );
  // itree = 67
  fBoostWeights.push_back(0.000824036);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.499588,-99)    );
  // itree = 68
  fBoostWeights.push_back(0.000411953);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.499794,-99)    );
  // itree = 69
  fBoostWeights.push_back(0.000205944);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.499897,-99)    );
  // itree = 70
  fBoostWeights.push_back(0.000102956);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.499949,-99)    );
  // itree = 71
  fBoostWeights.push_back(5.14698e-05);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.499974,-99)    );
  // itree = 72
  fBoostWeights.push_back(2.57309e-05);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.499987,-99)    );
  // itree = 73
  fBoostWeights.push_back(1.28634e-05);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.499994,-99)    );
  // itree = 74
  fBoostWeights.push_back(6.43069e-06);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.499997,-99)    );
  // itree = 75
  fBoostWeights.push_back(3.21484e-06);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.499998,-99)    );
  // itree = 76
  fBoostWeights.push_back(1.60717e-06);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.499999,-99)    );
  // itree = 77
  fBoostWeights.push_back(8.03458e-07);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 78
  fBoostWeights.push_back(4.01666e-07);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 79
  fBoostWeights.push_back(2.00801e-07);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 80
  fBoostWeights.push_back(1.00385e-07);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 81
  fBoostWeights.push_back(5.01846e-08);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 82
  fBoostWeights.push_back(2.50884e-08);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 83
  fBoostWeights.push_back(1.25425e-08);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 84
  fBoostWeights.push_back(6.27002e-09);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 85
  fBoostWeights.push_back(3.13435e-09);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 86
  fBoostWeights.push_back(1.56699e-09);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 87
  fBoostWeights.push_back(7.83498e-10);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 88
  fBoostWeights.push_back(3.91637e-10);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 89
  fBoostWeights.push_back(1.95805e-10);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 90
  fBoostWeights.push_back(9.79086e-11);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 91
  fBoostWeights.push_back(4.87971e-11);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 92
  fBoostWeights.push_back(2.44419e-11);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 93
  fBoostWeights.push_back(1.22138e-11);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 94
  fBoostWeights.push_back(6.12366e-12);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 95
  fBoostWeights.push_back(3.06244e-12);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 96
  fBoostWeights.push_back(1.54787e-12);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 97
  fBoostWeights.push_back(7.79488e-13);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 98
  fBoostWeights.push_back(3.9535e-13);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 99
  fBoostWeights.push_back(2.04392e-13);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 100
  fBoostWeights.push_back(9.90319e-14);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 101
  fBoostWeights.push_back(4.84057e-14);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 102
  fBoostWeights.push_back(2.15383e-14);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 103
  fBoostWeights.push_back(1.19904e-14);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 104
  fBoostWeights.push_back(6.66134e-15);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 105
  fBoostWeights.push_back(3.33067e-15);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 106
  fBoostWeights.push_back(6.66134e-16);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
  // itree = 107
  fBoostWeights.push_back(0);
  fForest.push_back( 
NN(
0, 
0, 
0, 328.658, 0, -1, 0.5,-99)    );
   return;
};
 
// Clean up
inline void ReadBDT::Clear() 
{
   for (unsigned int itree=0; itree<fForest.size(); itree++) { 
      delete fForest[itree]; 
   }
}
   inline double ReadBDT::GetMvaValue( const std::vector<double>& inputValues ) const
   {
      // classifier response value
      double retval = 0;

      // classifier response, sanity check first
      if (!IsStatusClean()) {
         std::cout << "Problem in class \"" << fClassName << "\": cannot return classifier response"
                   << " because status is dirty" << std::endl;
         retval = 0;
      }
      else {
         if (IsNormalised()) {
            // normalise variables
            std::vector<double> iV;
            iV.reserve(inputValues.size());
            int ivar = 0;
            for (std::vector<double>::const_iterator varIt = inputValues.begin();
                 varIt != inputValues.end(); varIt++, ivar++) {
               iV.push_back(NormVariable( *varIt, fVmin[ivar], fVmax[ivar] ));
            }
            retval = GetMvaValue__( iV );
         }
         else {
            retval = GetMvaValue__( inputValues );
         }
      }

      return retval;
   }
