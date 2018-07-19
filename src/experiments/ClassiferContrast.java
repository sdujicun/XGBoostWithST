package experiments;

import weka.filters.timeseries.shapelet_transforms.ShapeletTransformBasedOnLFDP;

public class ClassiferContrast {
	static String[] problems = { 
		"Adiac", 
		"Beef", 
		"ChlorineConcentration",
		"Coffee", 
		"DiatomSizeReduction", 
		"ItalyPowerDemand", 
		"Lightning7",
		"MedicalImages", 
		"MoteStrain", 
		"Symbols", 
		"Trace", 
		"TwoLeadECG" 
	};
	public static void main(String[] args){
		 System.out.println("dataset\t" + "C45\t"+"1NN\t" + "BN\t" +
				 "bayesNet\t" + "RandF\t" + "RotF\t"+ "SVM\t" + "WeightedEnsemble");
		for(String problem:problems){
			System.out.print(problem+"\t");
			new ShapeletTransformBasedOnLFDP().trainTestExample(problem);
		}
	}
}
