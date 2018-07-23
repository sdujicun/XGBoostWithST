package experiments;

import java.io.File;
import java.util.logging.Level;
import java.util.logging.Logger;

import utilities.ClassifierTools;
import utilities.fileIO.DataSets;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.kNN;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.meta.timeseriesensembles.WeightedEnsemble;
import weka.classifiers.trees.EnhancedRandomForest;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformWithSubclassSampleAndLFDP;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;

public class FSSTrainTimeTest {
	public static void main(String[] args) {
		String[] problems = { "ChlorineConcentration", "Coffee", "ECGFiveDays",
				"MoteStrain", "Trace", "TwoLeadECG" };

		System.out.println("dataset\t" + "shapelet selection time \t"
				+ "Weight Ensemble Train Time");
		for (int i = 0; i < problems.length; i++) {
			System.out.print(problems[i] + "\t");
			trainTime(problems[i]);
		}

	}

	public static void trainTime(String problem) {
		try {
			final String resampleLocation = DataSets.problemPath;
			final String dataset = problem;
			final String filePath = resampleLocation + File.separator + dataset
					+ File.separator + dataset;
			Instances train= utilities.ClassifierTools.loadData(filePath + "_TRAIN");
			
			
			ShapeletTransformWithSubclassSampleAndLFDP transform = new ShapeletTransformWithSubclassSampleAndLFDP();
			transform.setRoundRobin(true);
			transform.turnOffLog();
			// construct shapelet classifiers.
			transform.setClassValue(new BinarisedClassValue());
			transform.setSubSeqDistance(new ImprovedOnlineSubSeqDistance());
			transform.useCandidatePruning();
			int shapletNo = train.numInstances() * 2 > 100 ? train
					.numInstances() * 2 : 100;

			transform.setNumberOfShapelets(shapletNo);
			transform
					.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
			long d1 = System.nanoTime();
			Instances tranTrain = transform.process(train);
			long d2 = System.nanoTime();
			System.out.print((d2 - d1) * 0.000000001);

			// 8WeightedEnsemble
			WeightedEnsemble we = new WeightedEnsemble();
			d1 = System.nanoTime();
			we.buildClassifier(tranTrain);
			d2 = System.nanoTime();
			System.out.println((d2 - d1) * 0.000000001);

		} catch (Exception ex) {
			Logger.getLogger(
					ShapeletTransformWithSubclassSampleAndLFDP.class.getName())
					.log(Level.SEVERE, null, ex);
		}
	}
}
