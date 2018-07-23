package xgboost;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import utilities.ClassifierTools;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.shapelet.QualityMeasures;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformBasedOnLFDP;
import weka.filters.timeseries.shapelet_transforms.ShapeletTransformWithSubclassSampleAndLFDP;
import weka.filters.timeseries.shapelet_transforms.classValue.BinarisedClassValue;
import weka.filters.timeseries.shapelet_transforms.subsequenceDist.ImprovedOnlineSubSeqDistance;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class XGBoostWithST {

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

	static double bestEta = 0.1;
	static String bestBooster = "gbtree";

	public static void basicpredict(String dataset, int num_class,
			String trainPath, String testPath, String dateString)
			throws XGBoostError, IOException {

		DMatrix trainMat = new DMatrix(trainPath);
		DMatrix testMat = new DMatrix(testPath);

		// specify parameters
		Map<String, Object> params = new HashMap<String, Object>();

		params.put("silent", 1); // 为1的时候不会打印模型迭代的信息，为0可以看到打印的信息
		params.put("nthread", 3); // 不使用的话系统会默认得到最大的线程数目
		params.put("objective", "multi:softmax");// 目标函数值
		params.put("num_class", num_class);//
		params.put("max_depth", 10);// 树最大深度
		params.put("min_child_weight", 3);
		params.put("eval_metric", "merror");//
		
		Params paramsToUse = new Params();
		paramsToUse.setSTParams();
		if (null == paramsToUse.boosterMap
				|| (!paramsToUse.boosterMap.containsKey(dataset))) {
			bestBooster = "gbtree";
			bestEta = 0.1;
		} else {
			bestBooster = paramsToUse.boosterMap.get(dataset);
			bestEta = paramsToUse.etaMap.get(dataset);
		}

		params.put("booster", bestBooster);// gblinear gbtree dart
		params.put("eta", bestEta); // 为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重

		// specify watchList
		HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
		watches.put("train", trainMat);
		watches.put("test", testMat);
		// train a booster
		int round = 1000;

		Booster booster = XGBoost.train(trainMat, params, round, watches, null,
				null);

		float[][] result = booster.predict(testMat);
		float[] label = testMat.getLabel();
		int rightNo = 0;
		for (int i = 0; i < result.length; i++) {
			if (label[i] - result[i][0] == 0) {
				rightNo++;
			}

			// System.out.println("true class:"+label[i]+"predict class:"+result[i][0]);
		}
		double rightRate = 1.0 * rightNo / result.length;

		String path = "results/ST_result_" + dateString + ".txt";

		File resultFile = new File(path);
		if (!resultFile.exists()) {
			resultFile.createNewFile();
		}
		FileWriter writer = new FileWriter(resultFile, true);

		writer.write(dataset + "\t" + rightRate
				+ System.getProperty("line.separator"));
		writer.close();

		System.out.println(dataset + "  rightRate:" + rightRate);
	}
	
	
	public static void crosspredict(String dataset, int num_class,
			String trainPath, String testPath, String dateString)
			throws XGBoostError, IOException {

		DMatrix trainMat = new DMatrix(trainPath);
		DMatrix testMat = new DMatrix(testPath);

		// specify parameters
		Map<String, Object> params = new HashMap<String, Object>();
		params.put("silent", 1); // 为1的时候不会打印模型迭代的信息，为0可以看到打印的信息
		params.put("nthread", 3); // 不使用的话系统会默认得到最大的线程数目
		params.put("objective", "multi:softmax");// 目标函数值
		params.put("num_class", num_class);//
		params.put("max_depth", 10);// 树最大深度
		params.put("min_child_weight", 3);
		params.put("eval_metric", "merror");//

		// specify watchList
		HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
		watches.put("train", trainMat);
		watches.put("test", testMat);
		// train a booster
		int round = 1000;

		Params paramsToUse = new Params();
		paramsToUse.setSTParams();
		if (null == paramsToUse.boosterMap
				|| (!paramsToUse.boosterMap.containsKey(dataset))) {
			crossValidation(dataset, num_class, trainPath, testPath, dateString);
		} else {
			bestBooster = paramsToUse.boosterMap.get(dataset);
			bestEta = paramsToUse.etaMap.get(dataset);
		}

		params.put("booster", bestBooster);// gblinear gbtree dart
		params.put("eta", bestEta); // 为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重

		Booster booster = XGBoost.train(trainMat, params, round, watches, null,
				null);

		float[][] result = booster.predict(testMat);
		float[] label = testMat.getLabel();
		int rightNo = 0;
		for (int i = 0; i < result.length; i++) {
			if (label[i] - result[i][0] == 0) {
				rightNo++;
			}

			// System.out.println("true class:"+label[i]+"predict class:"+result[i][0]);
		}
		double rightRate = 1.0 * rightNo / result.length;

		String path = "results/ST_result_" + dateString + ".txt";

		File resultFile = new File(path);
		if (!resultFile.exists()) {
			resultFile.createNewFile();
		}
		FileWriter writer = new FileWriter(resultFile, true);

		writer.write(dataset + "\t" + rightRate
				+ System.getProperty("line.separator"));
		writer.close();

		System.out.println(dataset + "  rightRate:" + rightRate);
	}
	
	
	public static void XGTrainTime(String dataset, int num_class,
			String trainPath, String testPath, String dateString)
			throws XGBoostError, IOException {

		DMatrix trainMat = new DMatrix(trainPath);
		DMatrix testMat = new DMatrix(testPath);

		// specify parameters
		Map<String, Object> params = new HashMap<String, Object>();
		params.put("silent", 1); // 为1的时候不会打印模型迭代的信息，为0可以看到打印的信息
		params.put("nthread", 3); // 不使用的话系统会默认得到最大的线程数目
		params.put("objective", "multi:softmax");// 目标函数值
		params.put("num_class", num_class);//
		params.put("max_depth", 10);// 树最大深度
		params.put("min_child_weight", 3);
		params.put("eval_metric", "merror");//

		// specify watchList
		HashMap<String, DMatrix> watches = new HashMap<String, DMatrix>();
		watches.put("train", trainMat);
		watches.put("test", testMat);
		// train a booster
		int round = 1000;

		Params paramsToUse = new Params();
		paramsToUse.setSTParams();
		if (null == paramsToUse.boosterMap
				|| (!paramsToUse.boosterMap.containsKey(dataset))) {
			crossValidation(dataset, num_class, trainPath, testPath, dateString);
		} else {
			bestBooster = paramsToUse.boosterMap.get(dataset);
			bestEta = paramsToUse.etaMap.get(dataset);
		}

		params.put("booster", bestBooster);// gblinear gbtree dart
		params.put("eta", bestEta); // 为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重

		long d1 = System.nanoTime();
		XGBoost.train(trainMat, params, round, watches, null, null);

		long d2 = System.nanoTime();
		// ArrayList<Shapelet> sh = transform.getShapelets();
		System.out.print("Shapelet Selection Time\t"+(d2 - d1) * 0.000000001+ "\t");
		
		

		String path = "results/trainTime" + dateString + ".txt";

		File resultFile = new File(path);
		if (!resultFile.exists()) {
			resultFile.createNewFile();
		}
		FileWriter writer = new FileWriter(resultFile, true);

		writer.write(dataset + "\t" + (d2 - d1) * 0.000000001
				+ System.getProperty("line.separator"));
		writer.close();
	}
	public static void crossValidation(String dataset, int num_class,
			String trainPath, String testPath, String dateString)
			throws XGBoostError, IOException {
		DMatrix trainMat = new DMatrix(trainPath);

		int round = 1000;

		// specify parameters
		Map<String, Object> params = new HashMap<String, Object>();
		params.put("silent", 1); // 为1的时候不会打印模型迭代的信息，为0可以看到打印的信息
		params.put("nthread", 3); // 不使用的话系统会默认得到最大的线程数目
		params.put("objective", "multi:softmax");// 目标函数值
		params.put("num_class", num_class);//
		params.put("max_depth", 10);// 树最大深度
		params.put("min_child_weight", 3);
		params.put("eval_metric", "merror");//
		// params.put("eval_metric", "mlogloss");//

		double eta;
		String booster;

		String[] crossString;
		double besterror = Double.MAX_VALUE;
		for (int a = 0; a < 15; a++) {
			eta = 0.02 + a * 0.02;
			params.put("eta", eta); // 为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重

			String[] boosters = { "gblinear", "gbtree" };
			for (int c = 0; c < 2; c++) {
				booster = boosters[c];
				System.out
						.println(dataset + "\t" + eta + "\t" + booster + "\t");
				params.put("booster", booster);// gblinear gbtree dart
				crossString = XGBoost.crossValidation(trainMat, params, round,
						5, null, null, null);
				
				

				String lastString = crossString[crossString.length - 1];
				int index = lastString.indexOf(":");
				lastString = lastString.substring(index + 1);
				index = lastString.indexOf("\t");
				lastString = lastString.substring(0, index).trim();
				double temp = Double.parseDouble(lastString);
				if (temp < besterror) {
					besterror = temp;
					bestEta = eta;

					bestBooster = booster;
					System.out.println(dataset + "\t" + bestEta + "\t"
							+ bestBooster + "\t");
				}

			}
		}

		String path = "params/ST_params.txt";
		File resultFile = new File(path);
		if (!resultFile.exists()) {
			resultFile.createNewFile();
		}
		FileWriter writer = new FileWriter(resultFile, true);

		writer.write(dataset + "\t" + bestEta + "\t" + bestBooster + "\t"
				+ System.getProperty("line.separator"));
		writer.close();

	}

	public void test(boolean cross) throws XGBoostError, Exception {
		String dataDir = "G:/数据/TSC Problems/";
		Instances train, test;

		SimpleDateFormat formatter = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
		String dateString = formatter.format(new Date());
		String[] datasets = problems;
		for (String dataset : datasets) {
			System.out.println(dataset
					+ "--------------------------------------------");

			train = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TRAIN");
			test = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TEST");

			int num_class = train.numClasses();

			String trainPath = "middleFiles/ST/" + dataset + "_train.txt";
			String testPath = "middleFiles/ST/" + dataset + "_test.txt";

			File trainFile = new File(trainPath);
			File testFile = new File(testPath);

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

			Instances tranTrain, tranTest;
			if ((!trainFile.exists()) || (!testFile.exists())) {
				tranTrain = transform.process(train);
				tranTest = transform.process(test);
				int trainNo = train.numInstances();
				double[][] stTrainMatrix = new double[trainNo][tranTrain
						.numAttributes() - 1];
				for (int i = 0; i < trainNo; i++) {
					Instance temp = tranTrain.instance(i);
					for (int j = 0; j < temp.numAttributes() - 1; j++) {
						stTrainMatrix[i][j] = temp.value(j);
					}
				}
				Map map = MaxMinNormal.maxMinNormal(stTrainMatrix);
				stTrainMatrix = (double[][]) map.get("data");
				if (!trainFile.exists()) {
					trainFile.createNewFile();
					FileWriter writer = new FileWriter(trainFile, true);
					for (int i = 0; i < trainNo; i++) {
						Instance temp = tranTrain.instance(i);
						String newLine = temp.classValue() + " ";
						for (int j = 0; j < temp.numAttributes() - 1; j++) {
							// newLine += "sh" + j + ":" + temp.value(j) + "  ";
							newLine += "sh" + j + ":" + stTrainMatrix[i][j]
									+ "  ";
						}
						newLine += System.getProperty("line.separator");
						writer.write(newLine);
					}
					writer.close();
				}

				if (!testFile.exists()) {

					testFile.createNewFile();
					int testNo = train.numInstances();
					double[][] stTestMatrix = new double[testNo][tranTrain
							.numAttributes() - 1];
					for (int i = 0; i < testNo; i++) {
						Instance temp = tranTest.instance(i);
						for (int j = 0; j < temp.numAttributes() - 1; j++) {
							stTestMatrix[i][j] = temp.value(j);
						}
					}
					double[] min = (double[]) map.get("min");
					double[] max = (double[]) map.get("max");
					stTestMatrix = MaxMinNormal.maxMinNormal(min, max,
							stTestMatrix);
					FileWriter writer = new FileWriter(testFile, true);
					for (int i = 0; i < testNo; i++) {
						Instance temp = tranTest.instance(i);
						String newLine = temp.classValue() + " ";
						for (int j = 0; j < temp.numAttributes() - 1; j++) {
							// newLine += "sh" + j + ":" + temp.value(j) + "  ";
							newLine += "sh" + j + ":" + stTestMatrix[i][j]
									+ "  ";
						}
						newLine += System.getProperty("line.separator");
						writer.write(newLine);
					}
					writer.close();
				}

			}
			if (cross) {
				crosspredict(dataset, num_class, trainPath, testPath,
						dateString);
			} else {
				basicpredict(dataset, num_class, trainPath, testPath,
						dateString);
			}

		}

	}
	
	public void time() throws XGBoostError, Exception {
		String dataDir = "G:/数据/TSC Problems/";
		Instances train, test;

		SimpleDateFormat formatter = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
		String dateString = formatter.format(new Date());
		String[] datasets = problems;
		for (String dataset : datasets) {
			train = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TRAIN");
			test = ClassifierTools.loadData(dataDir + dataset + "/" + dataset
					+ "_TEST");

			int num_class = train.numClasses();

			String trainPath = "middleFiles/ST/" + dataset + "_train.txt";
			String testPath = "middleFiles/ST/" + dataset + "_test.txt";

			File trainFile = new File(trainPath);
			File testFile = new File(testPath);

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

			Instances tranTrain, tranTest;
			if ((!trainFile.exists()) || (!testFile.exists())) {
				tranTrain = transform.process(train);
				tranTest = transform.process(test);
				int trainNo = train.numInstances();
				double[][] stTrainMatrix = new double[trainNo][tranTrain
						.numAttributes() - 1];
				for (int i = 0; i < trainNo; i++) {
					Instance temp = tranTrain.instance(i);
					for (int j = 0; j < temp.numAttributes() - 1; j++) {
						stTrainMatrix[i][j] = temp.value(j);
					}
				}
				Map map = MaxMinNormal.maxMinNormal(stTrainMatrix);
				stTrainMatrix = (double[][]) map.get("data");
				if (!trainFile.exists()) {
					trainFile.createNewFile();
					FileWriter writer = new FileWriter(trainFile, true);
					for (int i = 0; i < trainNo; i++) {
						Instance temp = tranTrain.instance(i);
						String newLine = temp.classValue() + " ";
						for (int j = 0; j < temp.numAttributes() - 1; j++) {
							// newLine += "sh" + j + ":" + temp.value(j) + "  ";
							newLine += "sh" + j + ":" + stTrainMatrix[i][j]
									+ "  ";
						}
						newLine += System.getProperty("line.separator");
						writer.write(newLine);
					}
					writer.close();
				}

				if (!testFile.exists()) {

					testFile.createNewFile();
					int testNo = train.numInstances();
					double[][] stTestMatrix = new double[testNo][tranTrain
							.numAttributes() - 1];
					for (int i = 0; i < testNo; i++) {
						Instance temp = tranTest.instance(i);
						for (int j = 0; j < temp.numAttributes() - 1; j++) {
							stTestMatrix[i][j] = temp.value(j);
						}
					}
					double[] min = (double[]) map.get("min");
					double[] max = (double[]) map.get("max");
					stTestMatrix = MaxMinNormal.maxMinNormal(min, max,
							stTestMatrix);
					FileWriter writer = new FileWriter(testFile, true);
					for (int i = 0; i < testNo; i++) {
						Instance temp = tranTest.instance(i);
						String newLine = temp.classValue() + " ";
						for (int j = 0; j < temp.numAttributes() - 1; j++) {
							// newLine += "sh" + j + ":" + temp.value(j) + "  ";
							newLine += "sh" + j + ":" + stTestMatrix[i][j]
									+ "  ";
						}
						newLine += System.getProperty("line.separator");
						writer.write(newLine);
					}
					writer.close();
				}

			}
			
			XGTrainTime(dataset, num_class, trainPath, testPath,
						dateString);

		}

	}

}
