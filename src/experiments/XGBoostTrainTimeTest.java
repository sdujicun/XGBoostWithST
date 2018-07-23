package experiments;

import xgboost.XGBoostWithST;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class XGBoostTrainTimeTest {
	public static void main(String[] args) throws Exception {
		XGBoostWithST boost = new XGBoostWithST();
		boost.time();
	}
}
