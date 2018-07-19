package experiments;

import xgboost.XGBoostWithST;
import ml.dmlc.xgboost4j.java.XGBoostError;

public class XGBoostAccutacyTest {
	public static void main(String[] args) throws Exception {
//		 basicTestForXGBoostWithST();
		 crossTestForXGBoostWithST();
	}

	public static void basicTestForXGBoostWithST() throws Exception {
		XGBoostWithST boost = new XGBoostWithST();
		boost.test(false);

	}

	public static void crossTestForXGBoostWithST() throws Exception {
		XGBoostWithST boost = new XGBoostWithST();
		boost.test(true);
	}
}
