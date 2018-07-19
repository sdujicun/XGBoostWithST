package xgboost;

import java.util.HashMap;
import java.util.Map;

public class Params{
	public Map<String,Double> etaMap;
	public  Map<String,String> boosterMap;
	
	public void setSTParams(){
		etaMap=new HashMap<String,Double>();
		boosterMap=new HashMap<String,String>();
		//"Adiac", // 390,391,176,37
		etaMap.put("Adiac",0.02);
		boosterMap.put("Adiac","gblinear");
		//"Beef", // 30,30,470,5
		etaMap.put("Beef",0.08);
		boosterMap.put("Beef","gbtree");
		//"ChlorineConcentration", 
		etaMap.put("ChlorineConcentration",0.1);
		boosterMap.put("ChlorineConcentration","gbtree");
		//"Coffee", // 28,28,286,2
		etaMap.put("Coffee",0.01);
		boosterMap.put("Coffee","gblinear");
		//"DiatomSizeReduction", // 16,306,345,4
		etaMap.put("DiatomSizeReduction",0.02);
		boosterMap.put("DiatomSizeReduction","gblinear");
		//"ItalyPowerDemand", // 67,1029,24,2
		etaMap.put("ItalyPowerDemand",0.02);
		boosterMap.put("ItalyPowerDemand","gblinear");
		//"Lightning7", // 70,73,319,7
		etaMap.put("Lightning7",0.01);
		boosterMap.put("Lightning7","gblinear");
		//"MedicalImages", // 381,760,99,10
		etaMap.put("MedicalImages",0.3);
		boosterMap.put("MedicalImages","gbtree");
		//"MoteStrain", // 20,1252,84,2
		etaMap.put("MoteStrain",0.02);
		boosterMap.put("MoteStrain","gblinear");
		//"Symbols", // 25,995,398,6
		etaMap.put("Symbols",0.02);
		boosterMap.put("Symbols","gblinear");
		//"Trace", // 100,100,275,4
		etaMap.put("Trace",0.02);
		boosterMap.put("Trace","gblinear");
		//"TwoLeadECG", // 23,1139,82,2
		etaMap.put("TwoLeadECG",0.02);
		boosterMap.put("TwoLeadECG","gblinear");
		
	}
	
}
