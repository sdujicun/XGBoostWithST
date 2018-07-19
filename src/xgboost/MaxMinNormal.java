package xgboost;

import java.util.HashMap;
import java.util.Map;

public class MaxMinNormal {
	public static Map maxMinNormal(double[][] data){
		Map map=new HashMap();
		int length=data[1].length;
		double[] max=new double[length];
		double[] min=new double[length];
		for(int i=0;i<length;i++){
			max[i]=Double.MIN_VALUE;
			min[i]=Double.MAX_VALUE;
		}
		for(int j=0;j<data.length;j++){
			for(int i=0;i<length;i++){
				if(data[j][i]>max[i]){
					max[i]=data[j][i];
				}
				if(data[j][i]<min[i]){
					min[i]=data[j][i];
				}
			}
		}
		
		for(int i=0;i<length;i++){
			double minTemp=min[i];
			double diff=max[i]-minTemp;
			if(diff<=0){
				for(int j=0;j<data.length;j++ ){
					data[j][i]=0.0;
				}
			}else{
				for(int j=0;j<data.length;j++ ){
					data[j][i]=(data[j][i]-minTemp)/diff;
				}
			}
		}
		map.put("max", max);
		map.put("min", min);
		map.put("data", data);
		return map;
		
	}
	
	public static double[][] maxMinNormal(double[] min,double[] max, double[][] data){
		int length=min.length;
		for(int i=0;i<length;i++){
			double minTemp=min[i];
			double diff=max[i]-minTemp;
			if(diff<=0){
				for(int j=0;j<data.length;j++ ){
					data[j][i]=0.0;
				}
			}else{
				for(int j=0;j<data.length;j++ ){
					data[j][i]=(data[j][i]-minTemp)/diff;
				}
			}
		}
		return data;
		
	}
}
