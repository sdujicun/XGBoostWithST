package weka.filters.timeseries;

import utilities.ClassifierTools;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.NormalizeCase;
import weka.filters.SimpleBatchFilter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 * Filter to reduce dimensionality of and discretise a time series into SAX
 * form, does not normalize, must be done separately if wanted.
 * 
 * Output attributes can be in two forms - discrete alphabet or real values 0 to
 * alphabetsize-1
 * 
 * Default number of intervals = 8 Default alphabet size = 4
 *
 * @author James
 */
public class SAX extends SimpleBatchFilter {

	private int numIntervals = 8;
	private int alphabetSize = 4;
	private boolean useRealAttributes = false;
	private FastVector alphabet = null;

	private Instances inputFormat;

	private static final long serialVersionUID = 1L;

	// individual strings for each symbol in the alphabet, up to ten symbols
	// private static final String[] alphabetSymbols = {
	// "a","b","c","d","e","f","g","h","i","j" };
	private static final String[] alphabetSymbols = {
				"a0", "b0", "c0", "d0", "e0","f0", "g0", "h0", "i0", "j0", "k0", "l0", "m0", "n0", "o0", "p0", "q0", "r0","s0", "t0", "u0", "v0", "w0", "x0", "y0", "z0",
				"a1", "b1", "c1", "d1", "e1","f1", "g1", "h1", "i1", "j1", "k1", "l1", "m1", "n1", "o1", "p1", "q1", "r1","s1", "t1", "u1", "v1", "w1", "x1", "y1", "z1",
				"a2", "b2", "c2", "d2", "e2","f2", "g2", "h2", "i2", "j2", "k2", "l2", "m2", "n2", "o2", "p2", "q2", "r2","s2", "t2", "u2", "v2", "w2", "x2", "y2", "z2",
				"a3", "b3", "c3", "d3", "e3","f3", "g3", "h3", "i3", "j3", "k3", "l3", "m3", "n3", "o3", "p3", "q3", "r3","s3", "t3", "u3", "v3", "w3", "x3", "y3", "z3",
				"a4", "b4", "c4", "d4", "e4","f4", "g4", "h4", "i4", "j4", "k4", "l4", "m4", "n4", "o4", "p4", "q4", "r4","s4", "t4", "u4", "v4", "w4", "x4", "y4", "z4",
				"a5", "b5", "c5", "d5", "e5","f5", "g5", "h5", "i5", "j5", "k5", "l5", "m5", "n5", "o5", "p5", "q5", "r5","s5", "t5", "u5", "v5", "w5", "x5", "y5", "z5",
				"a6", "b6", "c6", "d6", "e6","f6", "g6", "h6", "i6", "j6", "k6", "l6", "m6", "n6", "o6", "p6", "q6", "r6","s6", "t6", "u6", "v6", "w6", "x6", "y6", "z6",
				"a7", "b7", "c7", "d7", "e7","f7", "g7", "h7", "i7", "j7", "k7", "l7", "m7", "n7", "o7", "p7", "q7", "r7","s7", "t7", "u7", "v7", "w7", "x7", "y7", "z7",
				"a8", "b8", "c8", "d8", "e8","f8", "g8", "h8", "i8", "j8", "k8", "l8", "m8", "n8", "o8", "p8", "q8", "r8","s8", "t8", "u8", "v8", "w8", "x8", "y8", "z8",
				"a9", "b9", "c9", "d9", "e9","f9", "g9", "h9", "i9", "j9", "k9", "l9", "m9", "n9", "o9", "p9", "q9", "r9","s9", "t9", "u9", "v9", "w9", "x9", "y9", "z9"
			};

	public int getNumIntervals() {
		return numIntervals;
	}

	public int getAlphabetSize() {
		return alphabetSize;
	}

	public FastVector getAlphabet() {
		if (alphabet == null)
			generateAlphabet();
		return alphabet;
	}

	public static FastVector getAlphabet(int alphabetSize) {
		FastVector alphabet = new FastVector();
		for (int i = 0; i < alphabetSize; ++i)
			alphabet.addElement(alphabetSymbols[i]);

		return alphabet;
	}

	public void setNumIntervals(int intervals) {
		numIntervals = intervals;
	}

	public void setAlphabetSize(int alphasize) {
		alphabetSize = alphasize;
	}

	public void useRealValuedAttributes(boolean b) {
		useRealAttributes = b;
	}

	public void generateAlphabet() {
		alphabet = new FastVector();
		for (int i = 0; i < alphabetSize; ++i)
			alphabet.addElement(alphabetSymbols[i]);
	}

	// lookup table for the breakpoints for a gaussian curve where the area
	// under
	// curve T between Ti and Ti+1 = 1/a, 'a' being the size of the alphabet.
	// columns up to a=10 stored
	// lit. suggests that a = 3 or 4 is bet in almost all cases, up to 6 or 7 at
	// most
	// for specific datasets
	public double[] generateBreakpoints(int alphabetSize) throws Exception {

		double maxVal = Double.MAX_VALUE;
		double[] breakpoints = null;

		switch (alphabetSize) {
		case 2: {
			breakpoints = new double[] { 0, maxVal };
			break;
		}
		case 3: {
			breakpoints = new double[] { -0.43, 0.43, maxVal };
			break;
		}
		case 4: {
			breakpoints = new double[] { -0.67, 0, 0.67, maxVal };
			break;
		}
		case 5: {
			breakpoints = new double[] { -0.84, -0.25, 0.25, 0.84, maxVal };
			break;
		}
		case 6: {
			breakpoints = new double[] { -0.97, -0.43, 0, 0.43, 0.97, maxVal };
			break;
		}
		case 7: {
			breakpoints = new double[] { -1.07, -0.57, -0.18, 0.18, 0.57, 1.07,
					maxVal };
			break;
		}
		case 8: {
			breakpoints = new double[] { -1.15, -0.67, -0.32, 0, 0.32, 0.67,
					1.15, maxVal };
			break;
		}
		case 9: {
			breakpoints = new double[] { -1.22, -0.76, -0.43, -0.14, 0.14,
					0.43, 0.76, 1.22, maxVal };
			break;
		}
		case 10: {
			breakpoints = new double[] { -1.28, -0.84, -0.52, -0.25, 0.0, 0.25,
					0.52, 0.84, 1.28, maxVal };
			break;
		}

		default: {
			throw new Exception("No breakpoints stored for alphabet size "
					+ alphabetSize);
		}
		}

		return breakpoints;
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {

		// Check all attributes are real valued, otherwise throw exception
		for (int i = 0; i < inputFormat.numAttributes(); i++) {
			if (inputFormat.classIndex() != i) {
				if (!inputFormat.attribute(i).isNumeric()) {
					throw new Exception(
							"Non numeric attribute not allowed for SAX conversion");
				}
			}
		}

		FastVector attributes = new FastVector();

		// If the alphabet is to be considered as discrete values (i.e non
		// real),
		// generate nominal values based on alphabet size
		if (!useRealAttributes)
			generateAlphabet();

		Attribute att;
		String name;

		for (int i = 0; i < numIntervals; i++) {
			name = "SAXInterval_" + i;

			if (!useRealAttributes)
				att = new Attribute(name, alphabet);
			else
				att = new Attribute(name);

			attributes.addElement(att);
		}

		if (inputFormat.classIndex() >= 0) { // Classification set, set class
			// Get the class values as a fast vector
			Attribute target = inputFormat.attribute(inputFormat.classIndex());

			FastVector vals = new FastVector(target.numValues());
			for (int i = 0; i < target.numValues(); i++) {
				vals.addElement(target.value(i));
			}
			attributes.addElement(new Attribute(inputFormat.attribute(
					inputFormat.classIndex()).name(), vals));
		}

		Instances result = new Instances("SAX" + inputFormat.relationName(),
				attributes, inputFormat.numInstances());
		if (inputFormat.classIndex() >= 0) {
			result.setClassIndex(result.numAttributes() - 1);
		}
		return result;
	}

	@Override
	public String globalInfo() {
		throw new UnsupportedOperationException("Not supported yet."); // To
																		// change
																		// body
																		// of
																		// generated
																		// methods,
																		// choose
																		// Tools
																		// |
																		// Templates.
	}

	@Override
	public Instances process(Instances input) throws Exception {

		inputFormat = new Instances(input, 0);
		Instances inputCopy = new Instances(input);
		Instances output = determineOutputFormat(input);

		// Convert input to PAA format
		PAA paa = new PAA();
		paa.setNumIntervals(numIntervals);
		inputCopy = paa.process(inputCopy);

		// Now convert PAA -> SAX
		for (int i = 0; i < inputCopy.numInstances(); i++) {
			double[] data = inputCopy.instance(i).toDoubleArray();

			// remove class attribute if needed
			double[] temp;
			int c = inputCopy.classIndex();
			if (c >= 0) {
				temp = new double[data.length - 1];
				System.arraycopy(data, 0, temp, 0, c); // assumes class
														// attribute is in last
														// index
				data = temp;
			}

			convertSequence(data);

			// Now in SAX form, extract out the terms and set the attributes of
			// new instance
			Instance newInstance;
			if (input.classIndex() >= 0)
				newInstance = new DenseInstance(numIntervals + 1);
			else
				newInstance = new DenseInstance(numIntervals);

			for (int j = 0; j < numIntervals; j++)
				newInstance.setValue(j, data[j]);

			if (inputCopy.classIndex() >= 0)
				newInstance.setValue(output.classIndex(), inputCopy.instance(i)
						.classValue());

			output.add(newInstance);
		}

		return output;
	}

	private void convertSequence(double[] data) throws Exception {
		double[] gaussianBreakpoints = generateBreakpoints(alphabetSize);

		for (int i = 0; i < numIntervals; ++i) {
			// find symbol corresponding to each mean
			for (int j = 0; j < alphabetSize; ++j)
				if (data[i] < gaussianBreakpoints[j]) {
					data[i] = j;
					break;
				}
		}
	}

	/**
	 * Will perform a SAX transformation on a single series passed as a double[]
	 * 
	 * @param alphabetSize
	 *            size of SAX alphabet
	 * @param numIntervals
	 *            size of resulting word
	 * @throws Exception
	 */
	public static double[] convertSequence(double[] data, int alphabetSize,
			int numIntervals) throws Exception {
		SAX sax = new SAX();
		sax.setNumIntervals(numIntervals);
		sax.setAlphabetSize(alphabetSize);
		sax.useRealValuedAttributes(true);

		double[] d = PAA.convertInstance(data, numIntervals);
		sax.convertSequence(d);

		return d;
	}

	/**
	 * Will perform a SAX transformation on a single data series passed as a
	 * double[], input format must already be known.
	 * 
	 * Generally to be used in the SAX_1NN classifier (essentially a wrapper
	 * classifier that just feeds SAX-filtered data to a 1NN classifier) to
	 * filter individual instances during testing
	 * 
	 * Instance objects need the header info as well as the basic data
	 * 
	 * @param alphabetSize
	 *            size of SAX alphabet
	 * @param numIntervals
	 *            size of resulting word
	 * @throws Exception
	 */
	public Instance convertInstance(Instance inst, int alphabetSize,
			int numIntervals) throws Exception {

		Instances newInsts = new Instances(inputFormat, 1);
		newInsts.add(inst);

		newInsts = process(newInsts);

		return newInsts.firstInstance();
	}

	public String getRevision() {
		throw new UnsupportedOperationException("Not supported yet."); // To
																		// change
																		// body
																		// of
																		// generated
																		// methods,
																		// choose
																		// Tools
																		// |
																		// Templates.
	}

	public static void main(String[] args) {
		System.out.println("SAXtest\n\n");

		try {
			Instances test = ClassifierTools
					.loadData("C:\\tempbakeoff\\TSC Problems\\Car\\Car_TEST.arff");

			new NormalizeCase().standardNorm(test);

			SAX sax = new SAX();
			sax.setNumIntervals(2);
			sax.setAlphabetSize(3);
			sax.useRealValuedAttributes(false);
			Instances result = sax.process(test);

			System.out.println(test);
			System.out.println("\n\n\nResults:\n\n");
			System.out.println(result);
		} catch (Exception e) {
			System.out.println(e);
			e.printStackTrace();
		}
	}

}
