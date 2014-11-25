/* -*- mode: java; c-basic-offset: 8; indent-tabs-mode: t; tab-width: 8 -*- */

/* Copyright 2010 Mark Longair */

/*
  This file is part of the ImageJ plugin "Frangi_".

  The ImageJ plugin "Frangi_" is free software; you can
  redistribute it and/or modify it under the terms of the GNU General
  Public License as published by the Free Software Foundation; either
  version 3 of the License, or (at your option) any later version.

  The ImageJ plugin "Frangi_" is distributed in the hope that it
  will be useful, but WITHOUT ANY WARRANTY; without even the implied
  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

package fiji.features;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.measure.Calibration;
import ij.plugin.PlugIn;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import net.imglib2.Cursor;
import net.imglib2.RandomAccess;
import net.imglib2.algorithm.gauss3.Gauss3;
import net.imglib2.exception.ImgLibException;
import net.imglib2.exception.IncompatibleTypeException;
import net.imglib2.img.ImagePlusAdapter;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.img.imageplus.ImagePlusImg;
import net.imglib2.img.imageplus.ImagePlusImgFactory;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;
import Jama.EigenvalueDecomposition;
import Jama.Matrix;

public class Frangi_<T extends RealType<T>> implements PlugIn {

	protected Img<T> image;

	/** A comparator for sorting floats by absolute value */

	public static class AbsoluteFloatComparator implements Comparator {
		public int compare(Object d1, Object d2) {
			return Double.compare( Math.abs(((Float)d1).floatValue()),
					       Math.abs(((Float)d2).floatValue()) );
		}
	}


	/** Produces a String with comma-separated list version of an int array */

	public static String intArrayToString( int [] array ) {
		StringBuilder builder = new StringBuilder();
		boolean firstTime = true;
		for (int i : array) {
			if( firstTime )
				firstTime = false;
			else
				builder.append(", ");
			builder.append(i);
		}
		return builder.toString();
	}

	/** A helper class for calculating the "vesselness" of an
	   image, designed for use from ExecutorService */

	public static class VesselnessCalculator< T extends RealType<T>> implements Callable< Img< FloatType> > {

		protected double alpha = 0.5;
		protected double beta = 0.5;

		protected int scaleIndex;

		protected final Img<T> inputImage;
		protected final float [] spacing;
		protected MultiTaskProgress progress;

		protected double minimumVesselness = Double.MAX_VALUE;
		protected double maximumVesselness = Double.MIN_VALUE;

		public double getMinimumVesselness() {
			return minimumVesselness;
		}

		public double getMaximumVesselness() {
			return maximumVesselness;
		}

		public VesselnessCalculator( final Img<T> input, float[] spacing, int scaleIndex, MultiTaskProgress progress ) {
			this.inputImage = input;
			this.spacing = spacing;
			this.scaleIndex = scaleIndex;
			this.progress = progress;
		}

		protected volatile Img<FloatType> result;

		public Img<FloatType> getResult() {
			return result;
		}

		public Img< FloatType > call() throws Exception {

			final Img<T> input = inputImage;
			final float [] spacing = this.spacing;

			// Denominators used in calculating the vesselness later:
			double ad = 2 * alpha * alpha;
			double bd = 2 * beta * beta;

			/* The cursors may go outside the image, in which case
			   we supply mirror values: */
			Cursor<T> cursor = input.localizingCursor();

			ImgFactory<FloatType> floatFactory = new ImagePlusImgFactory<FloatType>();

			Img<FloatType> resultImage = floatFactory.create( input, new FloatType() );
			RandomAccess<FloatType> resultCursor = resultImage.randomAccess();

			int numberOfDimensions = input.numDimensions();
			long totalPoints = input.size();
			long reportingInterval = totalPoints / 100;

			Matrix hessian = new Matrix( numberOfDimensions, numberOfDimensions );

			/* Two cursors for finding points around the point of
			   interest, used for calculating the second
			   derivatives at that point: */

			RandomAccess<T> ahead = Views.extendMirrorSingle( input ).randomAccess();
			RandomAccess<T> behind = Views.extendMirrorSingle( input ).randomAccess();

			AbsoluteFloatComparator comparator = new AbsoluteFloatComparator();

			long pointsDone = 0;

			while( cursor.hasNext() ) {

				cursor.fwd();

				if( (pointsDone % reportingInterval) == 0 ) {
					double done = pointsDone/(double)totalPoints;
					progress.updateProgress(done,scaleIndex);
				}

				for( int m = 0; m < numberOfDimensions; ++m )
					for( int n = 0; n < numberOfDimensions; ++n ) {

						ahead.setPosition( cursor );
						behind.setPosition( cursor );

						ahead.fwd(m);
						behind.bck(m);

						ahead.fwd(n);
						behind.fwd(n);

						float firstDerivativeA = (ahead.get().getRealFloat() - behind.get().getRealFloat()) / (2 * spacing[m]);

						ahead.bck(n); ahead.bck(n);
						behind.bck(n); behind.bck(n);

						float firstDerivativeB = (ahead.get().getRealFloat() - behind.get().getRealFloat()) / (2 * spacing[m]);

						double value = (firstDerivativeA - firstDerivativeB) / (2 * spacing[n]);
						hessian.set(m,n,value);
					}

				double frobeniusNorm = hessian.normF();

				// Now find the eigenvalues and eigenvalues of the Hessian matrix:
				EigenvalueDecomposition e = hessian.eig();

				/* Nonsense involved in sorting this array of
				   eigenvalues by their absolute values: */

				double [] eigenvaluesArray = e.getRealEigenvalues();
				ArrayList<Float> eigenvaluesArrayList = new ArrayList<Float>(eigenvaluesArray.length);
				for( double ev : eigenvaluesArray )
					eigenvaluesArrayList.add(new Float(ev));

				Collections.sort( eigenvaluesArrayList, comparator );

				double v = 0;

				if( numberOfDimensions == 2 ) {

					double c = 15;
					double cd = 2 * c * c;

					double l1 = eigenvaluesArrayList.get(0);
					double l2 = eigenvaluesArrayList.get(1);

					double rb = l1 / l2;
					double s = Math.sqrt( l1*l1 + l2*l2 );

					double bn = - rb*rb;
					double cn = - s*s;

					if( l2 <= 0 )
						v = Math.exp(bn/bd) * (1 - Math.exp(cn/cd));

				} else if( numberOfDimensions == 3 ) {

					double c = 500;
					double cd = 2 * c * c;

					double l1 = eigenvaluesArrayList.get(0);
					double l2 = eigenvaluesArrayList.get(1);
					double l3 = eigenvaluesArrayList.get(2);

					double rb = Math.abs(l1) / Math.sqrt( Math.abs(l2*l3) );
					double ra = Math.abs(l2) / Math.abs(l3);
					double s = Math.sqrt( l1*l1 + l2*l2 + l3*l3 );

					double an = - ra*ra;
					double bn = - rb*rb;
					double cn = - s*s;

					if( l2 <= 0 && l3 <= 0 )
						v = (1 - Math.exp(an/ad)) * Math.exp(bn/bd) * (1 - Math.exp(cn/cd));

				} else
					throw new RuntimeException("Currently only 2 or 3 dimensional images are supported.");

				if( ! Double.isNaN(v) ) {
					maximumVesselness = Math.max(v,maximumVesselness);
					minimumVesselness = Math.min(v,minimumVesselness);
				}

				resultCursor.setPosition(cursor);
				resultCursor.get().set( (float)v );

				++ pointsDone;
			}

			this.result = resultImage;
			return result;
		}
	}

	/** Returns a version of input that is processed for
	   "vesselness".  This is a convenience method for headless
	   use. */

	public ImagePlus process( ImagePlus input,
				  int scales,
				  double minimumScale,
				  double maximumScale ) {
		return process( input,
				scales,
				minimumScale,
				maximumScale,
				false,
				false,
				false,
				null );
	}

	DecimalFormat f3 = new DecimalFormat("0.000E0");

	public String formatReal(double n) {
		return f3.format(n);
	}


	/** Returns a version of input that is processed for "vesselness" */

	public ImagePlus process( ImagePlus input,
				  int scales,
				  double minimumScale,
				  double maximumScale,
				  boolean showGaussianImages,
				  boolean showFilteredImages,
				  boolean showWhichScales,
				  MultiTaskProgress progress ) {

		int processors = Runtime.getRuntime().availableProcessors();

		image = (Img< T >)ImagePlusAdapter.wrap( input );

		float[] spacing = new float[ image.numDimensions() ];
		
		spacing[ 0 ] = (float)input.getCalibration().pixelWidth;
		spacing[ 1 ] = (float)input.getCalibration().pixelHeight;

		if ( spacing.length > 2 )
			spacing[ 2 ] = (float)input.getCalibration().pixelDepth;

		double range = maximumScale - minimumScale;
		double increment = 0;
		if( scales > 1 )
			increment = range / (scales - 1);

		ArrayList< Img<FloatType> > vesselImages =
			new ArrayList< Img<FloatType> >();

		ExecutorService es = Executors.newFixedThreadPool(processors);

		List<VesselnessCalculator<T>> calculators =
			new ArrayList<VesselnessCalculator<T>>();

		/* Smooth the input image with a Gaussian convolution
		   at each scale.  Also create a VesselnessCalculator
		   object for each scale and add it to the
		   ExecutorService */

		for( int scaleIndex = 0; scaleIndex < scales; ++scaleIndex ) {

			double currentScale = minimumScale + scaleIndex * increment;
			float [] newSpacing = new float[spacing.length];
			for( int i = 0; i < newSpacing.length; ++i )
				newSpacing[i] = (float)( spacing[i] * (currentScale / spacing[0]) );

			double [] sigmas = new double[newSpacing.length];
			for( int i = 0; i < newSpacing.length; ++i )
				sigmas[i] = newSpacing[i] / spacing[i];

			final Img< T > result = image.factory().create( image, image.firstElement() );
			try
			{
				Gauss3.gauss( sigmas, Views.extendMirrorSingle( image ), result );
			}
			catch (IncompatibleTypeException e) { e.printStackTrace(); }


			if( showGaussianImages )
			{
				ImagePlus imp = null;

				if ( result instanceof ImagePlusImg )
					try { imp = ((ImagePlusImg)result).getImagePlus(); } catch (ImgLibException e) {}

				if ( imp == null )
					imp = ImageJFunctions.wrap( result, "Gaussian smoothed images at scale "+scaleIndex ).duplicate();
				else
					imp.setTitle( "Gaussian smoothed images at scale "+scaleIndex );

				imp.show();
			}

			VesselnessCalculator<T> calculator =
				new VesselnessCalculator<T>( result, newSpacing, scaleIndex, progress );
			calculators.add( calculator );
		}

		List<Future<Img<FloatType>>> futures = null;

		/* Actually run all the VesselnessCalculators with
		   invokeAll, which returns when all have finished */

		try {
			futures = es.invokeAll(calculators);
		} catch( InterruptedException ie ) {
			/* We never call interrupt on these threads,
			   so this should never happen... */
		}

		/* We need to get() on each Future to check if there
		   was an exception thrown: */

		try {
			for( Future<Img<FloatType>> future : futures ) {
				future.get();
			}
		} catch( InterruptedException id ) {
			/* Similarly this should never happen */
		} catch( Exception e ) {
			IJ.error("The following exception was thrown: "+e);
			e.printStackTrace();
			return null;
		}

		/* Get the images from each VesselnessCalculator,
		   displaying them if showFilteredImages was
		   specified: */

		int scale = 0;
		for( VesselnessCalculator vc : calculators ) {
			Img<FloatType> image = vc.getResult();
			vesselImages.add( image );
			if( showFilteredImages ) {
				ImagePlus imagePlusVersion = null;//ImageJFunctions.wrap( image, "Filtered image at scale "+scale ).duplicate();

				if ( image instanceof ImagePlusImg )
					try { imagePlusVersion = ((ImagePlusImg)image).getImagePlus(); } catch (ImgLibException e) {}

				if ( imagePlusVersion == null )
					imagePlusVersion = ImageJFunctions.wrap( image, "Filtered image at scale "+scale ).duplicate();
				else
					imagePlusVersion.setTitle( "Filtered image at scale "+scale );

				imagePlusVersion.setDisplayRange( vc.getMinimumVesselness(),
								  0.5 * vc.getMaximumVesselness() );

				imagePlusVersion.show();
			}
			++ scale;
		}


		/* Now we combine all the vesselness images from each
		   scale, by taking the maximum value at each point in
		   the image across all the vesselness images */

		ImgFactory<FloatType> floatFactory = new ImagePlusImgFactory<FloatType>();

		Img<FloatType> resultImage = floatFactory.create( image, new FloatType() );
		Cursor<FloatType> resultCursor = resultImage.localizingCursor();

		Img<FloatType> whichImage = null;
		Cursor<FloatType> whichCursor = null;

		if( showWhichScales ) {
			whichImage = floatFactory.create( image, new FloatType() );
			whichCursor = whichImage.localizingCursor();
		}

		ArrayList<Cursor<FloatType>> cursors =
			new ArrayList<Cursor<FloatType>>();
		for( Img<FloatType> vesselImage : vesselImages ) {
			cursors.add(vesselImage.localizingCursor());
		}

		float maximumValueInResult = Float.MIN_VALUE;
		float minimumValueInResult = Float.MAX_VALUE;

		while( resultCursor.hasNext() ) {
			resultCursor.fwd();
			if( showWhichScales )
				whichCursor.fwd();
			/* If bestScale remains 0, that indicates that
			   all images had NaN as their values at that
			   point */
			int bestScale = 0;
			float largestValue = Float.MIN_VALUE;
			int scaleIndex = 0;
			for( Cursor<FloatType> cursor : cursors ) {
				cursor.fwd();
				float v = cursor.get().getRealFloat();
				if( v > largestValue ) {
					bestScale = scaleIndex + 1;
					largestValue = v;
				}
				++ scaleIndex;
			}
			if( showWhichScales )
				whichCursor.get().set( bestScale == 0 ? 0 : (float)minimumScale + (bestScale - 1) * (float)increment );
			resultCursor.get().set(largestValue);
			maximumValueInResult = Math.max(maximumValueInResult,largestValue);
			minimumValueInResult = Math.min(minimumValueInResult,largestValue);
		}

		/* Remember to close all of the cursors */

		if( showWhichScales ) {
			ImagePlus whichImagePlus = null;
			
			if ( whichImage instanceof ImagePlusImg )
				try { whichImagePlus = ((ImagePlusImg)whichImage).getImagePlus(); } catch (ImgLibException e) {}

			if ( whichImagePlus == null )
				whichImagePlus = ImageJFunctions.wrap( whichImage, "Scales used" ).duplicate();
			else
				whichImagePlus.setTitle( "Scales used" );

			whichImagePlus.getProcessor().setMinAndMax(0,maximumScale);
			whichImagePlus.show();
		}

		if( progress != null )
			progress.done();

		ImagePlus resultImagePlus = null;

		if ( resultImage instanceof ImagePlusImg )
			try { resultImagePlus = ((ImagePlusImg)resultImage).getImagePlus(); } catch (ImgLibException e) {}

		if ( resultImagePlus == null )
			resultImagePlus = ImageJFunctions.wrap( resultImage, "vesselness of "+input.getTitle() ).duplicate();
		else
			resultImagePlus.setTitle( "vesselness of "+input.getTitle() );

		resultImagePlus.setDisplayRange( minimumValueInResult,
						 0.5 * maximumValueInResult );
		resultImagePlus.setCalibration( input.getCalibration() );
		return resultImagePlus;
	}

	/** An implementation of the MultiTaskProgress interface that
	    updates the ImageJ progress bar */

	public static class Progress implements MultiTaskProgress {
		ArrayList<Double> tasksProportionsDone;
		int totalTasks;

		public Progress( int totalTasks ) {
			tasksProportionsDone =
				new ArrayList<Double>();
			this.totalTasks = totalTasks;
			for( int i = 0; i < totalTasks; ++i )
				tasksProportionsDone.add(0.0);
		}

		synchronized public void updateProgress(double proportion, int taskIndex) {
			tasksProportionsDone.set(taskIndex,proportion);
			updateStatus();
		}

		protected void updateStatus() {
			double totalDone = 0;
			for( double p : tasksProportionsDone )
				totalDone += p;
			IJ.showProgress(totalDone/totalTasks);
		}

		public void done() {
			IJ.showProgress(1.0);
		}
	}

	/** Ask for parameters via a GenericDialog and process the image */

	public void run( String ignored ) {

		ImagePlus imagePlus = IJ.getImage();
		if( imagePlus == null ) {
			IJ.error("There's no image open to work on.");
			return;
		}

		Calibration c = imagePlus.getCalibration();
		double pixelWidth = Math.abs(c.pixelWidth);

		GenericDialog gd = new GenericDialog("Frangi Options");
		gd.addNumericField("Number of scales", 1, 0);
		gd.addNumericField("Minimum scale", pixelWidth, 6);
		gd.addNumericField("Maximum scale", pixelWidth, 6);
		gd.addCheckbox("Show Gaussian smoothed images:",false);
		gd.addCheckbox("Show filtered images at each scale:",false);
		gd.addCheckbox("Show which scales were used at each point:",false);

		gd.showDialog();
		if( gd.wasCanceled() )
			return;

		int scales = (int)Math.round(gd.getNextNumber());
		if( scales < 1 ) {
			IJ.error("The minimum number of scales to try is 1");
			return;
		}
		double minimumScale = gd.getNextNumber();
		double maximumScale = gd.getNextNumber();
		if( maximumScale < minimumScale ) {
			IJ.error("The maximum scale cannot be less than the minimum scale");
			return;
		}

		boolean showGaussianImages = gd.getNextBoolean();
		boolean showFilteredImages = gd.getNextBoolean();
		boolean showWhichScales = gd.getNextBoolean();

		Progress progress = new Progress(scales);

		ImagePlus result = process( imagePlus,
					    scales,
					    minimumScale,
					    maximumScale,
					    showGaussianImages,
					    showFilteredImages,
					    showWhichScales,
					    progress );

		if( result != null )
			result.show();
	}
}
