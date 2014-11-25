/* -*- mode: java; c-basic-offset: 8; indent-tabs-mode: t; tab-width: 8 -*- */

/* Copyright 2010 Mark Longair */

/*
  This file is part of the ImageJ plugin "Curvatures_".

  The ImageJ plugin "Curvatures_" is free software; you can
  redistribute it and/or modify it under the terms of the GNU General
  Public License as published by the Free Software Foundation; either
  version 3 of the License, or (at your option) any later version.

  The ImageJ plugin "Curvatures_" is distributed in the hope that it
  will be useful, but WITHOUT ANY WARRANTY; without even the implied
  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

package fiji.features;

import ij.IJ;
import ij.ImagePlus;
import ij.plugin.PlugIn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

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

public class Curvatures_<T extends RealType<T>> implements PlugIn {

	protected Img<T> image;

	/* This comparator is useful for sorting a collection of Float
	   objects by their absolute value, largest first */

	public class ReverseAbsoluteFloatComparator implements Comparator {
		public int compare(Object d1, Object d2) {
			return Double.compare( Math.abs(((Float)d2).floatValue()),
					       Math.abs(((Float)d1).floatValue()) );
		}
	}

	/** Generate an ArrayList of images, each of which contains a
	    particular eigenvalue of the Hessian matrix at each point
	    in the image. */

	public ArrayList< Img<FloatType> > hessianEigenvalueImages( Img<T> input, float [] spacing ) {

		/* Various cursors may go outside the image, in which
		   case we supply mirror values: */

		Cursor<T> cursor = input.localizingCursor();

		ImgFactory<FloatType> floatFactory = new ImagePlusImgFactory<FloatType>();

		ArrayList< Img<FloatType> > eigenvalueImages = new ArrayList< Img<FloatType> >();
		ArrayList< RandomAccess<FloatType> > eCursors = new ArrayList< RandomAccess<FloatType> >();

		int numberOfDimensions = input.numDimensions();

		/* Create an eigenvalue images and a cursor for each */

		for( int i = 0; i < numberOfDimensions; ++i ) {
			Img<FloatType> eigenvalueImage = floatFactory.create( input, new FloatType() );
			eigenvalueImages.add( eigenvalueImage );
			eCursors.add( eigenvalueImage.randomAccess() );
		}

		Matrix hessian = new Matrix( numberOfDimensions, numberOfDimensions );

		/* Two cursors for finding points around the point of
		   interest, used for calculating the second
		   derivatives at that point: */
		RandomAccess<T> ahead = Views.extendMirrorSingle( input ).randomAccess();
		RandomAccess<T> behind = Views.extendMirrorSingle( input ).randomAccess();

		ReverseAbsoluteFloatComparator c = new ReverseAbsoluteFloatComparator();

		while( cursor.hasNext() ) {

			cursor.fwd();

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

			// Now find the eigenvalues and eigenvalues of the Hessian matrix:
			EigenvalueDecomposition e = hessian.eig();

			/* Nonsense involved in sorting this array of
			   eigenvalues by their absolute values: */

			double [] eigenvaluesArray = e.getRealEigenvalues();
			ArrayList<Float> eigenvaluesArrayList = new ArrayList<Float>(eigenvaluesArray.length);
			for( double ev : eigenvaluesArray )
				eigenvaluesArrayList.add(new Float(ev));

			Collections.sort( eigenvaluesArrayList, c );

			// Set the eigenvalues at the point of interest in each image:

			for( int i = 0; i < numberOfDimensions; ++i ) {
				RandomAccess<FloatType> eCursor = eCursors.get(i);
				eCursor.setPosition(cursor);
				eCursor.get().set( eigenvaluesArrayList.get(i).floatValue() );
			}
		}

		return eigenvalueImages;
	}

	public void run( String ignored ) {

		ImagePlus imagePlus = IJ.getImage();
		if( imagePlus == null ) {
			IJ.error("There's no image open to work on.");
			return;
		}

		float realSigma = 2;

		image = (Img< T >)ImagePlusAdapter.wrap( imagePlus );

		float[] spacing = new float[ image.numDimensions() ];
		
		spacing[ 0 ] = (float)imagePlus.getCalibration().pixelWidth;
		spacing[ 1 ] = (float)imagePlus.getCalibration().pixelHeight;

		if ( spacing.length > 2 )
			spacing[ 2 ] = (float)imagePlus.getCalibration().pixelDepth;

		double [] sigmas = new double[spacing.length];
		for( int i = 0; i < spacing.length; ++i )
			sigmas[i] = 1.0 / (double)spacing[i];

		final Img< T > result = image.factory().create( image, image.firstElement() );
		try
		{
			Gauss3.gauss( sigmas, Views.extendMirrorSingle( image ), result );
		}
		catch (IncompatibleTypeException e) { e.printStackTrace(); }

		ImagePlus imp = null;
		
		if ( result instanceof ImagePlusImg )
			try { imp = ((ImagePlusImg)result).getImagePlus(); } catch (ImgLibException e) {}

		if ( imp == null )
			imp = ImageJFunctions.wrap( result, "Blurred" ).duplicate();
		else
			imp.setTitle( "Blurred" );

		imp.show();

		ArrayList< Img<FloatType> > eigenvalueImages = hessianEigenvalueImages(result,spacing);

		for( Img<FloatType> resultImage : eigenvalueImages )
		{
			imp = null;

			if ( result instanceof ImagePlusImg )
				try { imp = ((ImagePlusImg)resultImage).getImagePlus(); } catch (ImgLibException e) {}

			if ( imp == null )
				imp = ImageJFunctions.wrap( resultImage, "EigenValues" ).duplicate();
			else
				imp.setTitle( "EigenValues" );
		}
	}
}