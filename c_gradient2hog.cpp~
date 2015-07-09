#include "mex.h"
#define _USE_MATH_DEFINES // <-- enables to use M_PI in math.h
#include <math.h>   // <-- ceil, floor, sqrt, fabs, M_PI, atan2 are defined here 
#include <float.h>  // <-- DBL_EPSILON is defined here 
#include <assert.h> // for ASSERTE
#include <crtdbg.h>

/***************************************************************************************/
void BilinearWeights(const double x, const double xStart, const double bw, double& leftWeight, double& rightWeight) {
	// center of bin N                  center of bin N+1  
	//        |    leftWeight        rightWeight|
	//        |---------------------|-----------|
	//      xStart                  x       xStart+bw

	leftWeight  = 1-((x-xStart)/(double)bw);
	rightWeight = (x-xStart)/(double)bw;
}
/***************************************************************************************/
// bin0=(0..bw), bin1=(bw..2bw), ...
void BilinearWeights2(const double x, const double bw, const int noOfBins, int& leftBin, int& rightBin, double& leftWeight, double& rightWeight) {
    // center bin:
    int    x2id = (int)(x/bw);
    double x2center = x2id * bw + bw/2.0;    
    // left bin:
    int    x1id = x2id - 1;
		if (x1id < 0) { x1id = noOfBins - 1;}
    double x1center =x1id * bw  + bw/2;    
    // right bin:
    int    x3id =x2id + 1;
		if (x3id >= noOfBins) { x3id = 0; }
		double x3center = x3id * bw  + bw/2.0;    
    double dist = fabs(x2center - x);
    if (x < x2center ) { //use left and current bin	
			leftWeight = dist/bw;
			rightWeight = 1 - leftWeight;
			leftBin = x1id;
			rightBin = x2id;
    } else { // use current and right bin
			rightWeight = dist/bw;
			leftWeight = 1 - rightWeight;
			leftBin = x2id;
			rightBin = x3id;
    }    
}
//**************************************************************************************************
double round(double number) {
    return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}
//**************************************************************************************************
void CalculateSpatialGaussWindow(double* gauss, const int gSize, const double sigma) {
  double mju= (gSize-1)/2.0;
  int pos=0;
  for (int c=0; c<gSize; c++) {
    for (int r=0; r<gSize; r++) {
      *(gauss+pos)= exp(  -((r-mju)*(r-mju) + (c-mju)*(c-mju)) / (double)(2*sigma*sigma));
      pos++;
    }
  }
}
//**************************************************************************************************
void c_gradient2hog(double* re, double* im, mxArray* resArr, int m, int n, int noOfBins, int maxDegr, 
		    int cellSize, int useGauss, int sigmaPix, int useBilinearWeights,
		    int returnMagDegArrays, mxArray **magReturn, mxArray **degReturn) {
  int subs[3];
  double *magnitude, *degree;
  double dx, dy;
  int pos, i, j;
  double binBw = maxDegr/(double)noOfBins;
  double* gauss;
  if (useGauss) {
    gauss = (double*) mxCalloc (cellSize*cellSize, sizeof(double));
    CalculateSpatialGaussWindow(gauss, cellSize, sigmaPix);
  }
    
  mxArray *magnitudeArr = mxCreateDoubleMatrix(m, n, mxREAL);
  mxArray *degreeArr    = mxCreateDoubleMatrix(m, n, mxREAL);
  magnitude = mxGetPr(magnitudeArr);
  degree    = mxGetPr(degreeArr);
  // calculate mag and deg:
  pos=0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      dx = *(re+pos);
      dy = *(im+pos);      
      double mag = sqrt(dx*dx + dy*dy);
      *(magnitude+pos) = mag;
      if (fabs(mag)> 2*DBL_EPSILON) { //+90deg->edge perpendicular to the slope, +360 to avoid negative:
	*(degree+pos) = ((int)round(180.0/M_PI*atan2(dy, dx)) + 90 + 360) % maxDegr; 
      }     
      pos++;
    }
  }

  // Prepare arrays and bin indexes:
  double* res = mxGetPr(resArr); // multi dimensional array (m x n x noOfBins) for storing the result
  // Calculate start of each dimension:
  int* binIdx = (int*)mxMalloc(sizeof(int)*noOfBins); 
  subs[0]=0; subs[1]=0;
  for (int bin=0; bin<noOfBins; bin++) {
    subs[2]=bin;
    binIdx[bin] = mxCalcSingleSubscript(resArr, 3, subs);
  }

  // For each pixel calculate hog:
  pos=0;
  int wa = cellSize/2;
  int wb = cellSize-wa;
  for (int c = 0; c < n; c++) {
    for (int r = 0; r < m; r++) {      
      // Scan neibourhood of the pixel:      
      int posG=0;
      //mexPrintf("%d,%d) ===================\n", r,c);
      for (int cc = c-wa; cc < c+wb; cc++) {
	for (int rr = r-wa; rr < r+wb; rr++) {
	  if (rr < 0 || cc <0 || rr >= m || cc>= n)  continue; // out of image
	  int pos2 = cc * m + rr;
	  int bin = (int)( (*(degree+pos2))/binBw );	  
	  if (bin<0 || bin >= noOfBins) 
	    mexErrMsgIdAndTxt("MW:c_gradient2hog", "c_gradient2hog: bin out of range %d", bin);   
	  if (*(magnitude+pos2) > 2*DBL_EPSILON) {
	    if (!useBilinearWeights) {
	      if (!useGauss) {
		*(res+binIdx[bin]+pos) += *(magnitude+pos2); // add mag to histogram
	      } else {
		*(res+binIdx[bin]+pos) += *(magnitude+pos2) * (*(gauss+posG)); // add mag*gauss to histogram
		//mexPrintf("  %d,%d) %lf\n", rr, cc, *(gauss+posG));
	      }	    
	    } else { // use bilinear weights:
	      int bin1, bin2;
	      double leftWeight, rightWeight, val1, val2;
	      BilinearWeights2(*(degree+pos2), binBw, noOfBins, bin1, bin2, leftWeight, rightWeight);
	      val1 = *(magnitude+pos2) * leftWeight;
	      val2 = *(magnitude+pos2) * rightWeight;
	      *(res+binIdx[bin1]+pos) += val1;
	      *(res+binIdx[bin2]+pos) += val2;
	    }
	  }
	  posG++;
	}
      }      
      pos++;
    }
  }
  mxFree(binIdx);  


  if (returnMagDegArrays) { // do not destroy, they are returned to matlab:
    *magReturn = magnitudeArr;
    *degReturn = degreeArr;
  } else { // destroy, they are not needed:
    mxDestroyArray(magnitudeArr);
    mxDestroyArray(degreeArr);
  }
  if (useGauss) mxFree(gauss);  
  _ASSERTE( _CrtCheckMemory( ) );		



}

//**************************************************************************************************
/* The gateway routine */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
#define NO_OF_INPUTS 7
  double  *re;
  double  *im;
  double  *res;
  int m, n;
  int nsubs; 
  int dims[3];
  int i,j;
  int noOfBins, maxDegr, cellSize, useGauss;
  int sigmaPix;
  int useBilinearWeights;
  
  // Check input data:
  if (nrhs != NO_OF_INPUTS) 
    mexErrMsgIdAndTxt( "c_gradient2hog:invalidNumInputs", "% d inputs required, usage: c_gradient2hog(complex_gradient, noOfBins, maxDegr, cellSize, useGauss, sigmaPix, useBilinearWeights)", NO_OF_INPUTS);

  if (!mxIsComplex(prhs[0])) mexErrMsgTxt("First input argument must be a complex array.");   
  m = mxGetM(prhs[0]);
  n = mxGetN(prhs[0]);
  nsubs = mxGetNumberOfDimensions(prhs[0]);
  if (nsubs!=2) mexErrMsgTxt("Array must have 2 dimensions.");
  if (m<=1 || n <=1) mexErrMsgTxt("Array must have size >2");
  for (i=1; i<NO_OF_INPUTS; i++) {
    if (mxIsComplex(prhs[i])) mexErrMsgTxt("input argument must be real (not complex).");     
    if (mxGetM(prhs[i])!=1 || mxGetN(prhs[i])!=1) mexErrMsgTxt("input argument must be single value."); 
  }
  noOfBins = (int)*(mxGetPr(prhs[1]));
  maxDegr  = (int)*(mxGetPr(prhs[2])); 
  cellSize = (int)*(mxGetPr(prhs[3]));
  useGauss = (int)*(mxGetPr(prhs[4]));
  sigmaPix = (int)*(mxGetPr(prhs[5]));
  useBilinearWeights = (int)*(mxGetPr(prhs[6]));
  // Check output data:
  if (nlhs > 3) mexErrMsgIdAndTxt( "c_gradient2hog:maxlhs", "Too many output arguments.");

  re = mxGetPr(prhs[0]);
  im = mxGetPi(prhs[0]);

  // Create output matrix m x n x noOfBins 
  dims[0]= m; 
  dims[1]= n;
  dims[2]= noOfBins;
  plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);


  int returnMagDegArrays = (nlhs == 3) ? 1 : 0;
  c_gradient2hog(re, im, plhs[0] , m, n, noOfBins, maxDegr, cellSize, useGauss, sigmaPix, useBilinearWeights,
		 returnMagDegArrays, &(plhs[1]), &(plhs[2]));
}	

//**************************************************************************************************
