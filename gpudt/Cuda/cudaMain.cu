/*
Author: Cao Thanh Tung
Date: 15/03/2011

File Name: cudaMain.cu

This file contains the main CUDA code including all initialization and so on. 

===============================================================================

Copyright (c) 2011, School of Computing, National University of Singapore. 
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/delaunay.html

If you use GPU-DT and you like it or have comments on its usefulness etc., we 
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the National University of University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission from the National University of Singapore. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

*/

#pragma warning(disable: 4311 4312)

#include <device_functions.h>
#include <stdio.h>
#include <string.h>
#include "../gpudt.h"
#include "cuda.h"
#include "common.h"

#include "cudaCCW.h"
/***********************************************************
 * Declarations
 ***********************************************************/

#define WBLOCK                256        
#define ROUND_TO_BLOCK(x)    (((x) / WBLOCK + 1) * WBLOCK)

// Convert a triangle index to the pointer to its structure
#define index_to_pointer(t) \
    (((t) < 0) ? 0 : (gpudtTriangles + ((t) >> 2) * TSize) | \
                     (unsigned int) ((t) & 3))

REAL hostConst[13];         // Constants used in exact CUDA CCW test

bool initialized = false; 

short2 **pingpongColor, **pingpongPattern;   

int *voronoiPtr;            // Discrete voronoi diagram
short2 *patternPtr;         // Voronoi vertex patterns
int nTris;                  // Total number of sites and triangles generated    
int *ctriangles;            // Triangle list
int *cvertarr;              // Vertex array
int *tvertices;             // mapToId array, indicating the index of the pixel that
                            // was actually mapped to the position of the current pixel. 
REAL2 *cvertices;           // Shifted coordinates of the sites
REAL2 *covertices;          // Original coordinates of the sites    
int *cconstraints;          // Constraints

int *cflag;        

short *cnewtri;             // Mark the empty slots in our triangle list
int step = 0;               // Shared between multiple step to work on the cnewtri array
int sizeTexture; 

REAL scale, shiftX, shiftY; 

// Texture reference for the triangle list
texture<int, 1, cudaReadModeElementType> texTri; 

/**************************************************************
 * Exported methods
 **************************************************************/
extern "C" void cudaExactInit(); 
extern "C" void cudaInitialize(); 
extern "C" void cudaFinalize(); 
extern "C" void cudaAllocation(); 
extern "C" void cudaDeallocation(); 


/************************************************************
 * Variables and functions shared with the main GPUDT module
 ************************************************************/
extern int *boundary;
extern gpudtVertex *gpudtVertices; 
extern PGPUDTPARAMS gpudtParams; 
extern PGPUDTOUTPUT gpudtOutput; 
extern int nStack; 
extern int *gpudtStack; 
extern int nPoints, nVerts; 

extern int gpudtFixConvexHull(int *additionalTriangles, int fboWidth);

/********************************
 * MODULES
 ********************************/

/*********************************************************************************
 * We use index to reference triangles and vertices in CUDA.
 * In CPU we use pointer to point directly to their structure. 
 * We need to perform a fast conversion here. 
 *********************************************************************************/
__global__ void kernelIndicesToPointers(int *ctriangles, int gpudtTriangles, 
                                        int TSize, int VSize, int nTris) {
    __shared__ int ct[WBLOCK * 9]; 

    int tId = threadIdx.x; 
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x, x9 = x * 9;
    int i, id; 
            
    if (x >= nTris)
        return ;

    // Cooperatively read all triangles processed by one block
    for (i = 0, id = tId; i < 9; i++, id += WBLOCK)
        ct[id] = ctriangles[x9 + id]; 

    __syncthreads(); 
    
    i = tId * 9;         
    ct[i] = index_to_pointer(ct[i]); i++; 
    ct[i] = index_to_pointer(ct[i]); i++; 
    ct[i] = index_to_pointer(ct[i]); i++; 
  
    __syncthreads(); 

    for (i = 0, id = tId; i < 9; i++, id += WBLOCK)
        ctriangles[x9 + id] = ct[id]; 
}

/************************************************************
 * Map continuous coordinates to discrete coordinates
 ************************************************************/
__global__ void kernelMapping(REAL2 *covertices, REAL2 *cvertices, REAL scale, 
                              REAL shiftX, REAL shiftY, int nVerts, int nPoints) {
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; 

    if (x >= nVerts)
        return ; 

    REAL2 t = covertices[x]; 

    if (x < nPoints) 
    {
        t.x = floor((t.x-shiftX) / scale); 
        t.y = floor((t.y-shiftY) / scale); 
    }

    cvertices[x] = t; 
}

//***************************************************************************************
//*                                                                                        *
//*  exactinit()   Initialize the variables used for exact arithmetic.                  *
//*                                                                                     *
//*  `cuda_epsilon' is the largest power of two such that 1.0 + cuda_epsilon = 1.0 in   *
//*  floating-point arithmetic.  `cuda_epsilon' bounds the relative roundoff            *
//*  error.  It is used for floating-point error analysis.                              *
//*                                                                                     *
//*  `cuda_splitter' is used to split floating-point numbers into two half-             *
//*  length significands for exact multiplication.                                      *
//*                                                                                     *
//*  I imagine that a highly optimizing compiler might be too smart for its             *
//*  own good, and somehow cause this routine to fail, if it pretends that              *
//*  floating-point arithmetic is too much like double arithmetic.                      *
//*                                                                                     *
//*  Don't change this routine unless you fully understand it.                          *
//*                                                                                     *
//***************************************************************************************

__global__ void kernel_exactinit(REAL *constants)
{
    REAL half;
    REAL check, lastcheck;
    REAL cuda_epsilon = 1.0; 
    REAL cuda_splitter = 1.0; 
    int every_other;

    every_other = 1;
    half = 0.5;
    check = 1.0;
    // Repeatedly divide `cuda_epsilon' by two until it is too small to add to      
    //   one without causing roundoff.  (Also check if the sum is equal to     
    //   the previous sum, for machines that round up instead of using exact   
    //   rounding.  Not that these routines will work on such machines.)       
    do {
        lastcheck = check;
        cuda_epsilon *= half;
        if (every_other) {
            cuda_splitter *= 2.0;
        }
        every_other = !every_other;
        check = 1.0 + cuda_epsilon;
    } while ((check != 1.0) && (check != lastcheck));
    constants[0] = cuda_splitter + 1.0;
    constants[1] = cuda_epsilon;
    /* Error bounds for orientation and incircle tests. */
    constants[2]/*cuda_resulterrbound*/ = (3.0 + 8.0 * cuda_epsilon) * cuda_epsilon;
    constants[3]/*cuda_ccwerrboundA*/ = (3.0 + 16.0 * cuda_epsilon) * cuda_epsilon;
    constants[4]/*cuda_ccwerrboundB*/ = (2.0 + 12.0 * cuda_epsilon) * cuda_epsilon;
    constants[5]/*cuda_ccwerrboundC*/ = (9.0 + 64.0 * cuda_epsilon) * cuda_epsilon * cuda_epsilon;
    constants[6]/*cuda_iccerrboundA*/ = (10.0 + 96.0 * cuda_epsilon) * cuda_epsilon;
    constants[7]/*cuda_iccerrboundB*/ = (4.0 + 48.0 * cuda_epsilon) * cuda_epsilon;
    constants[8]/*cuda_iccerrboundC*/ = (44.0 + 576.0 * cuda_epsilon) * cuda_epsilon * cuda_epsilon;
    constants[9]/*cuda_o3derrboundA*/ = (7.0 + 56.0 * cuda_epsilon) * cuda_epsilon;
    constants[10]/*cuda_o3derrboundB*/ = (3.0 + 28.0 * cuda_epsilon) * cuda_epsilon;
    constants[11]/*cuda_o3derrboundC*/ = (26.0 + 288.0 * cuda_epsilon) * cuda_epsilon * cuda_epsilon;

    constants[12] = 1.0 / sin(REAL(0.0)); // infinity
}

/************************************************************
 * Common methods
 ************************************************************/
void cudaAllocation() 
{
    // Allocate memory
    cutilSafeCall( cudaMalloc((void **) &cflag, sizeof(int)) ); 
    
    pingpongColor = (short2 **) malloc(2 * sizeof(short2 *)); 
    pingpongPattern = (short2 **) malloc(2 * sizeof(short2 *)); 

    sizeTexture = gpudtParams->fboSize * gpudtParams->fboSize * sizeof(short2); 

    // Allocate 2 textures
    cutilSafeCall( cudaMalloc((void **) &pingpongColor[0], sizeTexture) ); 
    cutilSafeCall( cudaMalloc((void **) &pingpongPattern[0], sizeTexture) ); 
    cutilSafeCall( cudaMalloc((void **) &pingpongColor[1], sizeTexture) ); 
    //cutilSafeCall( cudaMalloc((void **) &pingpongPattern[1], sizeTexture) ); 

    // We allocate one memory chunk to store the triangle list.
    // However, we temporarily use it to store two pingpong buffers. 
    // These two will no longer needed when we start constructing the mesh.
    int ctrianglesSize = ROUND_TO_BLOCK(nVerts) * 18 * sizeof(int); 
    char *ptr; 

    cutilSafeCall( cudaMalloc((void **) &ptr, max(ctrianglesSize, sizeTexture)) ); 
    ctriangles = (int *) ptr; 
    //pingpongColor[1] = (short2 *) ptr; 
    pingpongPattern[1] = (short2 *) (ptr/* + sizeTexture*/); 

    // Allocate the rest
    cutilSafeCall( cudaMalloc((void **) &covertices, nVerts * 2 * sizeof(REAL)) );
	cutilSafeCall( cudaMalloc((void **) &cvertices, nVerts * 2 * sizeof(REAL)) ); 

    
}

void cudaDeallocation() 
{
    // Release memory
    cutilSafeCall( cudaFree(covertices) ); 
    cutilSafeCall( cudaFree(ctriangles) ); 
	cutilSafeCall( cudaFree(pingpongColor[0]) ); 
	cutilSafeCall( cudaFree(pingpongPattern[0]) ); 
    cutilSafeCall( cudaFree(cflag) ); 

	free(pingpongColor); 
	free(pingpongPattern); 
}

void cudaInitialize() 
{  
	cutilSafeCall( cudaMalloc((void **) &tvertices, nVerts * sizeof(int)) ); 

    // Upload site data to GPU, use the scaled coordinates
    cutilSafeCall( cudaMemcpy(covertices, gpudtVertices, nVerts * 2 * sizeof(REAL), cudaMemcpyHostToDevice) ); 

    // Scale the points
    // Record mapped coordinates into cvertices 
    // The shifting process will be performed on this array
    dim3 block = dim3(WBLOCK); 
    dim3 grid = dim3(STRIPE, nVerts / (STRIPE * block.x) + 1); 
    kernelMapping<<< grid, block >>>(covertices, cvertices, scale, shiftX, shiftY, nVerts, nPoints); 
    cutilCheckError(); 
}

void cudaFinalize() {
    cutilSafeCall( cudaMemcpy(gpudtOutput->triangles, ctriangles, 9 * nTris * sizeof(int), cudaMemcpyDeviceToHost) ); 

    // Return the number of triangles generated
    gpudtOutput->nTris = nTris; 
}

void cudaExactInit()
{
    REAL *constants; 

    cutilSafeCall( cudaMalloc((void **)&constants, 13 * sizeof(REAL)) ); 

    kernel_exactinit<<< 1, 1 >>>(constants); 
    cutilCheckError(); 

    cutilSafeCall( cudaMemcpy(hostConst, constants, 13 * sizeof(REAL), cudaMemcpyDeviceToHost) ); 

//    for (int i = 0; i < 13; i++)
//        printf("%.30f\n", hostConst[i]); 

    cutilSafeCall( cudaFree(constants) ); 
}
