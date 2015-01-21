/*
Author: Cao Thanh Tung
Date: 15/03/2011

File Name: cudaReconstruction.cu

This file include all CUDA code to perform the reconstruction step

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
#define SBLOCK                128
#define MBLOCK                32

/**************************************************************
 * Exported methods
 **************************************************************/
extern "C" void cudaReconstruction();

/**************************************************************
 * Definitions
 **************************************************************/
#define MAX(a, b)            (((a) > (b)) ? (a) : (b))
#define SHIFT                5

// Buffer to be used when constructing vertex array in CPU
#define BUFFER_ITEM        910
#define BUFFER_SIZE        BUFFER_ITEM * 9        // Buffer 32K

// Decode an oriented triangle. 
// An oriented triangle consists of 32 bits. 
// - 30 highest bits represent the triangle index, 
// - 2 lowest bits represent the orientation (the starting vertex, 0, 1 or 2)
#define decode_tri(x)    ((x) >> 2)
#define decode_ori(x)    ((x) & 3)

__constant__ int minus1mod3[4] = {2, 0, 1, -1};
__constant__ int plus1mod3[4] = {1, 2, 0, -1};

// Mark those shifted sites (either can or cannot)
// Shared with the Shifting stage. 
int *cactive;                
/* cactive is used in the shifting stage. 
 * The code used in cactive is as follow: 
 *         0        : Missing sites
 *        -1        : Boundary
 *        -2, 2    : Marked during the reconstruction stage
 *        -x, x   : Different steps in the shifting detection algorithm. 
 *      +inf    : Unprocessed
 *
 * Any value > 0 indicate that that vertex can be shifted safely. 
 */

/************************************************************
 * Variables and functions shared with the main module
 ************************************************************/
extern int *voronoiPtr;        // Discrete voronoi diagram
extern short2 *patternPtr;        // Voronoi vertex patterns

extern int nTris, nVerts, nPoints;    
extern int *ctriangles;        
extern int *cvertarr;        
extern int *tvertices; 
extern REAL2 *cvertices;    
extern REAL2 *covertices;    
extern PGPUDTPARAMS gpudtParams;    
extern int *cactive;                

extern int sizeTexture; 
extern REAL scale, shiftX, shiftY; 

extern int *boundary;
extern gpudtVertex *gpudtVertices; 
extern gpudtTriangle *gpudtTriangles;

extern int gpudtFixConvexHull(int *additionalTriangles, int fboWidth, int *boundary);

/*********************************************************************************
 * Count the number of triangle generated for each row of the texture. 
 * Used to calculate the offset to which each thread processing a texture row 
 * will insert the generated triangles.
 * Also, collect the boundary pixels of the texture to be used in the next CPU step
 *********************************************************************************/
__global__ void kernelCountRow(int *voronoiPtr, short2 *patternPtr, int *count, int width, int min, int max, int *cboundary) {
    // Get the row we are working on
    int x = blockIdx.x * blockDim.x + threadIdx.x; 

    // Collect the boundary (up, left, down, right)
    if (x > 0 && x <= max) {
        cboundary[width * 0 + x] = voronoiPtr[min * width + x]; 
        cboundary[width * 1 + x] = voronoiPtr[x * width + min]; 
        cboundary[width * 2 + x] = voronoiPtr[max * width + x]; 
        cboundary[width * 3 + x] = voronoiPtr[x * width + max]; 
    }

    // Actual counting
    if (x < min || x >= max)
        return ;  

    int xwidth = x * width; 
    int result = 0;
    short2 t = patternPtr[xwidth + min]; 
    
    // Keep jumping and counting
    while (t.y > 0 && t.y < max) {
        result += 1 + (t.x >> 2); 
        t = patternPtr[xwidth + t.y + 1];
    }

    count[x] = result; 
}

/*********************************************************************************
 * Prefix sum on the counted value to calculate the offset
 *********************************************************************************/
void cudaPrefixSum(int *cpuCount, int min, int max) {
    cpuCount[min-1] = 0; 
    for (int i = min; i < max; i++)
        cpuCount[i] += cpuCount[i-1]; 
}

/*********************************************************************************
 * Generate triangles from the Voronoi vertices and insert them into the triangle list.
 *********************************************************************************/
__global__ void kernelGenerateTriangles(int *voronoiPtr, short2 *patternPtr, int3 *ctriangles, 
										int *offset, int width, int min, int max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 

    if (x < min || x >= max)
        return ;  

    int xwidth = x * width; 
    short2 pattern = patternPtr[xwidth + min];
    int i0, i1, i2, i3;
    int3 *pT = &ctriangles[offset[x-1]]; 

    // Jump through all voronoi vertices in a texture row
    while (pattern.y > 0 && pattern.y < max) {
        i0 = voronoiPtr[xwidth + pattern.y]; 
        i1 = voronoiPtr[xwidth + pattern.y + 1]; 
        i2 = voronoiPtr[xwidth + width + pattern.y + 1];
        i3 = voronoiPtr[xwidth + width + pattern.y]; 

        if (pattern.x == 0) *pT = make_int3(i3, i1, i2); 
        if (pattern.x == 1) *pT = make_int3(i0, i2, i3); 
        if (pattern.x == 2) *pT = make_int3(i1, i3, i0); 
        if (pattern.x == 3) *pT = make_int3(i2, i0, i1); 
        if (pattern.x == 4) {
            // Generate 2 triangles. 
            // Since the hole is convex, no need to do CCW test
            *pT = make_int3(i2, i0, i1); pT++; 
            *pT = make_int3(i3, i0, i2); 
        }
        
        pattern = patternPtr[xwidth + pattern.y + 1]; 
        pT++; 
    }
}

/************************************************************
 * Scale back the point set
 ************************************************************/
__global__ void kernelScaleBack(REAL2 *cvertices, REAL scale, REAL shiftX, REAL shiftY, int nPoints) {
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; 

    if (x >= nPoints)
        return ; 

    REAL2 t = cvertices[x]; 

    t.x = shiftX + t.x * scale; 
    t.y = shiftY + t.y * scale; 

    cvertices[x] = t; 
}

/*********************************************************************************
 * Map all sites to its ID, including missing sites
 *********************************************************************************/
__global__ void kernelMapToId(int *voronoiPtr, int nVerts, REAL2 *cvertices, 
                              int *tvertices, int width) {
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; 

    if (x >= nVerts)
        return ; 

    REAL2 t = cvertices[x]; 
    tvertices[x] = voronoiPtr[(int(t.y) + 1) * width + (int(t.x) + 1)];
}

/*********************************************************************************
 * Check all triangles to detect possible shifting of their sites
 *********************************************************************************/
__global__ void kernelDetectShifting(int nTris, int *ctriangles, int *cactive, BYTE *cprocessed, 
                                     BYTE *ccanshift, REAL2 *covertices, REAL2 *cvertices, BYTE step) {
    __shared__ int ct[SBLOCK * 3]; 

    int tId = threadIdx.x; 
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x, x3 = x * 3;
            
    if (x >= nTris)
        return ;

    // Cooperatively read all triangles processed by one block
    ct[tId] = ctriangles[x3 + tId]; tId += SBLOCK; 
    ct[tId] = ctriangles[x3 + tId]; tId += SBLOCK; 
    ct[tId] = ctriangles[x3 + tId];

    __syncthreads(); 

    tId = threadIdx.x; 

    if (x + tId >= nTris)
        return ; 

    int p0, p1, p2, c0, c1, c2; 
    REAL2 vtx0, vtx1, vtx2, cx0, cx1, cx2; 
    
    x3 = tId * 3; 
    p0 = ct[x3 + 0];  p1 = ct[x3 + 1];  p2 = ct[x3 + 2]; 
    c0 = cactive[p0]; c1 = cactive[p1]; c2 = cactive[p2]; 

    if (c0 <= 2 && c1 <= 2 && c2 <= 2)    // All are processed
        return ; 

    vtx0 = covertices[p0]; vtx1 = covertices[p1]; vtx2 = covertices[p2]; 

    if (c0 == 2) cx0 = vtx0; else cx0 = cvertices[p0];
    if (c1 == 2) cx1 = vtx1; else cx1 = cvertices[p1]; 
    if (c2 == 2) cx2 = vtx2; else cx2 = cvertices[p2]; 

    // Process p0
    if (c0 > 2)
        if ((p1 < p0 && c1 > 2) || (p2 < p0 && c2 > 2)) 
            cprocessed[p0] = step;    // This vertex is not processed in this step
        else
            if (cuda_ccw(cx1, cx2, vtx0) <= 0) 
                ccanshift[p0] = step;    // Cannot shift this vertex, there is a crossing

    // Process p1
    if (c1 > 2)
        if ((p2 < p1 && c2 > 2) || (p0 < p1 && c0 > 2)) 
            cprocessed[p1] = step;    // This vertex is not processed in this step
        else
            if (cuda_ccw(cx2, cx0, vtx1) <= 0) 
                ccanshift[p1] = step;    // Cannot shift this vertex, there is a crossing

    // Process p2
    if (c2 > 2) 
        if ((p0 < p2 && c0 > 2) || (p1 < p2 && c1 > 2)) 
            cprocessed[p2] = step;    // This vertex is not processed in this step
        else
            if (cuda_ccw(cx0, cx1, vtx2) <= 0) 
                ccanshift[p2] = step;    // Cannot shift this vertex, there is a crossing
}

/*********************************************************************************
 * Process the output of the shifting detection step
 * Record result into cactive array
 * Use label -1 and 1 for this step
 *********************************************************************************/
__global__ void kernelMergeShiftingDetectionResult(int nVerts, int *cactive, BYTE *cprocessed, 
                                                   BYTE *ccanshift, REAL2 *cvertices, 
                                                   REAL2 *covertices, BYTE step, int nPoints) {
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x; 

    if (x >= nVerts)
        return ; 

    int ca = cactive[x]; 
    
    if (ca > 2) {    // no result yet
        if (x >= nPoints) {
            cactive[x] = 2; 
            return ; 
        }

        BYTE cp = cprocessed[x]; 
        
        if (cp != step)    {    // processed in this step
            BYTE cc = ccanshift[x]; 

            if (cc == step)        // Marked in this step: cannot shift
                cactive[x] = -2; 
            else {
                cactive[x] = 2;        // Can shift
                cvertices[x] = covertices[x]; 
            }
        }
    }
}

/*********************************************************************************
 * Vertex array calculation done in CPU. 
 * Each triangle is linked with 3 other triangles sharing a common vertex.
 * Block processing to utilize CPU cache. 
 *********************************************************************************/
void cpuVertexArray(int *triangles, int nVerts, int nTris, int *vertArr) {
    int buffer[BUFFER_SIZE]; 
    int last = nTris * 9, ptr = BUFFER_SIZE - 9; 

    memset(vertArr, 255, nVerts * sizeof(int)); 
    memset(buffer, 255, BUFFER_SIZE * sizeof(int)); 

    for (int i = nTris-1, i3 = i * 3; i >= 0; i--, i3 -= 3) {
        buffer[ptr + 3] = triangles[i3]; 
        buffer[ptr + 4] = triangles[i3+1]; 
        buffer[ptr + 5] = triangles[i3+2]; 
        buffer[ptr + 6] = vertArr[buffer[ptr + 4]]; vertArr[buffer[ptr + 4]] = i << 2; 
        buffer[ptr + 7] = vertArr[buffer[ptr + 5]]; vertArr[buffer[ptr + 5]] = (i << 2) | 1; 
        buffer[ptr + 8] = vertArr[buffer[ptr + 3]]; vertArr[buffer[ptr + 3]] = (i << 2) | 2; 
        ptr -= 9; 

        if (ptr < 0) {
            memcpy(&triangles[last - BUFFER_SIZE], buffer, BUFFER_SIZE * sizeof(int)); 
            last -= BUFFER_SIZE; ptr = BUFFER_SIZE - 9; 
        }
    }
    
    if (ptr < BUFFER_SIZE - 9) {
        int left = BUFFER_SIZE - 9 - ptr; 
        memcpy(&triangles[last - left], &buffer[ptr + 9], left * sizeof(int)); 
    }
}

/*********************************************************************************
 * Find 3 neighbours sharing one edge with each triangle.
 *********************************************************************************/
__global__ void kernelFindNextTriangles(int *ctriangles, int nTris) {
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nTris) 
        return ; 

    int p1, p2;
    int nCounter, pNextTri, pTri, pOri, pTri9; 
    int x9 = x * 9;

    p2 = ctriangles[x9+3]; 
    p1 = ctriangles[x9+5]; 
    nCounter = 0; 

    // orientation 0
    // Travel through the list of triangles sharing vertex 0 with this triangle.
    // In this list we can find at most two triangles sharing edge (p0, p1) and 
    // (p2, p0) with our triangle. 
    pNextTri = ctriangles[x9+6];

    while (pNextTri >= 0 && nCounter < 2) {
        pTri = decode_tri(pNextTri); 
        pOri = decode_ori(pNextTri); 
        pTri9 = pTri * 9; 

        if (p2 == ctriangles[pTri9 + 3 + minus1mod3[pOri]]) {    // NextDest
            ctriangles[x9 + 2] = pNextTri; 
            ctriangles[pTri9 + pOri] = (x << 2) | 2;  
            nCounter++; 
        }

        if (p1 == ctriangles[pTri9 + 3 + pOri]) {    // NextApex
            ctriangles[x9 + 0] = (pTri << 2) | minus1mod3[pOri];  
            ctriangles[pTri9 + minus1mod3[pOri]] = (x << 2);  
            nCounter++; 
        }

        pNextTri = ctriangles[pTri9 + 6 + pOri]; 
    }

    // orientation 1
    // Find the triangle with edge (p1, p2)
    pNextTri = ctriangles[x9+7]; 

    while (pNextTri >= 0) {
        pTri = decode_tri(pNextTri); 
        pOri = decode_ori(pNextTri); 
        pTri9 = pTri * 9; 

        if (p2 == ctriangles[pTri9 + 3 + pOri]) {    // NextApex
            ctriangles[x9 + 1] = (pTri << 2) | minus1mod3[pOri]; 
            ctriangles[pTri9 + minus1mod3[pOri]] = (x << 2) | 1;  
            break ; 
        }

        pNextTri = ctriangles[pTri9 + 6 + pOri]; 
    }
}

//////////////////////////////////////////////////////////////

void cudaReconstruction() {
    int *count,                // Number of triangle generated in one texture row
        *cboundary;            // Boundary pixels of the texture

    dim3 grid, block; 

    /****************************************************************************************
     * Initialization
     ****************************************************************************************/
    int texSize    = gpudtParams->fboSize; 
    int min        = 1; 
    int max        = texSize - 2;    // Assume width = height
    
    // EXACT test constants
    cutilSafeCall( cudaMemcpyToSymbol(constData, hostConst, 13 * sizeof(REAL)) ); 

    /****************************************************************************************
     * CUDA: Count the number of Voronoi Vertices in each row of the texture
     ****************************************************************************************/
    cutilSafeCall( cudaMalloc((void **) &count, texSize * sizeof(int)) ); 
    cutilSafeCall( cudaMalloc((void **) &cboundary, texSize * 4 * sizeof(int)) ); 

    int *cpuCount = (int *) malloc(texSize * sizeof(int)); 
    int *boundary = (int *) malloc(texSize * 4 * sizeof(int)); 

    block = dim3(MBLOCK);    
    grid = dim3(texSize / block.x);
    kernelCountRow<<< grid, block >>>(voronoiPtr, patternPtr, count, texSize, min, max, cboundary);
    cutilCheckError(); 

    cutilSafeCall( cudaMemcpy(boundary, cboundary, texSize * 4 * sizeof(int), cudaMemcpyDeviceToHost) ); 
    cutilSafeCall( cudaMemcpy(cpuCount, count, texSize * sizeof(int), cudaMemcpyDeviceToHost) ); 

    /****************************************************************************************
     * CPU: Prefix sum
     ****************************************************************************************/

    cudaPrefixSum(cpuCount, min, max); 
    cutilSafeCall( cudaMemcpy(count, cpuCount, texSize * sizeof(int), cudaMemcpyHostToDevice) );
    nTris = cpuCount[max-1];

    free(cpuCount); 

    /****************************************************************************************
     * CUDA: Generate triangles
     ****************************************************************************************/
    // We use a very small block size here because there are 
    // very few texture rows, we want to fully utilize the multiprocessors.
    block = dim3(MBLOCK);
    grid = dim3(texSize / block.x);
    kernelGenerateTriangles<<< grid, block >>>(voronoiPtr, patternPtr, (int3 *) ctriangles, 
		count, texSize, min, max);
    cutilCheckError(); 

    block = dim3(WBLOCK);    
    grid = dim3(STRIPE, nPoints / (block.x * STRIPE) + 1); 
    kernelMapToId<<< grid, block >>>(voronoiPtr, nPoints, cvertices, tvertices, texSize); 
    cutilCheckError(); 

    // Fix the convex hull right here. 
    int boundaryTris = gpudtFixConvexHull((int *) gpudtTriangles + nTris * 3, texSize, boundary);

    cutilSafeCall( cudaMemcpy(gpudtTriangles, ctriangles, nTris * 3 * sizeof(int), cudaMemcpyDeviceToHost) );
    cutilSafeCall( cudaMemcpy(ctriangles + nTris * 3, (int *) gpudtTriangles + nTris * 3, boundaryTris * 3 * sizeof(int), cudaMemcpyHostToDevice) );

    nTris += boundaryTris; 

	int tris3Size = nTris * 3 * sizeof(int); 
    int trisSize = tris3Size * 3; 

    free(boundary); 
    cutilSafeCall( cudaFree(cboundary) ); 
    cutilSafeCall( cudaFree(count) ); 

    /****************************************************************************************
     * CUDA: Some CCW tests, Map sites to Id, construct vertex array
     ****************************************************************************************/
    // Scale the point set back
    block = dim3(WBLOCK); 
    grid = dim3(STRIPE, nPoints / (STRIPE * block.x) + 1); 
    kernelScaleBack<<< grid, block >>>(cvertices, scale, shiftX, shiftY, nPoints); 
    cutilCheckError(); 

    // Prepare some memory for this computation
    BYTE *cprocessed, *ccanshift; 

	cutilSafeCall( cudaMalloc((void **) &cvertarr, nVerts * sizeof(int)) ); 
	cutilSafeCall( cudaMalloc((void **) &cactive, nVerts * sizeof(int)) ); 
	cutilSafeCall( cudaMalloc((void **) &cprocessed, nVerts * sizeof(BYTE)) ); 
	cutilSafeCall( cudaMalloc((void **) &ccanshift, nVerts *sizeof(BYTE)) ); 

    cutilSafeCall( cudaMemset(cprocessed, 0, nVerts) ); 
    cutilSafeCall( cudaMemset(ccanshift, 0, nVerts) ); 
    cutilSafeCall( cudaMemset(cactive, 127, nVerts * sizeof(int)) ); 

    // Process multiple passes of CCW tests. Each site on average belongs to 6 triangles.
    for (BYTE step = 1; step < 7; step++) {
        block = dim3(SBLOCK); 
        grid = dim3(STRIPE, nTris / (block.x * STRIPE) + 1); 
        kernelDetectShifting<<< grid, block >>>(nTris, ctriangles, cactive, cprocessed, 
            ccanshift, covertices, cvertices, step);
	    cutilCheckError(); 

        block = dim3(WBLOCK); 
        grid = dim3(STRIPE, nVerts / (block.x * STRIPE) + 1); 
        kernelMergeShiftingDetectionResult<<< grid, block >>>(nVerts, cactive, cprocessed, 
            ccanshift, cvertices, covertices, step, nPoints); 
	    cutilCheckError(); 
    }

    // Calculate the vertex array at the same time
    int *vertarr = (int *) malloc(nVerts * sizeof(int)); 
    cpuVertexArray((int *) gpudtTriangles, nVerts, nTris, vertarr); 

    cutilCheckError(); 

    cutilSafeCall( cudaMemcpy(ctriangles, gpudtTriangles, trisSize, cudaMemcpyHostToDevice) ); 
    cutilSafeCall( cudaMemcpy(cvertarr, vertarr, nVerts * sizeof(int), cudaMemcpyHostToDevice) ); 

    free(vertarr); 
	cutilSafeCall( cudaFree(cprocessed)); 	
	cutilSafeCall( cudaFree(ccanshift)); 	

    /****************************************************************************************
     * Find next triangles
     ****************************************************************************************/
    block = dim3(WBLOCK); 
    grid = dim3(STRIPE, nTris / (STRIPE * block.x) + 1); 
    kernelFindNextTriangles<<< grid, block >>>(ctriangles, nTris); 
    cutilCheckError(); 

    /****************************************************************************************
     * Done!!!
     ****************************************************************************************/
}
