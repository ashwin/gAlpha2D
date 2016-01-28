/*
Author: Cao Thanh Tung
Date: 15/03/2011

File Name: cudaShifting.cu

This file include all CUDA code to perform the shifting step

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

Neither the name of the National University of Singapore nor the names of its contributors
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
#include "cudaScanLargeArray.h"

/***********************************************************
 * Declarations
 ***********************************************************/
#define WBLOCK                256        

// Texture reference for the triangle list
texture<int, 1, cudaReadModeElementType> texTri; 

#define INSERT_TRIANGLE(v0, v1, v2, tri) \
    tmp = tri; \
    ctriangles[tmp * 9 + 3] = v0; \
    ctriangles[tmp * 9 + 4] = v1; \
    ctriangles[tmp * 9 + 5] = v2; \
    ctriangles[tmp * 9 + 6] = atomicExch(&cvertarr[v1], (tmp << 2)); \
    ctriangles[tmp * 9 + 7] = atomicExch(&cvertarr[v2], (tmp << 2) | 1); \
    ctriangles[tmp * 9 + 8] = atomicExch(&cvertarr[v0], (tmp << 2) | 2); \
    cnewtri[tmp] = step

#define CONTAIN_KERNEL(org, dest, apex, vkernel) \
    (/*cuda_ccw((org), (dest), (vkernel)) >= 0 &&*/ \
     /*cuda_ccw((dest), (apex), (vkernel)) >= 0 &&*/ \
     cuda_ccw((apex), (org), (vkernel)) >= 0)

/**************************************************************
 * Exported methods
 **************************************************************/
extern "C" void cudaShifting(); 

/**************************************************************
 * Definitions
 **************************************************************/
// Decode an oriented triangle. 
// An oriented triangle consists of 32 bits. 
// - 30 highest bits represent the triangle index, 
// - 2 lowest bits represent the orientation (the starting vertex, 0, 1 or 2)
#define decode_tri(x)            ((x) >> 2)
#define decode_ori(x)            ((x) & 3)
#define encode_tri(tri, ori)    ((tri << 2) | ori)

/************************************************************
 * Variables and functions shared with the main module
 ************************************************************/
extern int nTris, nVerts, nPoints;                
extern int *ctriangles;            
extern int *cvertarr;            
extern int *tvertices; 
extern REAL2 *cvertices;        
extern REAL2 *covertices;        
extern int *cactive; 
extern short *cnewtri; 
extern int step; 
extern int *cflag; 

/*************************************************************
 * Detect all sites that can be shifted together 
 * without causing any crossing. 
 * We're guaranteed that all boundary sites are already marked
 *************************************************************/
__global__ void kernelShiftable(int *ctriangles, int *cvertarr, REAL2 *cvertices, 
                                REAL2 *covertices, int *cactive, 
                                int nVerts, int step, int *cflag) {
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nVerts || cactive[x] < step) 
        return ; 

    REAL2 vtx0, vtx1, vtx2 = covertices[x]; 
    int triStart, pNextTri, pTri, pOri, pTri9, id; 

    // Travel around the triangle fan
    pNextTri = cvertarr[x];

    pTri = decode_tri(pNextTri);
    pOri = decode_ori(pNextTri); 
    
    id = ctriangles[pTri * 9 + 3 + pOri]; 
    if (id < x && abs(cactive[id]) >= step)
        return ; 

    vtx1 = cvertices[id];  // Apex

    triStart = pTri; 

    do {
        pTri9 = pTri * 9; 

        id = ctriangles[pTri9 + 3 + (pOri + 2) % 3]; 
        if (id < x && abs(cactive[id]) >= step)        // Conflict
            return ; 

        vtx0 = cvertices[id];  // Dest

        if (cuda_ccw(vtx0, vtx1, vtx2) <= 0) {
            cactive[x] = -step; 
            *cflag = 1; 
            return;            // bad case
        }

        vtx1 = vtx0;  

        pNextTri = ctriangles[pTri9 + pOri]; 

        pTri = decode_tri(pNextTri);
        pOri = (decode_ori(pNextTri) + 1) % 3; 
    } while (pTri != triStart);
    
    *cflag = 1; 

    // Shift
    cactive[x] = step;        // Good case
    cvertices[x] = vtx2; 
}

/**************************************************************
 * Detect all missing sites
 **************************************************************/
__global__ void kernelMissingDetection(int nVerts, int *cactive, int *cvertarr) {
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    // Check for missing sites
    if (x < nVerts && cvertarr[x] < 0)
        cactive[x] = 0; 
}

/*************************************************************
 * Detect all shiftable bad cases
 *************************************************************/
__global__ void kernelDetectBadCases(int *ctriangles, int *cvertarr, int *cactive, 
                                     int nVerts, int step, int *cflag, int *tvertices, 
                                     int *deleted) {
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nVerts || cactive[x] >= 0) 
        return ; 

    int triStart, pNextTri, pTri, pOri, id, count = 0; 

    // Travel around the triangle fan
    pNextTri = cvertarr[x];

    pTri = decode_tri(pNextTri);
    pOri = decode_ori(pNextTri); 
    
    triStart = pTri; 

    int smallest = nVerts + 1; 

    // Walk to the right hand side. 
    do {
        count++; 
        id = ctriangles[pTri * 9 + 3 + pOri]; 
        if (id < x && (cactive[id] < 0 || cactive[id] == step))        // Conflict
            return ; 

        if (id < smallest)
            smallest = id; 

        pNextTri = ctriangles[pTri * 9 + (pOri + 2) % 3]; 

        pTri = decode_tri(pNextTri);
        pOri = decode_ori(pNextTri); 
    } while (pTri != triStart);
    
    *cflag = 1; 
    // Just try to bind this vertex to one of its neighbor
    // so that later when we want to insert this vertex back, 
    // we know where to start. 
    tvertices[x] = smallest; 
    cactive[x] = step;
    deleted[x] = count + 1; 
}

/*************************************************************
 * Delete triangles marked by this step. 
 * - Mark it as deleted (negative value for cnewtri)
 * - There's no need to remove the bond with neighboring 
 *   triangles. We're not boundary triangles, so later
 *   new triangles will be inserted (starshape holes). 
 * - Count the number of vertices on the triangle fan. 
 *************************************************************/

__global__ void kernelDeleteTriangles(int *ctriangles, int *cactive, int nVerts, int step, 
                                      int *cdeleted, short *cnewtri, int *cvertarr, 
                                      int *cstack, int *cavailtri, int *coffset) {
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nVerts || cactive[x] != step) 
        return ; 

    int offset = coffset[x]; 
    int pNextTri, pTri, pOri, id, top = 0; 

    pNextTri = cvertarr[x];        // Start
    pTri = decode_tri(pNextTri);
    pOri = decode_ori(pNextTri);
    int firstTri = pTri; 

    do {
        id = ctriangles[pTri * 9 + 3 + pOri];        // apex
        cnewtri[pTri] = -step;    // Mark as deleted
        cdeleted[id] = -1;        // Mark as affected

        cstack[offset + top] = id;    // Add to stack
        cavailtri[offset + top] = pTri;        // Available triangle
        top++; 

        pNextTri = ctriangles[pTri * 9 + (pOri + 2) % 3]; 

        pTri = decode_tri(pNextTri);
        pOri = decode_ori(pNextTri); 
    } while (pTri != firstTri); 
    
    cvertarr[x] = -1; 
}

/************************************************************
 * Fix the vertex array for those affected sites 
 ************************************************************/
__global__ void kernelFixVertexArray(int *ctriangles, int *cvertarr, short *cnewtri, 
                                     int *cdeleted, int nVerts) {    
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nVerts || cdeleted[x] != -1) 
        return ; 

    int p = cvertarr[x], pnext = p; 

    // Find the first valid triangle
    while (pnext >= 0 && cnewtri[decode_tri(pnext)] < 0)
        pnext = ctriangles[decode_tri(pnext) * 9 + 6 + decode_ori(pnext)]; 
    
    if (pnext != p)
        cvertarr[x] = pnext;

    while (pnext >= 0) {
        // Find an invalid triangle
        do {
            p = pnext; 
            pnext = ctriangles[decode_tri(p) * 9 + 6 + decode_ori(p)]; 
        } while (pnext >= 0 && cnewtri[decode_tri(pnext)] >= 0); 
        
        if (pnext >= 0)    {
            // Now pnext is deleted, so we fix the link for p. 

            // Find the next valid triangle
            while (pnext >= 0 && cnewtri[decode_tri(pnext)] < 0)
                pnext = ctriangles[decode_tri(pnext) * 9 + 6 + decode_ori(pnext)]; 
            
            ctriangles[decode_tri(p) * 9 + 6 + decode_ori(p)] = pnext; 
        }
    }
}

/******************************************************************
 * Patches holes created by removing bad case sites inside the
 * triangle mesh.
 ******************************************************************/
__global__ void kernelPatchHoles(int *ctriangles, int *cactive, short *cnewtri, 
                                 int *coffset, int *cstack, int *cavailtri,
                                 REAL2 *cvertices, int step, int nVerts, 
                                 int *cdeleted, int *cvertarr) {
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nVerts || cactive[x] != step) 
        return ; 

    REAL2 kernel = cvertices[x]; 
    REAL2 vtx0, vtx1, vtx2; 

    int offset = coffset[x]; 
    int v0, v1, v2; 
    int top = 0, tmp; 
    int size = cdeleted[x] - 1, avail = size; 

    for (int i = 0; i < size; i++) {
        v2 = cstack[offset + i]; 
        vtx2 = cvertices[v2]; 

        if (top >= 2) {
            while (true) {
                if (cuda_ccw(vtx0, vtx1, vtx2) > 0 &&
                    !CONTAIN_KERNEL(vtx0, vtx1, vtx2, kernel)) {
                    // Add new triangle
                    INSERT_TRIANGLE(v0, v1, v2, cavailtri[offset + (--avail)]); 

                    // Pop the stack
                    top--; 

                    v1 = v0; vtx1 = vtx0; 

                    if (top < 2)
                        break; 
                    
                    v0 = cstack[offset + top - 2]; 
                    vtx0 = cvertices[v0]; 
                }
                else
                    break; 
            }
        }

        cstack[offset + (top++)] = v2; 
        v0 = v1; v1 = v2; 
        vtx0 = vtx1; vtx1 = vtx2; 
    }

    int stackStart = 0; 

    while ((top - stackStart) > 4) {
        v2 = cstack[offset + stackStart++]; 
        vtx2 = cvertices[v2]; 

        while (true) {
            if (cuda_ccw(vtx0, vtx1, vtx2) > 0 &&
                !CONTAIN_KERNEL(vtx0, vtx1, vtx2, kernel)) {
                // Add new triangle
                INSERT_TRIANGLE(v0, v1, v2, cavailtri[offset + (--avail)]); 
                
                // Pop the stack
                top--; 

                v1 = v0; vtx1 = vtx0; 
                v0 = cstack[offset + top - 2]; 
                vtx0 = cvertices[v0]; 
            }
            else
                break; 
        }

        cstack[offset + top++] = v2; 
        v0 = v1; v1 = v2; 
        vtx0 = vtx1; vtx1 = vtx2; 
    }

    if (top - stackStart == 3) {
        v0 = cstack[offset + top - 3]; 
        v1 = cstack[offset + top - 2]; 
        v2 = cstack[offset + top - 1]; 
        
        INSERT_TRIANGLE(v0, v1, v2, cavailtri[offset + (--avail)]); 

    } else if (top - stackStart == 4) {
        v0 = cstack[offset + top - 4]; 
        v1 = cstack[offset + top - 3]; 
        v2 = cstack[offset + top - 2]; 
        int v3 = cstack[offset + top - 1];         

        vtx0 = cvertices[v0];
        vtx1 = cvertices[v1]; 
        vtx2 = cvertices[v2]; 
        REAL2 vtx3 = cvertices[v3]; 

        if (cuda_ccw(vtx0, vtx1, vtx2) > 0 &&
            cuda_ccw(vtx0, vtx2, vtx3) > 0) {
            INSERT_TRIANGLE(v0, v1, v2, cavailtri[offset + (--avail)]); 
            INSERT_TRIANGLE(v0, v2, v3, cavailtri[offset + (--avail)]); 

        } else {
            INSERT_TRIANGLE(v1, v2, v3, cavailtri[offset + (--avail)]); 
            INSERT_TRIANGLE(v1, v3, v0, cavailtri[offset + (--avail)]); 
        }
    }
}                                

/******************************************************************
 * Update the links between triangles after adding new triangles
 ******************************************************************/
__global__ void kernelUpdateTriangleLinks(int *ctriangles, int *cvertarr, short *cnewtri, 
                                          int nTris, int step) {
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nTris || cnewtri[x] != step) 
        return ; 

    int p0, p1, p2, n0 = -1, n1 = -1, n2 = -1;
    int nCounter, pNextTri, pTri, pOri, pTri9; 
    int x9 = x * 9;

    p2 = ctriangles[x9 + 3]; 
    p0 = ctriangles[x9 + 4]; 
    p1 = ctriangles[x9 + 5]; 
    nCounter = 0; 

    // orientation 0
    // Travel through the list of triangles sharing vertex 0 with this triangle.
    // In this list we can find at most two triangles sharing edge (p0, p1) and 
    // (p2, p0) with our triangle. 
    pNextTri = cvertarr[p0];

    while (pNextTri >= 0 && nCounter < 2) {
        pTri = decode_tri(pNextTri); 
        pOri = decode_ori(pNextTri); 
        pTri9 = pTri * 9; 

        if (p2 == ctriangles[pTri9 + 3 + (pOri + 2) % 3]) {    // NextDest
            n2 = pNextTri; 
            ctriangles[pTri9 + pOri] = (x << 2) | 2;  
            nCounter++; 
        }

        if (p1 == ctriangles[pTri9 + 3 + pOri]) {    // NextApex
            n0 = (pTri << 2) | ((pOri + 2) % 3);  
            ctriangles[pTri9 + (pOri + 2) % 3] = (x << 2);  
            nCounter++; 
        }

        pNextTri = ctriangles[pTri9 + 6 + pOri]; 
    }

    // orientation 1
    // Find the triangle with edge (p1, p2)
    pNextTri = cvertarr[p1]; 

    while (pNextTri >= 0) {
        pTri = decode_tri(pNextTri); 
        pOri = decode_ori(pNextTri); 
        pTri9 = pTri * 9; 

        if (p2 == ctriangles[pTri9 + 3 + pOri]) {    // NextApex
            n1 = (pTri << 2) | ((pOri + 2) % 3); 
            ctriangles[pTri9 + (pOri + 2) % 3] = (x << 2) | 1;  
            break ; 
        }

        pNextTri = ctriangles[pTri9 + 6 + pOri]; 
    }

    ctriangles[x9 + 0] = n0; 
    ctriangles[x9 + 1] = n1; 
    ctriangles[x9 + 2] = n2; 
}

//////////////////////////////////////////////////////////////

void cudaShifting() {
    // Constants for exact tests
    cutilSafeCall( cudaMemcpyToSymbol(constData, hostConst, 13 * sizeof(REAL)) ); 

	/******************************************************************************
     * First, detect all missing sites and sites on the boundary
     ******************************************************************************/
    dim3 block = dim3(WBLOCK); 
    dim3 grid = dim3(STRIPE, nVerts / (STRIPE * block.x) + 1); 
    
    kernelMissingDetection<<< grid, block >>>(nVerts, cactive, cvertarr); 
    cutilCheckError(); 

    /************************************************************************
     * Detect all sites that can be shifted together without any crossing
     ************************************************************************/
    step = 3;    // Reserve the label -2...2 for something else
    int flag; 

    block = dim3(128); 
    grid = dim3(STRIPE, nPoints / (STRIPE * block.x) + 1); 

    do {
        cutilSafeCall( cudaMemset(cflag, 0, sizeof(int)) ); 
        kernelShiftable<<< grid, block >>>(ctriangles, cvertarr, cvertices, 
            covertices, cactive, nPoints, step, cflag); 

        cutilSafeCall( cudaMemcpy(&flag, cflag, sizeof(int), cudaMemcpyDeviceToHost) ); 

        step++; 
    } while (flag > 0);

	cutilCheckError(); 

    /*******************************************************************************
     * Shift bad cases
     *******************************************************************************/
    int *cdeleted, *coffset;
    int *cstack, *cavailtri;

    preallocBlockSums(nVerts * 2); 
	cutilSafeCall( cudaMalloc((void **) &cnewtri, 2 * nVerts * sizeof(short)) );
	cutilSafeCall( cudaMalloc((void **) &cdeleted, nVerts * sizeof(int)) );
	cutilSafeCall( cudaMalloc((void **) &coffset, nVerts * sizeof(int)) );

    // Initialize 
    cutilSafeCall( cudaMemset(cnewtri, 0, nVerts * 2 * sizeof(short)) ); 

    do {
        cutilSafeCall( cudaMemset(cdeleted, 0, nPoints * sizeof(int)) ); 
        cutilSafeCall( cudaMemset(cflag, 0, sizeof(int)) ); 

        block = dim3(128); 
        grid = dim3(STRIPE, nPoints / (STRIPE * block.x) + 1); 

        kernelDetectBadCases<<< grid, block >>>(ctriangles, cvertarr, cactive, nPoints, 
                                                step, cflag, tvertices, cdeleted); 
        cutilCheckError(); 
        cutilSafeCall( cudaMemcpy(&flag, cflag, sizeof(int), cudaMemcpyDeviceToHost) ); 

        if (flag > 0) {
            // Prepare the stack
            prescanArray(coffset, cdeleted, nPoints); 
            int lastoffset, lastdeleted; 
            cutilSafeCall( cudaMemcpy(&lastoffset, coffset + nPoints - 1, sizeof(int), cudaMemcpyDeviceToHost) ); 
            cutilSafeCall( cudaMemcpy(&lastdeleted, cdeleted + nPoints - 1, sizeof(int), cudaMemcpyDeviceToHost) ); 
            
            cutilSafeCall( cudaMalloc((void **) &cstack, (lastoffset + lastdeleted) * sizeof(int)) ); 
            cutilSafeCall( cudaMalloc((void **) &cavailtri, (lastoffset + lastdeleted) * sizeof(int)) ); 

            // Delete triangles and construct the stack
            grid = dim3(STRIPE, nPoints / (block.x * STRIPE) + 1); 
            kernelDeleteTriangles<<< grid, block >>>(ctriangles, cactive, nPoints, step, 
                cdeleted, cnewtri, cvertarr, cstack, cavailtri, coffset); 

            // Fix the triangle mesh
            grid = dim3(STRIPE, nVerts / (STRIPE * block.x) + 1); 
            kernelFixVertexArray<<< grid, block >>>(ctriangles, cvertarr, cnewtri, 
                cdeleted, nVerts); 

            // Patch holes
            grid = dim3(STRIPE, nPoints / (STRIPE * block.x) + 1); 
            kernelPatchHoles<<< grid, block >>>(ctriangles, cactive, cnewtri, 
                coffset, cstack, cavailtri, cvertices, step, nPoints, cdeleted, cvertarr);

            grid = dim3(STRIPE, nTris / (STRIPE * block.x) + 1); 
            kernelUpdateTriangleLinks<<< grid, block >>>(ctriangles, cvertarr, cnewtri, 
                nTris, step); 

            cutilSafeCall( cudaFree(cavailtri) ); 
            cutilSafeCall( cudaFree(cstack) ); 
        }

        step++; 

    } while (flag > 0); 

	cutilSafeCall( cudaFree(cdeleted)); 
	cutilSafeCall( cudaFree(coffset)); 	

    /*******************************
     * DONE
     *******************************/

	cutilSafeCall( cudaFree(cactive)); 
	cutilSafeCall( cudaFree(cvertices)); 	
    // After shifting, we can start using the actual coordinates. 
}