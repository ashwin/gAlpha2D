/*
Author: Cao Thanh Tung
Date: 15/03/2011

File Name: cudaBoundary.cu

This file include all CUDA code to remove the fake boundary of the triangle mesh
that as been created to facilitate the parallel shifting and inserting missing
sites steps. 

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
#include "cudaScanLargeArray.h"

/***********************************************************
 * Declarations
 ***********************************************************/
#define WBLOCK                256    

#define INSERT_TRIANGLE(v0, v1, v2, tri) \
    tmp = tri; \
    ctriangles[tmp * 9 + 3] = v0; \
    ctriangles[tmp * 9 + 4] = v1; \
    ctriangles[tmp * 9 + 5] = v2; \
    ctriangles[tmp * 9 + 6] = atomicExch(&cvertarr[v1], (tmp << 2)); \
    ctriangles[tmp * 9 + 7] = atomicExch(&cvertarr[v2], (tmp << 2) | 1); \
    ctriangles[tmp * 9 + 8] = atomicExch(&cvertarr[v0], (tmp << 2) | 2); \
    cnewtri[tmp] = step

/**************************************************************
 * Exported methods
 **************************************************************/
extern "C" void cudaFixBoundary(); 

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
extern REAL2 *covertices;        
extern short *cnewtri; 
extern int step; 
extern int *cflag; 

/*************************************************************
 * Detect all removable boundary sites
 *************************************************************/
__global__ void kernelDetectBoundarySites(int *ctriangles, int *cvertarr, int *cactive, 
                    int *crightmost, int nPoints, int boundary, int step, int *cflag, int *deleted) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= boundary || cvertarr[nPoints + x] < 0) 
        return ; 

    int myId = nPoints + x; 
    int pNextTri, pTri, pOri, id, count = 0; 

    // Travel around the triangle fan
    pNextTri = cvertarr[myId];

    pTri = decode_tri(pNextTri);
    pOri = decode_ori(pNextTri); 
    
    // Walk to the right hand side. 
    do {
        count++; 
        id = ctriangles[pTri * 9 + 3 + (pOri + 2) % 3];    // Dest
        
        if (id > myId && cactive[id - nPoints] >= step)        // Conflict
            return ; 

        pNextTri = ctriangles[pTri * 9 + pOri]; 

        if (pNextTri >= 0) {
            pTri = decode_tri(pNextTri);
            pOri = (decode_ori(pNextTri) + 1) % 3; 
        }
    } while (pNextTri >= 0);

    // Store the rightmost triangle
    crightmost[x] = encode_tri(pTri, pOri); 

    pNextTri = cvertarr[myId]; 

    pTri = decode_tri(pNextTri);
    pOri = decode_ori(pNextTri); 
    
    // Walk to the left hand side. 
    do {
        count++; 
        id = ctriangles[pTri * 9 + 3 + pOri];  // Apex

        if (id > myId && cactive[id - nPoints] >= step)        // Conflict
            return ; 

        pNextTri = ctriangles[pTri * 9 + (pOri + 2) % 3]; 

        pTri = decode_tri(pNextTri);
        pOri = decode_ori(pNextTri); 
    } while (pNextTri >= 0);
    
    *cflag = 1; 
    cactive[x] = step;
    deleted[x] = count; 
    cvertarr[myId] = -1; 
}

/*************************************************************
 * Delete triangles marked by this step. 
 * - Mark it as deleted (negative value for cnewtri)
 * - There's no need to remove the bond with neighboring 
 *   triangles. We're not boundary triangles, so later
 *   new triangles will be inserted (starshape holes). 
 *************************************************************/

__global__ void kernelDeleteBoundaryTriangles(int *ctriangles, int *cactive, int nPoints, 
                        int boundary, int step, int *cdeleted, short *cnewtri, int *cstack, 
                        int *cavailtri, int *coffset, int *crightmost) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= boundary || cactive[x] != step) 
        return ; 

    int offset = coffset[x]; 
    int pNextTri, pTri, pOri, id, top = 0; 

    pNextTri = crightmost[x];        // Start
    pTri = decode_tri(pNextTri);
    pOri = decode_ori(pNextTri);

    id = ctriangles[pTri * 9 + 3 + (pOri + 2) % 3]; // Dest
    cdeleted[id] = -step; 
    cstack[offset + top] = id; 
    top++; 

    do {
        id = ctriangles[pTri * 9 + 3 + pOri];        // apex
        cnewtri[pTri] = -step;    // Mark as deleted
        cdeleted[id] = -step;        // Mark as affected
        
        int pOppTri = ctriangles[pTri * 9 + (pOri + 1) % 3]; 

        if (pOppTri >= 0)
            ctriangles[decode_tri(pOppTri) * 9 + decode_ori(pOppTri)] = -1; 

        cstack[offset + top] = id;    // Add to stack
        cavailtri[offset + top] = pTri;        // Available triangle
        top++; 

        pNextTri = ctriangles[pTri * 9 + (pOri + 2) % 3]; 

        pTri = decode_tri(pNextTri);
        pOri = decode_ori(pNextTri); 
    } while (pNextTri >= 0); 
}

/************************************************************
 * Fix the vertex array for those affected sites 
 ************************************************************/
__global__ void kernelFixBoundaryVertexArray(int *ctriangles, int *cvertarr, short *cnewtri, 
                                     int *cdeleted, int nVerts, int step) {    
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nVerts || cdeleted[x] != -step) 
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

/*************************************************************
 * Patch the hole created by removing a vertex on the boundary
 * of the triangle mesh.
 *************************************************************/
__global__ void kernelPatchHoleBoundary(int *ctriangles, int *cactive, short *cnewtri, 
                        int *coffset, int *cstack, int *cavailtri, REAL2 *covertices, 
                        int step, int nPoints, int boundary, int *cdeleted, int *cvertarr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x >= boundary || cactive[x] != step) 
        return ; 
                
    REAL2 vtx0, vtx1, vtx2; 

    int offset = coffset[x]; 
    int v0, v1, v2; 
    int top = 0, tmp; 
    int size = cdeleted[x], avail = size; 

    for (int i = 0; i < size; i++) {
        v2 = cstack[offset + i]; 
        vtx2 = covertices[v2]; 

        if (top >= 2) {
            while (true) {
                if (cuda_ccw(vtx0, vtx1, vtx2) > 0.0) {
                    // Add new triangle
                    INSERT_TRIANGLE(v0, v1, v2, cavailtri[offset + (--avail)]); 

                    // Pop the stack
                    top--; 

                    v1 = v0; vtx1 = vtx0; 

                    if (top < 2)
                        break; 
                    
                    v0 = cstack[offset + top - 2]; 
                    vtx0 = covertices[v0]; 
                }
                else
                    break; 
            }
        }

        cstack[offset + (top++)] = v2; 
        v0 = v1; v1 = v2; 
        vtx0 = vtx1; vtx1 = vtx2; 
    }
}

/******************************************************************
 * Update the links between triangles after adding new triangles
 ******************************************************************/
__global__ void kernelUpdateTriangleLinksBoundary(int *ctriangles, int *cvertarr, short *cnewtri, 
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


__global__ void kernelMarkValidTriangles(short *cnewtri, int *cvalid, int nTris)
{
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nTris) 
        return ; 

    cvalid[x] = (cnewtri[x] >= 0) ? 1 : 0; 
}

__global__ void kernelCollectEmptySlots(short *cnewtri, int *cprefix, int *cempty, int nTris)
{
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nTris || cnewtri[x] >= 0) 
        return ; 

    int id = x - cprefix[x]; 

    cempty[id] = x; 
}

__global__ void kernelFillEmptySlots(short *cnewtri, int *cprefix, int *cempty, int *ctriangles, 
                                     int nTris, int newnTris, int offset)
{
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nTris || cnewtri[x] < 0) 
        return ; 

    int value;

    if (x < newnTris) 
        value = x; 
    else {
        value = cempty[cprefix[x] - offset]; 

        for (int i = 0; i < 9; i++)
            ctriangles[value * 9 + i] = ctriangles[x * 9 + i]; 
    }        

    cprefix[x] = value; 
}

__global__ void kernelFixIndices(int *ctriangles, int *newindex, int nTris) {
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
    
    if (x + tId < nTris) {
        i = tId * 9;         
        if (ct[i] >= 0)
            ct[i] = encode_tri(newindex[decode_tri(ct[i])], decode_ori(ct[i])); 
        i++; 
        if (ct[i] >= 0)
            ct[i] = encode_tri(newindex[decode_tri(ct[i])], decode_ori(ct[i])); 
        i++; 
        if (ct[i] >= 0)
            ct[i] = encode_tri(newindex[decode_tri(ct[i])], decode_ori(ct[i])); 
        i++; 
        i++; 
        i++; 
        i++; 
        if (ct[i] >= 0)
            ct[i] = encode_tri(newindex[decode_tri(ct[i])], decode_ori(ct[i])); 
        i++; 
        if (ct[i] >= 0)
            ct[i] = encode_tri(newindex[decode_tri(ct[i])], decode_ori(ct[i])); 
        i++; 
        if (ct[i] >= 0)
            ct[i] = encode_tri(newindex[decode_tri(ct[i])], decode_ori(ct[i]));        
    }

    __syncthreads(); 

    for (i = 0, id = tId; i < 9; i++, id += WBLOCK)
        ctriangles[x9 + id] = ct[id]; 
}

//////////////////////////////////////////////////////////////

void cudaFixBoundary() {
    // Constants for EXACT tests
    cutilSafeCall( cudaMemcpyToSymbol(constData, hostConst, 13 * sizeof(REAL)) ); 

    dim3 block = dim3(WBLOCK); 
    dim3 grid; 
    
    /*******************************************************************************
     * Removes the fake boundary
     *******************************************************************************/
    int *cactive, *cdeleted, *coffset, *caffected;
    int *cstack, *cavailtri, *crightmost;

    int boundary = nVerts - nPoints; 

	cutilSafeCall( cudaMalloc((void **) &caffected, nVerts * sizeof(int)) ); 
    cutilSafeCall( cudaMalloc((void **) &cdeleted, boundary * sizeof(int)) ); 
    cutilSafeCall( cudaMalloc((void **) &cactive, boundary * sizeof(int)) ); 
    cutilSafeCall( cudaMalloc((void **) &coffset, boundary * sizeof(int)) ); 
    cutilSafeCall( cudaMalloc((void **) &crightmost, boundary * sizeof(int)) ); 
    cutilSafeCall( cudaMemset(cactive, 127, boundary * sizeof(int)) ); 
    cutilSafeCall( cudaMemset(caffected, 0, nVerts * sizeof(int)) ); 

//    printf("boundary = %i\n", boundary); 

    int flag; 
    do {
        cutilSafeCall( cudaMemset(cdeleted, 0, boundary * sizeof(int)) ); 
        cutilSafeCall( cudaMemset(cflag, 0, sizeof(int)) ); 

        block = dim3(32); 
        grid = dim3(boundary / block.x + 1); 

        kernelDetectBoundarySites<<< grid, block >>>(ctriangles, cvertarr, cactive, crightmost,
            nPoints, boundary, step, cflag, cdeleted);  
        cutilCheckError(); 

        cutilSafeCall( cudaMemcpy(&flag, cflag, sizeof(int), cudaMemcpyDeviceToHost) ); 

        if (flag > 0) {
            // Prepare the stack
            prescanArray(coffset, cdeleted, boundary); 
            int lastoffset, lastdeleted; 
            cutilSafeCall( cudaMemcpy(&lastoffset, coffset + boundary - 1, sizeof(int), cudaMemcpyDeviceToHost) ); 
            cutilSafeCall( cudaMemcpy(&lastdeleted, cdeleted + boundary - 1, sizeof(int), cudaMemcpyDeviceToHost) ); 
            
            //printf("Stack size: %i\n", lastoffset + lastdeleted); 

            cutilSafeCall( cudaMalloc((void **) &cstack, (lastoffset + lastdeleted) * sizeof(int)) ); 
            cutilSafeCall( cudaMalloc((void **) &cavailtri, (lastoffset + lastdeleted) * sizeof(int)) ); 

            // Delete triangles and construct the stack
            grid = dim3(boundary / block.x + 1); 
            kernelDeleteBoundaryTriangles<<< grid, block >>>(ctriangles, cactive, nPoints, 
                boundary, step, caffected, cnewtri, cstack, cavailtri, coffset, crightmost); 
 
            // Fix the triangle mesh
            block = dim3(WBLOCK); 
            grid = dim3(STRIPE, nVerts / (STRIPE * block.x) + 1); 
            kernelFixBoundaryVertexArray<<< grid, block >>>(ctriangles, cvertarr, cnewtri, 
                caffected, nVerts, step); 
 
            // Patch holes
            block = dim3(32); 
            grid = dim3(boundary / block.x + 1); 
            kernelPatchHoleBoundary<<< grid, block >>>(ctriangles, cactive, cnewtri, 
                coffset, cstack, cavailtri, covertices, step, nPoints, boundary, cdeleted, 
                cvertarr);

            block = dim3(WBLOCK); 
            grid = dim3(STRIPE, nTris / (STRIPE * block.x) + 1); 
            kernelUpdateTriangleLinksBoundary<<< grid, block >>>(ctriangles, cvertarr, cnewtri, 
                nTris, step); 

            cutilSafeCall( cudaFree(cavailtri) ); 
            cutilSafeCall( cudaFree(cstack) ); 
        }

        step++; 
    } while (flag > 0); 

    // printf("Remove fake boundary! Total steps: %i ; Total time: %.3fms\n", lastStep - firstStep, totalTime); 

    cutilSafeCall( cudaFree(crightmost) ); 
    cutilSafeCall( cudaFree(coffset) ); 
    cutilSafeCall( cudaFree(cactive) ); 
    cutilSafeCall( cudaFree(cdeleted) ); 
	cutilSafeCall( cudaFree(caffected) ); 
   
    /*********************************************************
     * Compact the triangle list 
     *********************************************************/
    int *cvalid, *cprefix;

	cutilSafeCall( cudaMalloc((void **) &cvalid, 2 * nVerts * sizeof(int)) ); 
	cutilSafeCall( cudaMalloc((void **) &cprefix, 2 * nVerts * sizeof(int)) ); 

    block = dim3(WBLOCK); 
    grid = dim3(STRIPE, nTris / (STRIPE * block.x) + 1); 

    // Mark the valid triangles in the list
    kernelMarkValidTriangles<<< grid, block >>>(cnewtri, cvalid, nTris); 
    cutilCheckError(); 

    // Compute the offset of them in the new list
    prescanArray(cprefix, cvalid, nTris); 

    int newnTris, lastitem, offset; 
    cutilSafeCall( cudaMemcpy(&newnTris, cprefix + nTris - 1, sizeof(int), cudaMemcpyDeviceToHost) ); 
    cutilSafeCall( cudaMemcpy(&lastitem, cvalid + nTris - 1, sizeof(int), cudaMemcpyDeviceToHost) ); 
    newnTris += lastitem; 
    cutilSafeCall( cudaMemcpy(&offset, cprefix + newnTris, sizeof(int), cudaMemcpyDeviceToHost) ); 

//    printf("nTris = %i, new nTris = %i\n", nTris, newnTris); 

    // Find all empty slots in the list
    kernelCollectEmptySlots<<< grid, block >>>(cnewtri, cprefix, cvalid, nTris); 
    cutilCheckError(); 

    // Move those valid triangles at the end of the list
    // to the holes in the list. 
    grid = dim3(STRIPE, nTris / (STRIPE * block.x) + 1); 
    kernelFillEmptySlots<<< grid, block >>>(cnewtri, cprefix, cvalid, ctriangles, 
        nTris, newnTris, offset); 
    cutilCheckError(); 

    // Fix the links after the index of our triangles are mixed up
    grid = dim3(STRIPE, newnTris / (STRIPE * block.x) + 1); 
    kernelFixIndices<<< grid, block >>>(ctriangles, cprefix, newnTris); 
    cutilCheckError(); 

	cutilSafeCall( cudaFree(cprefix) ); 
	cutilSafeCall( cudaFree(cvalid) ); 
    
    nTris = newnTris; 

   /*******************************
     * DONE
     *******************************/
	cutilSafeCall( cudaFree(cnewtri) ); 

    // Last time we need prefix sum, so can deallocate this
    deallocBlockSums(); 
}