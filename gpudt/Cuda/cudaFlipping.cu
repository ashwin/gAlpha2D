/*
Author: Cao Thanh Tung, Qi Meng
Date: 15/03/2011

File Name: cudaFlipping.cu

This file include all CUDA code to perform the flipping step

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

/***********************************************************
* Declarations
***********************************************************/
#define WBLOCK      256        

#define SET_TRIANGLE(vOrg, vDest, vApex, nOrg, nDest, nApex, tri, ori) \
    ctriangles[(tri) * 9 + 3 + ((ori) + 1) % 3] = (vOrg); \
    ctriangles[(tri) * 9 + 3 + ((ori) + 2) % 3] = (vDest); \
    ctriangles[(tri) * 9 + 3 + (ori)] = (vApex); \
    ctriangles[(tri) * 9 + 6 + (ori)] = (nOrg); \
    ctriangles[(tri) * 9 + 6 + ((ori) + 1) % 3] = (nDest); \
    ctriangles[(tri) * 9 + 6 + ((ori) + 2) % 3] = (nApex) 

#define UPDATE_TEMP_LINK(pTriOri, pNext) \
    if ((pTriOri) >= 0) \
    ctriangles[decode_tri(pTriOri) * 9 + 6 + decode_ori(pTriOri)] = -(pNext)

#define UPDATE_LINK(pTriOri, pNext) \
    if ((pTriOri) >= 0) \
    ctriangles[decode_tri(pTriOri) * 9 + decode_ori(pTriOri)] = (pNext)

/**************************************************************
* Exported methods
**************************************************************/
extern "C" int cudaFlipping(int **suspective); 

/**************************************************************
* Definitions
**************************************************************/
// Decode an oriented triangle. 
// An oriented triangle consists of 32 bits. 
// - 30 highest bits represent the triangle index, 
// - 2 lowest bits represent the orientation (the starting vertex, 0, 1 or 2)
#define decode_tri(x)            ((x) >> 2)
#define decode_ori(x)            ((x) & 3)
#define encode_tri(tri, ori)     (((tri) << 2) | (ori))

/************************************************************
* Variables and functions shared with the main module
************************************************************/
extern int nTris, nVerts, nPoints, nConstraints;   
extern int *ctriangles;            
extern REAL2 *covertices;        
extern int *cflag; 
extern int *cconstraints;
extern int *cvertarr;
extern BYTE *ifEdgeIsConstraint_cpu;
extern PGPUDTPARAMS  gpudtParams;

/***************************************************************************
* Determine which edge is constraint edge, 
* mark those edges. 
***************************************************************************/
__global__ void KernelMarkConstrain1(int *ctriangles, int *cvertarr, REAL2 *covertices, 
                                     int* cflag, int nConstraints, int* tconstrainLink_flip, BYTE *ifEdgeIsConstraint)
{

    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x+threadIdx.x;
    if (x >= nConstraints) 
        return ;

    int point1 = tconstrainLink_flip[2*x];
    int point2 = tconstrainLink_flip[2*x+1];	
    REAL2 point1_R,point2_R,point3_R;
    point1_R = covertices[point1];
    point2_R = covertices[point2];
    REAL test1=-1;
    int ApTri, ApOri, adjacent;
    int pTri, pOri, pApex;	
    int p = cvertarr[point1]; 	

    pTri = decode_tri(p);   pOri = decode_ori(p);
    pApex = ctriangles[pTri*9 + 3 +  pOri];	
    point3_R = covertices[pApex];
    test1 = cuda_ccw(point1_R, point2_R, point3_R);
    int pnext = -1;
    if(test1==0 && point2==pApex)//point2 == pApex
    {		

        ifEdgeIsConstraint[pTri*3 + (pOri+2)%3] = 1; 
        adjacent = ctriangles[pTri*9 + (pOri+2)%3];
        ApTri  = decode_tri(adjacent);
        if(ApTri>=0)
        {
            ApOri  = decode_ori(adjacent);
            ifEdgeIsConstraint[ApTri*3 + (ApOri+0)%3] = 1; 
        }
        return;
    }
    else if(test1<0)//pApex is on the right
    {

        pnext = ctriangles[pTri*9 + (pOri+2)%3];		
        do{
            p = pnext;
            pTri = decode_tri(p);   pOri = decode_ori(p);
            pApex = ctriangles[pTri*9 + 3 +  pOri];				
            if(pApex == point2)
            {
                ifEdgeIsConstraint[pTri*3 + (pOri+2)%3] = 1; 
                adjacent = ctriangles[pTri*9 + (pOri+2)%3];
                ApTri  = decode_tri(adjacent);				

                if(ApTri>=0)
                {					  
                    ApOri  = decode_ori(adjacent);
                    ifEdgeIsConstraint[ApTri*3 + (ApOri+0)%3] = 1;					
                }
                return;
            }
            else
            {
                pnext = ctriangles[pTri*9 + (pOri+2)%3];	
            }

        }while(true);
    }
    else//pApex is on the left  or test==0&&pApex!=point2		
    {
        if( ctriangles[pTri*9 + 3 +  (pOri+2)%3] == point2)//the first triangle is the needed triangle
        {
            ifEdgeIsConstraint[pTri*3 + (pOri+0)%3] = 1;
            adjacent = ctriangles[pTri*9 + (pOri+0)%3];
            ApTri  = decode_tri(adjacent);
            if(ApTri>=0)
            {
                ApOri  = decode_ori(adjacent);
                ifEdgeIsConstraint[ApTri*3 + (ApOri+0)%3] = 1; 
            }
            return;
        } 

        pnext = ctriangles[pTri*9 + (pOri+0)%3];	

        do{

            p = pnext;			
            if(decode_tri(p)<0)
            {
                ifEdgeIsConstraint[pTri*3 + (pOri+0)%3] = 1;
                return;
            }
            pTri = decode_tri(p);   pOri = decode_ori(p);			
            pApex = ctriangles[pTri*9 + 3 +  (pOri+0)%3];//pApex
            if(pApex == point2)
            {				
                ifEdgeIsConstraint[pTri*3 + (pOri+1)%3] = 1;
                adjacent = ctriangles[pTri*9 + (pOri+1)%3];				
                ApTri  = decode_tri(adjacent);
                if(ApTri>=0)
                {
                    ApOri = decode_ori(adjacent);
                    ifEdgeIsConstraint[ApTri*3 + (ApOri+0)%3] = 1; 
                }
                return;
            }
            else
            {
                pnext = ctriangles[pTri*9 + (pOri+1)%3];	
            }
        }while(true);
    } 
}

/*************************************************************
* Detect all sites that can be shifted together 
* without causing any crossing. 
* We're guaranteed that all boundary sites are already marked
*************************************************************/
__global__ void kernelNeedFlipping(int *ctriangles, REAL2 *covertices, BYTE *flipOri, 
                                   int *flipStep, int *flipBy, BYTE *cmarker, 
                                   int nTris, int step, int *cflag, BYTE *ifEdgeIsConstrain) 
{
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

    x += tId;

    if (x >= nTris || cmarker[x] != 0) 
        return ; 

    REAL2 pOrg, pDest, pApex, pOpp;
    int nOrg, nDest, nApex; 
    int pTri, pOri; 
    bool notsure = false; 

    nOrg = ct[tId * 9 + 0]; 
    nDest = ct[tId * 9 + 1]; 
    nApex = ct[tId * 9 + 2];

    pOrg = covertices[ct[tId * 9 + 4]]; 
    pDest = covertices[ct[tId * 9 + 5]]; 
    pApex = covertices[ct[tId * 9 + 3]];

    // nOrg
    int t;
    t = ifEdgeIsConstrain[x * 3 + 0];
    if( t == -1)// do when current edge is not constraint edge
    {
        pTri = decode_tri(nOrg); pOri = decode_ori(nOrg); 
        if (x < pTri || (pTri >= 0 && cmarker[pTri] != 0)) {
            pOpp = covertices[ctriangles[pTri * 9 + 3 + pOri]]; 

            switch (cuda_inCircle(pOrg, pDest, pApex, pOpp)) {
                case 0 : notsure = true; break;     
                case 1 : 
                    flipOri[x] = 2; 
                    flipStep[x] = step; 
                    atomicMin(&flipBy[x], x); 
                    atomicMin(&flipBy[pTri], x);
                    *cflag = 1; 
                    return ; 
            }
        }
    }

    // nDest
    t = ifEdgeIsConstrain[x * 3 + 1];
    if( t == -1) //do when current edge is not constraint edge
    {
        pTri = decode_tri(nDest); pOri = decode_ori(nDest); 
        if (x < pTri || (pTri >= 0 && cmarker[pTri] != 0)) {
            pOpp = covertices[ctriangles[pTri * 9 + 3 + pOri]]; 

            switch (cuda_inCircle(pDest, pApex, pOrg, pOpp)) {
                case 0 : notsure = true; break; 
                case 1 : 
                    flipOri[x] = 0; 
                    flipStep[x] = step; 
                    atomicMin(&flipBy[x], x); 
                    atomicMin(&flipBy[pTri], x);
                    *cflag = 1; 
                    return ; 
            }
        }
    }

    // nApex
    pTri = decode_tri(nApex); pOri = decode_ori(nApex); 
    t = ifEdgeIsConstrain[x * 3 + 2];
    if( t == -1)//do when current edge is not constraint edge
    {
        if (x < pTri || (pTri >= 0 && cmarker[pTri] != 0)) {
            pOpp = covertices[ctriangles[pTri * 9 + 3 + pOri]]; 

            switch (cuda_inCircle(pApex, pOrg, pDest, pOpp)) {
                case 0 : notsure = true; break; 
                case 1 : 
                    flipOri[x] = 1; 
                    flipStep[x] = step; 
                    atomicMin(&flipBy[x], x);       // The one with minimun index
                    atomicMin(&flipBy[pTri], x);    // will win. 
                    *cflag = 1; 
                    return ; 
            }
        }
    }

    if (notsure)    // Inaccurate in_circle test.
        flipStep[x] = -step; 
    else
        cmarker[x] = 1;
}

// To flip a pair of triangle, we need to update the links between neighbor
// triangles. There is a chance that we need to flip two pairs in which 
// there are two neighboring triangles, thus the flipping is performed
// in 3 phases: 
// - Phase 1: Update the triangle vertices. Also, note down the new 
//            neighbors of each triangles in the 3 nexttri links (previously
//            used to store the vertex array). 
// - Phase 2: Each new triangle will then inform its new neighbors of its 
//            existence by writing itself in the corresponding nexttri link
//            of its neighbor.
// - Phase 3: Each new triangle update its links to its neighbors using the
//            3 new nexttri links. If any of the link is not updated, that
//            means the corresponding neighbor is not flipped in this pass. 
//            We then actively update the corresponding link in that triangle.
__global__ void kernelUpdatePhase1(int *ctriangles, BYTE *cflipOri, int *cflipStep, 
                                   int *cflipBy, BYTE *cmarker, int nTris, int step, BYTE *ifEdgeIsConstraint) 
{
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nTris)
        return ; 

    if (cflipStep[x] < 0) {  // Inaccurate incircle test
        //cmarker[x] = 1; 
        return ; 
    }

    int ori, pOpp, pOppTri, pOppOri, pOrg, pDest, pApex, nOrg, nApex, npDest, npApex;
    int newtag = -1; 

    if (cflipBy[x] == x) {      // I'm the one who win the right to flip myself
        ori = cflipOri[x]; 
        pOpp = ctriangles[x * 9 + (ori + 1) % 3]; 
        pOppTri = decode_tri(pOpp);
        pOppOri = decode_ori(pOpp); 

        if (cflipBy[pOppTri] == x) {    // I'm also the one who win the right
            newtag = pOpp;              // to flip this neighbor, so I can flip.
            pOrg = ctriangles[x * 9 + 3 + (ori + 1) % 3]; 
            pDest = ctriangles[x * 9 + 3 + (ori + 2) % 3]; 
            pApex = ctriangles[x * 9 + 3 + ori];
            pOpp = ctriangles[pOppTri * 9 + 3 + pOppOri]; 
            nOrg = ctriangles[x * 9 + ori]; 
            nApex = ctriangles[x * 9 + (ori + 2) % 3]; 
            npDest = ctriangles[pOppTri * 9 + (pOppOri + 1) % 3]; 
            npApex = ctriangles[pOppTri * 9 + (pOppOri + 2) % 3]; 

            // Update vertices + nexttri links
            SET_TRIANGLE(pOrg, pDest, pOpp, 3 + nOrg, -1, 3 + nApex, x, ori); 
            SET_TRIANGLE(pApex, pOrg, pOpp, -1, 3 + npDest, 3 + npApex, pOppTri, pOppOri); 
            int temp0,temp2,tempp1,tempp2;

            temp0 = ifEdgeIsConstraint[x*3+(ori+0)%3];///old apex
            temp2 = ifEdgeIsConstraint[x*3+(ori+2)%3];///old dest
            tempp1 = ifEdgeIsConstraint[pOppTri*3+(pOppOri+1)%3];   //old org
            tempp2 = ifEdgeIsConstraint[pOppTri*3+(pOppOri+2)%3];   //old dest		

            ifEdgeIsConstraint[x*3+(ori+0)%3]= temp0;//apex
            ifEdgeIsConstraint[x*3+(ori+1)%3]= tempp1;//org
            ifEdgeIsConstraint[x*3+(ori+2)%3]= -1;//dest

            ifEdgeIsConstraint[pOppTri*3+(pOppOri+0)%3]= temp2;//apex
            ifEdgeIsConstraint[pOppTri*3+(pOppOri+1)%3]= -1;//org
            ifEdgeIsConstraint[pOppTri*3+(pOppOri+2)%3]= tempp2;//dest		
        }
    }

    // Record the opp triangle to be used in the next phase
    cflipStep[x] = newtag;  
}

__global__ void kernelUpdatePhase2(int *ctriangles, BYTE *cflipOri, int *cflipStep, 
                                   int *cflipBy, int nTris, int step) 
{
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nTris || cflipStep[x] < 0) 
        return ; 

    int ori, pOpp, pOppTri, pOppOri, nOrg, nApex, npDest, npApex;

    ori = cflipOri[x]; 
    pOpp = cflipStep[x]; 
    pOppTri = decode_tri(pOpp);
    pOppOri = decode_ori(pOpp); 
    nOrg = ctriangles[x * 9 + ori]; 
    nApex = ctriangles[x * 9 + (ori + 2) % 3]; 
    npDest = ctriangles[pOppTri * 9 + (pOppOri + 1) % 3]; 
    npApex = ctriangles[pOppTri * 9 + (pOppOri + 2) % 3]; 

    // Update my neighbors of my existence
    UPDATE_TEMP_LINK(nOrg, encode_tri(x, ori)); 
    UPDATE_TEMP_LINK(nApex, encode_tri(pOppTri, pOppOri)); 
    UPDATE_TEMP_LINK(npDest, encode_tri(x, (ori + 1) % 3)); 
    UPDATE_TEMP_LINK(npApex, encode_tri(pOppTri, (pOppOri + 2) % 3)); 
}

__global__ void kernelUpdatePhase3(int *ctriangles, BYTE *cflipOri, int *cflipStep, 
                                   int *cflipBy, BYTE *cmarker, int nTris, int step) 
{
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (x >= nTris || cflipStep[x] < 0) 
        return ; 

    int ori, pOpp, pOppTri, pOppOri, nOrg, nApex, npDest, npApex;

    ori = cflipOri[x]; 
    pOpp = cflipStep[x]; 
    pOppTri = decode_tri(pOpp);
    pOppOri = decode_ori(pOpp); 

    // Update other links
    nOrg = ctriangles[x * 9 + 6 + ori]; 
    nApex = ctriangles[x * 9 + 6 + (ori + 2) % 3]; 
    npDest = ctriangles[pOppTri * 9 + 6 + (pOppOri + 1) % 3]; 
    npApex = ctriangles[pOppTri * 9 + 6 + (pOppOri + 2) % 3]; 

    if (nOrg > 0) {        // My neighbor do not update me, update him
        nOrg = -(nOrg - 3); 
        UPDATE_LINK(-nOrg, encode_tri(x, ori)); 
    }

    if (nApex > 0) {
        nApex = -(nApex - 3); 
        UPDATE_LINK(-nApex, encode_tri(pOppTri, pOppOri)); 
    }

    if (npDest > 0) {
        npDest = -(npDest - 3); 
        UPDATE_LINK(-npDest, encode_tri(x, (ori + 1) % 3)); 
    }

    if (npApex > 0) {
        npApex = -(npApex - 3); 
        UPDATE_LINK(-npApex, encode_tri(pOppTri, (pOppOri + 2) % 3)); 
    }

    // Update my own links
    ctriangles[x * 9 + ori] = -nOrg; 
    ctriangles[x * 9 + (ori + 1) % 3] = -npDest; 
    ctriangles[x * 9 + (ori + 2) % 3] = encode_tri(pOppTri, (pOppOri + 1) % 3); 
    ctriangles[pOppTri * 9 + pOppOri] = -nApex; 
    ctriangles[pOppTri * 9 + (pOppOri + 1) % 3] = encode_tri(x, (ori + 2) % 3); 
    ctriangles[pOppTri * 9 + (pOppOri + 2) % 3] = -npApex; 

    // Mark the affected triangles, so that we will check again in the next pass
    cflipStep[x] = -1; 
    if (nOrg <= 0) cmarker[decode_tri(-nOrg)] = 0; 
    if (nApex <= 0) cmarker[decode_tri(-nApex)] = 0; 
    if (npDest <= 0) cmarker[decode_tri(-npDest)] = 0; 
    if (npApex <= 0) cmarker[decode_tri(-npApex)] = 0; 
}

__global__ void kernelCheckingPhase(int *ctriangles, int *cvertarr, REAL2 *covertices, 
                                    int *cflag, int nConstraints, int *tconstrainLink_flip)
{
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x+threadIdx.x;;

    if (x >= nConstraints) 
        return ; 	

    int point1 = tconstrainLink_flip[2*x];
    int point2 = tconstrainLink_flip[2*x+1];
    REAL2 point1_R,point2_R,point3_R,point4_R;
    point1_R = covertices[point1];
    point2_R = covertices[point2];

    int pTri,pOri,pApex,pDest;
    int p = cvertarr[point1];

    REAL test1=-1, test2=-1;
    int pnext;
    pTri = decode_tri(p);
    pOri = decode_ori(p);
    pApex = ctriangles[pTri*9 + 3 +  pOri];
    pDest = ctriangles[pTri*9 + 3 +  (pOri+2)%3];

    if(pApex == point2 || pDest==point2)
        return; 
    point3_R = covertices[pApex];
    test1 = cuda_ccw(point1_R,point2_R,point3_R);
    if(test1<0)//pApex is on the right of direction point1point2, so we need to find left side pTri*9+2
    {	
        pnext = ctriangles[pTri*9 + (2 + pOri)%3];
        //find the first triangle that intersection constraint x.	
        do
        {				
            pTri = decode_tri(pnext);
            pOri = decode_ori(pnext);		
            pApex = ctriangles[pTri*9 + 3 + (0+pOri)%3];	
            if(pApex==point2) 
                return;//pOrg=point1, pApex=point2;
            point4_R = covertices[pApex];
            test2 = cuda_ccw(point1_R,point2_R,point4_R);

            if( test1*test2 < 0)
            {				
                *cflag = 1;
                break; 
            }
            p = pnext;		
            pnext = ctriangles[pTri*9 + (2 + pOri)%3];

        } while (pnext!=cvertarr[point1]);
    }
    else//pApex is on the left of direction point1point2, so we need to find right side pTri*9+0
    {			
        pnext = cvertarr[point1];			
        //find the first triangle that intersection constraint x.	
        int qq = 0;
        do
        {
            qq ++;			
            pTri = decode_tri(pnext);
            pOri = decode_ori(pnext);	
            if(qq==1)
                pDest = ctriangles[pTri*9 + 3 + (2+pOri)%3]; 
            else
                pDest = ctriangles[pTri*9 + 3 + (0+pOri)%3]; 

            if(pDest==point2) 
                return;//pOrg=point1, pDest=point2;
            point4_R = covertices[pDest];
            test2 = cuda_ccw(point1_R,point2_R,point4_R);		

            if( test1*test2 < 0)
            {	                
                *cflag = 1;			
                break; 
            }
            p = pnext;	

            if(qq==1)
                pnext = ctriangles[pTri*9 + (0+pOri)%3];	
            else
                pnext = ctriangles[pTri*9 + (1+pOri)%3];	

        } while (pnext!=cvertarr[point1]);
    }
}

__global__ void kernelFixVertArray1(int *ctriangles, int nTris, int *cvertarr) 

{
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;	
    if (x >= nTris)
        return ; 

    int v0 = ctriangles[x * 9 + 4];
    int v1 = ctriangles[x * 9 + 5];
    int v2 = ctriangles[x * 9 + 3];

    ctriangles[x * 9 + 6] = atomicExch(&cvertarr[v0], (x << 2)); 
    ctriangles[x * 9 + 7] = atomicExch(&cvertarr[v1], (x << 2) | 1); 
    ctriangles[x * 9 + 8] = atomicExch(&cvertarr[v2], (x << 2) | 2); 
}

int cudaFlipping(int **suspective) 
{
    // Constants for the EXACT tests
    cutilSafeCall( cudaMemcpyToSymbol(constData, hostConst, 13 * sizeof(REAL)) ); 

    int flag; 
    int step = 2; 

    BYTE *ifEdgeIsConstraint;

    cudaMalloc( (void**)&ifEdgeIsConstraint, 3*nTris* sizeof(BYTE) );
    cutilSafeCall( cudaMemset(ifEdgeIsConstraint, -1, 3*nTris * sizeof(BYTE)) );	

    int *cflipStep, *cflipBy;
    BYTE *cmarker, *cflipOri; 
    dim3 grid, block; 

    cutilSafeCall( cudaMalloc((void **) &cflipOri, nVerts * 2) ); 
    cutilSafeCall( cudaMalloc((void **) &cflipStep, nVerts * 2 * sizeof(int)) ); 
    cutilSafeCall( cudaMalloc((void **) &cflipBy, nVerts * 2 * sizeof(int)) ); 
    cutilSafeCall( cudaMalloc((void **) &cmarker, nVerts * 2) ); 

    cutilSafeCall( cudaMemset(cflipStep, 255, nTris * sizeof(int)) ); 
    cutilSafeCall( cudaMemset(cmarker, 0, nTris) ); 


    block = dim3(128); 
    grid = dim3(STRIPE, nTris/(STRIPE * block.x) + 1);	

    cutilSafeCall( cudaMemset(cvertarr, 255, nVerts * sizeof(int)) );
    kernelFixVertArray1<<< grid, block >>>(ctriangles,nTris,cvertarr); //fix vertex array 
    cutilCheckError(); 


    grid = dim3(STRIPE, nTris / (STRIPE * block.x) + 1); 
    // For all traingles, mark all constraint edge
    KernelMarkConstrain1<<<grid,block>>>(ctriangles, cvertarr, covertices, cflag, nConstraints, cconstraints,ifEdgeIsConstraint); 	
    cutilSafeCall( cudaMemcpy(&flag, cflag, sizeof(int), cudaMemcpyDeviceToHost) ); 	
    cutilCheckError();

    // Start flipping
    do {
        cutilSafeCall( cudaMemset(cflag, 0, sizeof(int)) ); 
        cutilSafeCall( cudaMemset(cflipBy, 127, nTris * sizeof(int)) ); 

        // Perform incircle tests
        block = dim3(WBLOCK); 
        grid = dim3(STRIPE, nTris / (STRIPE * block.x) + 1); 
        kernelNeedFlipping<<< grid, block >>>(ctriangles, covertices, cflipOri, cflipStep, 
            cflipBy, cmarker, nTris, step, cflag,ifEdgeIsConstraint); 
        cutilCheckError(); 

        cutilSafeCall( cudaMemcpy(&flag, cflag, sizeof(int), cudaMemcpyDeviceToHost) ); 

        if (flag > 0) 
        {          

            kernelUpdatePhase1<<< grid, block >>>(ctriangles, cflipOri, cflipStep, 
                cflipBy, cmarker, nTris, step,ifEdgeIsConstraint);	
            cutilCheckError();         

            kernelUpdatePhase2<<< grid, block >>>(ctriangles, cflipOri, cflipStep, 
                cflipBy, nTris, step); 
            cutilCheckError(); 

            kernelUpdatePhase3<<< grid, block >>>(ctriangles, cflipOri, cflipStep, 
                cflipBy, cmarker, nTris, step); 
            cutilCheckError(); 
        }
        step++; 


    } while (flag > 0);

    // All the suspectice incircle tests will be noted down 
    // and let the CPU perform the exact incircle tests.
    int marker = step - 1; 

    *suspective = (int *) malloc(nTris * sizeof(int)); 
    cutilSafeCall( cudaMemcpy(*suspective, cflipStep, nTris * sizeof(int), cudaMemcpyDeviceToHost) ); 

    ifEdgeIsConstraint_cpu =  new BYTE[3*nTris];
    cutilSafeCall( cudaMemcpy(ifEdgeIsConstraint_cpu, ifEdgeIsConstraint, 3*nTris * sizeof(BYTE), cudaMemcpyDeviceToHost) ); 

    cutilSafeCall( cudaFree(cmarker) ); 
    cutilSafeCall( cudaFree(cflipBy) ); 
    cutilSafeCall( cudaFree(cflipStep) ); 
    cutilSafeCall( cudaFree(cflipOri) ); 	
    cutilSafeCall( cudaFree(ifEdgeIsConstraint) );
    cutilSafeCall( cudaFree(cconstraints) );
    cutilSafeCall( cudaFree(cvertarr) );

    return marker; 	
}

