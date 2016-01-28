/*
Author: Rong Guodong, Stephanus, Cao Thanh Tung, Qi Meng
Date: 20/07/2011

File Name: gpudt.cpp

Main code of the GPU-DT library

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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "gpudt.h"
#include "Cuda/cuda.h"

#ifndef _WIN32
#define index gpuIndex
#endif

#define BORDER_SIZE            1

//****************************************************************************************************
// Global Variables
//****************************************************************************************************

int        nStack;
int        nVerts, nPoints, nConstraints; 
int        gpudtSize; 

PGPUDTPARAMS    gpudtParams             = NULL;
PGPUDTOUTPUT    gpudtOutput             = NULL;
gpudtVertex     *gpudtVertices          = NULL;
gpudtTriangle   *gpudtTriangles         = NULL; 

int *index; 
BYTE *ifEdgeIsConstraint_cpu;

extern int *ctriangles;
extern REAL scale, shiftX, shiftY;

// Estimate the point list size after adding fake points
#define POINT_LIST_SIZE(x)    ((x) + (int(sqrt(REAL(x)) / 10) > 4 ? int(sqrt(REAL(x)) / 10) : 4))

//****************************************************************************************************
// Robust Functions (taken from Triangle - Jonathan Richard Shewchuk)
//****************************************************************************************************

#include "predicates.h"

bool flip(gpudtOrientedTriangle *flipedge, PGPUDTOUTPUT pOutput, BYTE* ifEdgeIsConstraint_cpu)
{    
   	gpudtOrientedTriangle botleft, botright;
    gpudtOrientedTriangle topleft, topright;
    gpudtOrientedTriangle top;
    gpudtOrientedTriangle botlcasing, botrcasing;
    gpudtOrientedTriangle toplcasing, toprcasing;
    int leftvertex, rightvertex, botvertex;
    int farvertex;
    int ptr;                         /* Temporary variable used by sym(). */   

    /* Identify the vertices of the quadrilateral. */
    org(gpudtTriangles, *flipedge, rightvertex);
    dest(gpudtTriangles, *flipedge, leftvertex);
    apex(gpudtTriangles, *flipedge, botvertex);
    sym(gpudtTriangles, *flipedge, top);
    apex(gpudtTriangles, top, farvertex);
	
    int pTri = (*flipedge).tri;
	int pOri = (*flipedge).orient;

	if (ifEdgeIsConstraint_cpu[pTri * 3 + pOri] == 1)
		return false;

	int pOppTri = top.tri;
	int pOppOri = top.orient;

	int temp0 = ifEdgeIsConstraint_cpu[pTri * 3 + (pOri + 0) % 3];          // old apex
	int temp1 = ifEdgeIsConstraint_cpu[pTri * 3 + (pOri + 1) % 3];          // old org
	int temp2 = ifEdgeIsConstraint_cpu[pTri * 3 + (pOri + 2) % 3];          // old dest
	int tempp0 = ifEdgeIsConstraint_cpu[pOppTri * 3 + (pOppOri + 0) % 3];   // old apex
	int tempp1 = ifEdgeIsConstraint_cpu[pOppTri * 3 + (pOppOri + 1) % 3];   // old org
	int tempp2 = ifEdgeIsConstraint_cpu[pOppTri * 3 + (pOppOri + 2) % 3];   // old dest
			
    /* Identify the casing of the quadrilateral. */
    lprev(top, topleft);
    sym(gpudtTriangles, topleft, toplcasing);
    lnext(top, topright);
    sym(gpudtTriangles, topright, toprcasing);
    lnext(*flipedge, botleft);
    sym(gpudtTriangles, botleft, botlcasing);
    lprev(*flipedge, botright);
    sym(gpudtTriangles, botright, botrcasing);

    /* Rotate the quadrilateral one-quarter turn counterclockwise. */
    bond(gpudtTriangles, topleft, botlcasing);
    bond(gpudtTriangles, botleft, botrcasing);
    bond(gpudtTriangles, botright, toprcasing);
    bond(gpudtTriangles, topright, toplcasing);

    /* New vertex assignments for the rotated quadrilateral. */
    setorg(gpudtTriangles, *flipedge, farvertex);
    setdest(gpudtTriangles, *flipedge, botvertex);
    setapex(gpudtTriangles, *flipedge, rightvertex);
    setorg(gpudtTriangles, top, botvertex);
    setdest(gpudtTriangles, top, farvertex);
    setapex(gpudtTriangles, top, leftvertex);
    
	ifEdgeIsConstraint_cpu[pTri * 3 + (pOri + 0) % 3] = -1;//apex
	ifEdgeIsConstraint_cpu[pTri * 3 + (pOri + 1) % 3] = temp2;//org
	ifEdgeIsConstraint_cpu[pTri * 3 + (pOri + 2) % 3] = tempp1;//dest

	ifEdgeIsConstraint_cpu[pOppTri * 3 + (pOppOri + 0) % 3] = -1;//apex
	ifEdgeIsConstraint_cpu[pOppTri * 3 + (pOppOri + 1) % 3] = tempp2;//org
	ifEdgeIsConstraint_cpu[pOppTri * 3 + (pOppOri + 2) % 3] = temp1;//dest		

	return true; 
}

//****************************************************************************************************
// Supporting Functions
//****************************************************************************************************
bool gpudtInCircle(gpudtVertex *v0, gpudtVertex *v1, gpudtVertex *v2, gpudtVertex *v3)
{
    REAL x0, y0, x1, y1, dot0, dot1;
    x0 = v0->x-v2->x;
    y0 = v0->y-v2->y;
    x1 = v1->x-v2->x;
    y1 = v1->y-v2->y;
    dot0 = (x0*x1+y0*y1);

    if (dot0>0)
    {
        x0 = v0->x-v3->x;
        y0 = v0->y-v3->y;
        x1 = v1->x-v3->x;
        y1 = v1->y-v3->y;
        dot1 = (x0*x1+y0*y1);

        if (dot1<=0)
        {
            return (incircle(v0, v1, v2, v3) > 0.0);
        }
        else {
            return false; // both angles are less than 90 degree, so incircle test must fail
        }
    }
    else
    {
        return (incircle(v0, v1, v2, v3) > 0.0);
    }
}

void gpudtRecursiveFlip(gpudtOrientedTriangle edge, PGPUDTOUTPUT pOutput, BYTE *ifEdgeIsConstraint_cpu)
{
	int ptr; 
    gpudtOrientedTriangle abut, oppoTri;    
    gpudtOrientedTriangle edges[4]; 
    int triOrigin, triDestination, triApex;        
    int oppoApex;

	sym(gpudtTriangles, edge, abut);

	int pTri1 = edge.tri;
	int pTri2 = abut.tri;
	int a1 = -1, a2 = -1, a3 = -1;
	int b1 = -1, b2 = -1, b3 = -1;

	a1 = ifEdgeIsConstraint_cpu[3 * pTri1 + 0];
	a2 = ifEdgeIsConstraint_cpu[3 * pTri1 + 1];
	a3 = ifEdgeIsConstraint_cpu[3 * pTri1 + 2]; 

	b1 = ifEdgeIsConstraint_cpu[3 * pTri2 + 0];
	b2 = ifEdgeIsConstraint_cpu[3 * pTri2 + 1];
	b3 = ifEdgeIsConstraint_cpu[3 * pTri2 + 2];

    if ( !flip(&edge, pOutput, ifEdgeIsConstraint_cpu) )
		return ;   

    sym(gpudtTriangles, edge, abut);

    // Get the four neighbors along the four edges
    lnext(edge, edges[0]);
    lnext(edges[0], edges[1]);

    lnext(abut, edges[2]);
    lnext(edges[2], edges[3]);

	a1 = ifEdgeIsConstraint_cpu[3*pTri1+0];
	a2 = ifEdgeIsConstraint_cpu[3*pTri1+1];
	a3 = ifEdgeIsConstraint_cpu[3*pTri1+2];

	b1 = ifEdgeIsConstraint_cpu[3*pTri2+0];
	b2 = ifEdgeIsConstraint_cpu[3*pTri2+1];
	b3 = ifEdgeIsConstraint_cpu[3*pTri2+2];

    // First triangle
    for (int i=0; i<2; i++)
    {
        sym(gpudtTriangles, edges[i], oppoTri); 

        if (oppoTri.tri >= 0) 
        {
            org(gpudtTriangles, edges[i], triOrigin);
            dest(gpudtTriangles, edges[i], triDestination);
            apex(gpudtTriangles, edges[i], triApex);

            apex(gpudtTriangles, oppoTri, oppoApex);

            int pTri = edge.tri;
			int tt=-1;
			
			if (i == 0) 
                tt = ifEdgeIsConstraint_cpu[3 * pTri + (edge.orient + 1) % 3];	

			if (i == 1) 
                tt = ifEdgeIsConstraint_cpu[3 * pTri + (edge.orient + 2) % 3];

			if (tt != 1)
			{
				if (gpudtInCircle(&gpudtParams->points[triOrigin], 
                    &gpudtParams->points[triDestination], 
					&gpudtParams->points[triApex], 
                    &gpudtParams->points[oppoApex])) 
                {			
					gpudtRecursiveFlip(oppoTri,pOutput,ifEdgeIsConstraint_cpu);
					break; 
				}
			}
        }
    }

    // Second triangle
    for (int i=2; i<4; i++)
    {
        sym(gpudtTriangles, edges[i], oppoTri); 

        if (oppoTri.tri >= 0)
        {
            org(gpudtTriangles, edges[i], triOrigin);
            dest(gpudtTriangles, edges[i], triDestination);
            apex(gpudtTriangles, edges[i], triApex);
            apex(gpudtTriangles, oppoTri, oppoApex);

			int pTri = abut.tri;
			int tt = -1;

			if (i == 2) 
                tt = ifEdgeIsConstraint_cpu[3 * pTri + (abut.orient + 1) % 3];	

			if (i == 3) 
                tt = ifEdgeIsConstraint_cpu[3 * pTri + (abut.orient + 2) % 3];	

			if (tt != 1)
			{		
				if (gpudtInCircle(&gpudtParams->points[triOrigin], 
                    &gpudtParams->points[triDestination], 
					&gpudtParams->points[triApex], 
                    &gpudtParams->points[oppoApex])) 
                {
					gpudtRecursiveFlip(oppoTri,pOutput,ifEdgeIsConstraint_cpu);
					break; 
				}
			}						
        }
    }
}

//****************************************************************************************************
// Core Functions
//****************************************************************************************************

#define SCALEX(x) (( ( ((x) - shiftX) / scale )))
#define SCALEY(y) (( ( ((y) - shiftY) / scale )))

#define GETID(idresult, i, j) \
{ \
    int position; \
    if (j == BORDER_SIZE) \
        position = (gpudtSize * 0 + i); \
    else if (i == BORDER_SIZE) \
        position = (gpudtSize * 1 + j); \
    else if (j == maxj) \
        position = (gpudtSize * 2 + i); \
    else \
        position = (gpudtSize * 3 + j); \
    idresult = boundary[position]; \
}

/***********************************************************************
 * Fix the convex hull of the triangle mesh given the boundary points. 
 * New triangles are stored in the given array (allocated). 
 ***********************************************************************/ 
#define ADD_TRIANGLE(id0, id1, id2) \
    additionalTriangles[count * 3    ] = id2; \
    additionalTriangles[count * 3 + 1] = id0; \
    additionalTriangles[count * 3 + 2] = id1; \
    count++

REAL skewness(gpudtVertex *a, gpudtVertex *b, gpudtVertex *x) 
{
    REAL ux = x->x - floor(a->x), uy = x->y - floor(a->y);  
    REAL vx = floor(b->x) - floor(a->x), vy = floor(b->y) - floor(a->y);  
    REAL dot = ux * vx + uy * vy; 

    if (dot != 0)
        return abs(dot / (sqrt(ux * ux + uy * uy) * sqrt(vx * vx + vy * vy))); 
    else
        return 0.0; 
}

int gpudtFixConvexHull(int *additionalTriangles, int fboWidth, int *boundary)
{
    // Scale the point set first
    gpudtVertex *vtPtr = &gpudtVertices[0]; 
    gpudtVertex *ptPtr = &gpudtParams->points[0]; 

    for (int i = 0; i < nPoints; i++) {
        vtPtr->x = shiftX + floor(SCALEX(ptPtr->x)) * scale; 
        vtPtr->y = shiftY + floor(SCALEY(ptPtr->y)) * scale; 
        vtPtr++; ptPtr++; 
	}

    int *gpudtStack = new int[2*(fboWidth + fboWidth)];
    int i, j; 
    int count = 0; 
    int maxi = gpudtSize - 2, maxj = gpudtSize - 2; 
    int iCurrentId, iLastId, iFirstId, iLowestPos;
    int iPrevPrev, iPrev;
    REAL rLowestY;

    nStack = 0;

    // Find lowest Id to start
    GETID(iCurrentId, BORDER_SIZE, BORDER_SIZE);

    rLowestY    = gpudtVertices[iCurrentId].y;
    iLowestPos  = BORDER_SIZE;
    iLastId     = iCurrentId;

    for (i=1; i <= maxi; ++i)
    {        
        GETID(iCurrentId, i, BORDER_SIZE);
        if (iCurrentId != iLastId && 
            gpudtVertices[iCurrentId].y < rLowestY) {
            rLowestY = gpudtVertices[iCurrentId].y;
            iLowestPos = i;
        }
    }

    GETID(iCurrentId, iLowestPos, BORDER_SIZE);
    gpudtStack[nStack++] = iLastId = iFirstId = iCurrentId;

    // Traverse Lower Hull (iLowestPos to right)
    for (i=iLowestPos+1; i<=maxi; ++i)
    {        
        GETID(iCurrentId, i, BORDER_SIZE);
        if (iCurrentId != iLastId)
        {
            if (nStack<2)
            {
                gpudtStack[nStack++] = iLastId = iCurrentId;
            }
            else
            {
                iPrevPrev       = gpudtStack[nStack - 2];
                iPrev           = gpudtStack[nStack - 1];

                while (counterclockwise(&gpudtVertices[iPrevPrev], 
                    &gpudtVertices[iPrev], &gpudtVertices[iCurrentId]) < 0.0)
                {
                    ADD_TRIANGLE(iPrevPrev, iCurrentId, iPrev);

                    // Pop
                    --nStack;
                    if (nStack < 2)
                        break;

                    // Get new previous two in the stack
                    iPrev           = iPrevPrev;
                    iPrevPrev       = gpudtStack[nStack-2];
                }
                gpudtStack[nStack++] = iLastId = iCurrentId;
            }
        }
    }

    // Traverse Right Hull (bottom to top)
    for (j=BORDER_SIZE; j<=maxj; ++j)
    {
        GETID(iCurrentId, maxi, j);
        if (iCurrentId != iLastId)
        {
            if (nStack<2)
            {
                gpudtStack[nStack++] = iLastId = iCurrentId;
            }
            else
            {
                iPrevPrev       = gpudtStack[nStack - 2];
                iPrev           = gpudtStack[nStack - 1];

                while (counterclockwise(&gpudtVertices[iPrevPrev], 
                    &gpudtVertices[iPrev], &gpudtVertices[iCurrentId]) < 0.0)
                {
                    ADD_TRIANGLE(iPrevPrev, iCurrentId, iPrev);

                    // Pop
                    --nStack;
                    if (nStack < 2)
                        break;

                    // Get new previous two in the stack
                    iPrev           = iPrevPrev;
                    iPrevPrev       = gpudtStack[nStack-2];
                }
                gpudtStack[nStack++] = iLastId = iCurrentId;
            }
        }
    }

    // Traverse Upper Hull (right to left)
    for (int i=maxi-1; i>=BORDER_SIZE; --i)
    {
        GETID(iCurrentId, i, maxj);
        if (iCurrentId != iLastId)
        {
            if (nStack<2)
            {
                gpudtStack[nStack++] = iLastId = iCurrentId;
            }
            else
            {
                iPrevPrev       = gpudtStack[nStack - 2];
                iPrev           = gpudtStack[nStack - 1];

                while (counterclockwise(&gpudtVertices[iPrevPrev], 
                    &gpudtVertices[iPrev], &gpudtVertices[iCurrentId]) < 0.0)
                {
                    ADD_TRIANGLE(iPrevPrev, iCurrentId, iPrev);

                    // Pop
                    --nStack;
                    if (nStack < 2)
                        break;

                    // Get new previous two in the stack
                    iPrev           = iPrevPrev;
                    iPrevPrev       = gpudtStack[nStack-2];
                }
                gpudtStack[nStack++] = iLastId = iCurrentId;
            }
        }
    }

    // Traverse Left Hull (top to bottom)
    for (int j=maxj-1; j>=BORDER_SIZE; --j)
    {
        GETID(iCurrentId, BORDER_SIZE, j);
        if (iCurrentId != iLastId)
        {
            if (nStack<2)
            {
                gpudtStack[nStack++] = iLastId = iCurrentId;
            }
            else
            {
                iPrevPrev       = gpudtStack[nStack - 2];
                iPrev           = gpudtStack[nStack - 1];

                while (counterclockwise(&gpudtVertices[iPrevPrev], 
                    &gpudtVertices[iPrev], &gpudtVertices[iCurrentId]) < 0.0)
                {
                    ADD_TRIANGLE(iPrevPrev, iCurrentId, iPrev);

                    // Pop
                    --nStack;
                    if (nStack < 2)
                        break;

                    // Get new previous two in the stack
                    iPrev           = iPrevPrev;
                    iPrevPrev       = gpudtStack[nStack-2];
                }
                gpudtStack[nStack++] = iLastId = iCurrentId;
            }
        }
    }

    // Traverse Lower Hull (left to iLowestPos)
    for (i=BORDER_SIZE; i<=iLowestPos; ++i)
    {        
        GETID(iCurrentId, i, BORDER_SIZE);
        if (iCurrentId != iLastId)
        {
            if (nStack<2)
            {
                gpudtStack[nStack++] = iLastId = iCurrentId;
            }
            else
            {
                iPrevPrev       = gpudtStack[nStack - 2];
                iPrev           = gpudtStack[nStack - 1];

                while (counterclockwise(&gpudtVertices[iPrevPrev], 
                    &gpudtVertices[iPrev], &gpudtVertices[iCurrentId]) < 0.0)
                {
                    ADD_TRIANGLE(iPrevPrev, iCurrentId, iPrev);

                    // Pop
                    --nStack;
                    if (nStack < 2)
                        break;

                    // Get new previous two in the stack
                    iPrev           = iPrevPrev;
                    iPrevPrev       = gpudtStack[nStack-2];
                }
                gpudtStack[nStack++] = iLastId = iCurrentId;
            }
        }
    }

    if (iLastId != iFirstId)
    {
        iPrevPrev       = gpudtStack[nStack - 2];
        iPrev           = gpudtStack[nStack - 1];

        while (counterclockwise(&gpudtVertices[iPrevPrev], 
            &gpudtVertices[iPrev], &gpudtVertices[iCurrentId]) < 0.0)
        {
            ADD_TRIANGLE(iPrevPrev, iCurrentId, iPrev);

            // Pop
            --nStack;
            if (nStack < 2)
                break;

            // Get new previous two in the stack
            iPrev           = iPrevPrev;
            iPrevPrev       = gpudtStack[nStack-2];
        }
        gpudtStack[nStack++] = iFirstId;
    }

    // Add a few fake sites and construct a fake boundary for the triangle mesh
    int start = 0, pin = 0, nextpin = 1; 
    bool stop = false; 
    int fakeBoundarySize = nVerts - nPoints; 
    int usedUp = 0; 
    REAL ratio = ((REAL) nStack) / fakeBoundarySize; 

    for (int i = 0; i < nStack-1; i++) {
        int p1 = gpudtStack[i], p2 = gpudtStack[i+1]; 

        if (!stop) {
            bool move = true; 

            do {
                REAL ccw1 = counterclockwise(&gpudtVertices[p1], 
                    &gpudtVertices[index[pin]], &gpudtVertices[p2]); 
                REAL ccw2 = counterclockwise(&gpudtVertices[p1], 
                    &gpudtVertices[index[nextpin]], &gpudtVertices[p2]); 

                if (ccw1 > 0) 
                    if (ccw2 <= 0)
                        move = false; 
                    else {
                        REAL d1 = skewness(&gpudtVertices[p1], &gpudtVertices[p2], 
                            &gpudtVertices[index[pin]]);                             
                        REAL d2 = skewness(&gpudtVertices[p1], &gpudtVertices[p2], 
                            &gpudtVertices[index[nextpin]]);                             

                        if (d1 <= d2 && i <= ratio * usedUp)
                            move = false; 
                    }                

                if (move) {
                    if (i != 0) {
                        {ADD_TRIANGLE(p1, index[pin], index[nextpin]);} 
                        usedUp++; 
                    }
                    pin = nextpin; 
                    nextpin = (nextpin + 1) % (fakeBoundarySize); 
                
                    if (pin == start) {
                        stop = true; 
                        break; 
                    }
                } else {
                }
            } while (move); 
        }

        ADD_TRIANGLE(p1, index[pin], p2); 

        if (i == 0)
            start = pin; 
    }

    while (pin != start) {
        ADD_TRIANGLE(gpudtStack[0], index[pin], index[nextpin]); 
        pin = nextpin; 
        nextpin = (nextpin + 1) % (fakeBoundarySize); 
    }

    free(index); free(gpudtStack); 

    return count; 
}

#define MIN(a, b)       ((a) > (b) ? (b) : (a))
#define MAX(a, b)       ((a) > (b) ? (a) : (b))

REAL findMostSignificantBit(REAL value)
{
    REAL half = 0.5; 
    REAL digit = 1.0; 

    while (digit <= value) 
        digit *= 2.0; 

    while (digit > value) 
        digit *= half; 

    return digit; 
}

void reducePrecision(REAL &value, int precision, bool bigger)
{
    REAL half = 0.5; 
    REAL input = fabs(value); 
    REAL output = 0.0; 
    REAL digit = findMostSignificantBit(input); 

    if (precision <= 0)
    {
        if (bigger)
            output = digit * 2.0; 
        else
            output = 0.0; 
    }
	else
	{
		for (int i = 0; i < precision; i++) 
		{
			if (input >= digit) 
			{
				output += digit; 
				input -= digit; 
			}

			digit *= half; 
		}

		if (input > 0.0 && (bigger && value >= 0.0) || (!bigger && value < 0.0)) 
			output += digit * 2.0; 
	}

    if (value < 0.0)
        output = -output; 

    value = output; 
}

// Truncate some less significant bits of dest so that
// it's no more precise than source. 
int truncatePrec(REAL source, REAL dest, int maxPrecision) 
{
    if (dest >= source)
        return maxPrecision; 

    REAL half = 0.5; 
    REAL digit = findMostSignificantBit(source);

    int shift = 0; 

    for (; shift < maxPrecision; shift++, digit *= half)
        if (digit <= dest)
            break; 

    return maxPrecision - shift; 
}

void computeScalingFactors(int texSize, REAL &scale, REAL &shiftX, REAL &shiftY)
{
    REAL lastcheck;
    int precision = 0; 

    REAL gpudt_epsilon  = 1.0;
    REAL half           = 0.5;
    REAL check          = 1.0;

    do {
        lastcheck = check;
        gpudt_epsilon *= half;
        check = 1.0 + gpudt_epsilon;
        precision++;
    } while ((check != 1.0) && (check != lastcheck));

    int logTex = int(log((double)texSize) / log(2.0)); 

    REAL minX = gpudtParams->minX; 
    REAL maxX = gpudtParams->maxX; 
    REAL minY = gpudtParams->minY; 
    REAL maxY = gpudtParams->maxY;
    
    shiftX = minX; shiftY = minY; 
    scale = MAX(maxX - minX, maxY - minY) / (texSize - 3); 

    int shiftXPrec = precision, shiftYPrec = precision, scalePrec = precision; 

    REAL range = scale * texSize;       // No precision loss since texSize is a power of 2

    // X first
    if (fabs(shiftX) < range)         // Small shift, we try to cut precision on shiftX
        shiftXPrec = truncatePrec(range, fabs(shiftX), precision); 
    else
        scalePrec = MIN(scalePrec, truncatePrec(fabs(shiftX), range, precision)); 

    // then Y
    if (fabs(shiftY) < range)         // Small shift, we try to cut precision on shiftX
        shiftYPrec = truncatePrec(range, fabs(shiftY), precision); 
    else
        scalePrec = MIN(scalePrec, truncatePrec(fabs(shiftY), range, precision)); 

    reducePrecision(shiftX, shiftXPrec - 2, false);
    reducePrecision(shiftY, shiftYPrec - 2, false);

    scale = MAX(maxX - shiftX, maxY - shiftY) / (texSize - 3); 
    reducePrecision(scale, scalePrec - logTex - 2, true);
}

PGPUDTOUTPUT gpudtComputeDT(PGPUDTPARAMS pParam)
{
   	
	gpudtOrientedTriangle triangleLoop;
    gpudtOrientedTriangle oppoTri;    
    int triOrigin, triDestination, triApex, oppoApex;
    int ptr;    // temporary variable used by sym()        

    PGPUDTOUTPUT pOutput = new GPUDTOUTPUT;

    // Save params to global var
    gpudtOutput = pOutput;
    gpudtParams = pParam;

    // Exact Arithmetic init
    gpudt_exactinit();
    cudaExactInit(); 

	gpudtSize               = pParam->fboSize;
    nPoints                 = pParam->nPoints; 
    nConstraints            = pParam->nConstraints;
    nVerts                  = POINT_LIST_SIZE(nPoints);  // Include fake boundary

    // Allocate Memory
	cudaAllocation(); 

	gpudtVertices   = new gpudtVertex[nVerts];
    gpudtTriangles  = new gpudtTriangle[2 * nVerts];

    pOutput->triangles = gpudtTriangles;     

    // Scale input points
    computeScalingFactors(gpudtParams->fboSize, scale, shiftX, shiftY); 

    // Copy point set to the internal array
    memcpy(gpudtVertices, gpudtParams->points, nPoints * sizeof(gpudtVertex)); 

    // Add fake sites around the set
    // On average, one fake site is connected to 10 convex hull sites. 
    int fakeBoundarySize = nVerts - nPoints;    

    // Shuffle the fake sites
    index = new int[fakeBoundarySize]; 

    for (int i = 0; i < fakeBoundarySize; i++)
        index[i] = nPoints + i; 

    for (int i = 0; i < fakeBoundarySize; i++) {
        int u = rand() % fakeBoundarySize; 
        int v = rand() % fakeBoundarySize; 

        int tmp = index[u]; index[u] = index[v]; index[v] = tmp; 
    }

    REAL cx = (shiftX + gpudtParams->maxX) / 2.0; 
    REAL cy = (shiftY + gpudtParams->maxY) / 2.0; 
    REAL radius = MAX((gpudtParams->maxX - shiftX), (gpudtParams->maxY - shiftY)); 

    for (int i = 0; i < fakeBoundarySize; i++) {
        gpudtVertices[index[i]].x = cx + radius * 1.1 * cos(i * 2 * 3.1416 / fakeBoundarySize); 
        gpudtVertices[index[i]].y = cy + radius * 1.1 * sin(i * 2 * 3.1416 / fakeBoundarySize); 
    }
	
    // Cuda Initialization
    cudaInitialize(); 

    //************************************************************************************************
	// Compute the discrete voronoi diagram
    //************************************************************************************************
	cudaDiscreteVoronoiDiagram(); 
        
    //************************************************************************************************
    // Reconstruction
    //************************************************************************************************
	cudaReconstruction();
	
    //**********************************************************************************************
    // Shifting
    //**********************************************************************************************
	cudaShifting();
	
    //**********************************************************************************************
    // Insert missing sites
    //**********************************************************************************************
	cudaMissing();
    
	//**********************************************************************************************
    // Insert constraints
    //**********************************************************************************************
	cudaConstraint();	
	    
	//**********************************************************************************************
    // Remove boundary
    //**********************************************************************************************
	cudaFixBoundary(); 
	
    //**********************************************************************************************
    // Edge flipping
    //**********************************************************************************************
	int *suspective;
	int marker = cudaFlipping(&suspective);

	cudaFinalize();
 
	// We don't want to perform robust incircle test in CUDA, so for those ambiguous triangles, 
	// we perform a recursive flipping in CPU. 

    // Run through the list of triangles, checking each one    
    for (int ii = 0; ii<pOutput->nTris; ++ii)
        if (suspective[ii] == -marker ) 
        {
            triangleLoop.tri = ii; 

            // Check all three edges of the triangle 
            for (triangleLoop.orient=0; triangleLoop.orient<3; ++triangleLoop.orient)
            {
                sym(gpudtTriangles, triangleLoop, oppoTri);

                if (oppoTri.tri >= 0)
                {
                    // current triangle origin, destination and apex
                    org(gpudtTriangles, triangleLoop, triOrigin);
                    dest(gpudtTriangles, triangleLoop, triDestination);
                    apex(gpudtTriangles, triangleLoop, triApex);			
                    
                    // opposite triangle apex
                    apex(gpudtTriangles, oppoTri, oppoApex);
                    if (ifEdgeIsConstraint_cpu[ii*3 + triangleLoop.orient] != 1) // not a constraint;					
                    {						
                        if (gpudtInCircle(&gpudtParams->points[triOrigin], 
                            &gpudtParams->points[triDestination], 
                            &gpudtParams->points[triApex], 
                            &gpudtParams->points[oppoApex]))		
                        {						
                            gpudtRecursiveFlip(triangleLoop,pOutput,ifEdgeIsConstraint_cpu);
                            break;
                        }
                    } // constraint	

                } // has a neighbor

            } // loop 3 orientations

        } // suspective triangle

    free(suspective); 

	// Deallocate GPU memory
	cudaDeallocation(); 	

	delete [] gpudtVertices;
	delete [] ifEdgeIsConstraint_cpu;
	return pOutput;   
}

void gpudtReleaseDTOutput(PGPUDTOUTPUT pGpudtOutput)
{
    delete [] pGpudtOutput->triangles;
}

