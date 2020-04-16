/*
Author: Rong Guodong, Stephanus, Cao Thanh Tung
Date: 20/07/2011

File Name: gpudt.h

Configurations, instructions and some useful macro of GPU-DT

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

/* 
 * GPU-DT - Constrained Delaunay Triangulation using Graphics Hardware
 *
 * GPU-DT is a C Library that utilizes graphics hardware to compute Delaunay 
 * triangulation. The result of the library calls is a triangle mesh. From each 
 * triangle in the triangle mesh you can get the 3 vertices and three neighbor triangles.
 *
 * Some edges in the triangulation can be specified before hand by providing 
 * "constraints". GPU-DT can also use the graphics hardware to compute the constrained
 * Delaunay triangulation efficiently. 
 *
 * Requirement: 
 * - CUDA Toolkit
 * - A GPU with minimum CUDA 1.1 compute capability (SINGLE PRECISION) or
 *                      CUDA 1.3 compute capability (DOUBLE PRECISION).
 *
 * By default, GPU-DT uses Double precision computation. On hardward that does not
 * support Double Precision, Single Precision computation can be used instead. 
 * To enable the use of Single Precision computation, turn on the SINGLE_PRECISION
 * flag below. You might also want to change the CUDA build rule to use -sm_11 instead
 * of -sm_13
 * 
 * Tested on NVIDIA Geforce 9500GT, GTX280, GTX 460, GTX 470 and Tesla C1070.
 */



#ifndef __GPUDT_H__
#define __GPUDT_H__

/***********************************************************************
 * By default, GPU-DT uses Double precision numbers. To switch to 
 * Single precision, turn on this definition.
 ***********************************************************************/
//#define SINGLE_PRECISION

#ifdef SINGLE_PRECISION
    #define REAL    float
#else
    #define REAL    double
#endif

#define MARKER			-32768

// Typedefs
typedef struct 
{
    REAL x, y;
} gpudtVertex;

typedef struct gpudtTriangle
{
    int             tri[3];         // 3 neighbor triangles, represented as an encoded gpudtOrientedTriangle
    int             vtx[3];         // 3 vertices of the triangle
    int				tmp[3];			// for internal use only
} gpudtTriangle;

// Input parameters for GPUDT. 
typedef struct 
{
    REAL            minX, minY;     // Min and max coordinates of the point list.
    REAL            maxX, maxY;

    int             nPoints;        // Number of points
    gpudtVertex    *points;         // The point list

	int             nConstraints;   // Number of constrains.
	int            *constraints;    // The constraint list: two point index per constraint

    int fboSize;                    // Texture size to be used (256, 512, 1024, 2048, 4096)

} GPUDTPARAMS, *PGPUDTPARAMS;

// Output of GPUDT
// - triangles: List of triangles in the final Delaunay triangulation.
// - vertices : Scaled input points. Points are scaled to fit the texture. 
// - nTris    : Number of triangles returned.
typedef struct
{
    gpudtTriangle *triangles;
    int nTris;
} GPUDTOUTPUT, *PGPUDTOUTPUT;

// Core Functions
PGPUDTOUTPUT gpudtComputeDT(PGPUDTPARAMS pParam);
void gpudtReleaseDTOutput(PGPUDTOUTPUT pGpudtOutput);

//****************************************************************************************************
// Internal Structure
//****************************************************************************************************
typedef struct
{
    int tri;
    int orient;
} gpudtOrientedTriangle;

/********* Mesh manipulation primitives begin here                   *********/
/**                                                                         **/
/**                                                                         **/

/* Fast lookup arrays to speed some of the mesh manipulation primitives.     */

const int gpudt_plus1mod3[3] = {1, 2, 0};
const int gpudt_minus1mod3[3] = {2, 0, 1};

#define org(triList, otri, vertexptr)                                                  \
    vertexptr = triList[(otri).tri].vtx[gpudt_plus1mod3[(otri).orient]]

#define dest(triList, otri, vertexptr)                                                 \
    vertexptr = triList[(otri).tri].vtx[gpudt_minus1mod3[(otri).orient]]

#define apex(triList, otri, vertexptr)                                                 \
    vertexptr = triList[(otri).tri].vtx[(otri).orient]


/* decode() converts a pointer to an oriented triangle.  The orientation is  */
/*   extracted from the two least significant bits of the pointer.           */
#define decode(ptr, otri)                                                     \
    (otri).orient = (unsigned int) (ptr) & 3;         \
    (otri).tri = (ptr) >> 2//(ptr) < 0 ? -1 : ((ptr) ^ (otri).orient)

/* encode() compresses an oriented triangle into a single pointer.  It       */
/*   relies on the assumption that all triangles are aligned to four-byte    */
/*   boundaries, so the two least significant bits of (otri).tri are zero.   */

#define Encode(otri)                                                          \
    (((otri).tri <<2)| (otri).orient)

/* Copy an oriented triangle.                                                */
#define otricopy(otri1, otri2)                                                \
	(otri2).tri = (otri1).tri;                                                  \
	(otri2).orient = (otri1).orient

/* Test for equality of oriented triangles.                                  */
#define otriequal(otri1, otri2)                                               \
	(((otri1).tri == (otri2).tri) &&                                            \
	((otri1).orient == (otri2).orient))

/* lnext() finds the next edge (counterclockwise) of a triangle.             */
#define lnext(otri1, otri2)                                                   \
	(otri2).tri = (otri1).tri;                                                  \
	(otri2).orient = gpudt_plus1mod3[(otri1).orient]

#define lnextself(otri)                                                       \
	(otri).orient = gpudt_plus1mod3[(otri).orient]

/* lprev() finds the previous edge (clockwise) of a triangle.                */
#define lprev(otri1, otri2)                                                   \
	(otri2).tri = (otri1).tri;                                                  \
	(otri2).orient = gpudt_minus1mod3[(otri1).orient]

#define lprevself(otri)                                                       \
	(otri).orient = gpudt_minus1mod3[(otri).orient]

/* sym() finds the opposite edge of a triangle */
#define sym(triList, otri1, otri2)                                                     \
    ptr = triList[(otri1).tri].tri[(otri1).orient];                                          \
    decode(ptr, otri2);

#define symself(triList, otri)                                                         \
    ptr = triList[(otri).tri].tri[(otri).orient];                                            \
    decode(ptr, otri);

/* onext() spins counterclockwise around a vertex; that is, it finds the     */
/*   next edge with the same origin in the counterclockwise direction.  This */
/*   edge is part of a different triangle.                                   */
#define onext(triList, otri1, otri2)                                                   \
	lprev(otri1, otri2);                                                        \
	symself(triList, otri2);

#define onextself(triList, otri)                                                       \
	lprevself(otri);                                                            \
	symself(triList, otri);

/* oprev() spins clockwise around a vertex; that is, it finds the next edge  */
/*   with the same origin in the clockwise direction.  This edge is part of  */
/*   a different triangle.                                                   */
#define oprev(triList, otri1, otri2)                                                   \
	sym(triList, otri1, otri2);                                                          \
	lnextself(otri2);

#define oprevself(triList, otri)                                                       \
	symself(triList, otri);                                                              \
	lnextself(otri);

#define setorg(triList, otri, vertexptr)                                               \
    triList[(otri).tri].vtx[gpudt_plus1mod3[(otri).orient]] = vertexptr

#define setdest(triList, otri, vertexptr)                                              \
    triList[(otri).tri].vtx[gpudt_minus1mod3[(otri).orient]] = vertexptr

#define setapex(triList, otri, vertexptr)                                              \
    triList[(otri).tri].vtx[(otri).orient] = vertexptr

/* Bond two triangles together.                                              */
#define bond(triList, otri1, otri2)                                                    \
    if ((otri1).tri >= 0) triList[(otri1).tri].tri[(otri1).orient] = Encode(otri2);                                \
    if ((otri2).tri >= 0) triList[(otri2).tri].tri[(otri2).orient] = Encode(otri1)

#endif
