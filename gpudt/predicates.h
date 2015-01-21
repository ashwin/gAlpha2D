//*****************************************************************************
// Extracted and adapted by Cao Thanh Tung
// School of Computing, National University of Singapore. 
// Date: 26/01/2009
//
// Note: Some variable and method names are changed to avoid conflicts
//*****************************************************************************

/*****************************************************************************/
/*                                                                           */
/*  Routines for Arbitrary Precision Floating-point Arithmetic               */
/*  and Fast Robust Geometric Predicates                                     */
/*  (predicates.c)                                                           */
/*                                                                           */
/*  May 18, 1996                                                             */
/*                                                                           */
/*  Placed in the public domain by                                           */
/*  Jonathan Richard Shewchuk                                                */
/*  School of Computer Science                                               */
/*  Carnegie Mellon University                                               */
/*  5000 Forbes Avenue                                                       */
/*  Pittsburgh, Pennsylvania  15213-3891                                     */
/*  jrs@cs.cmu.edu                                                           */
/*                                                                           */
/*  This file contains C implementation of algorithms for exact addition     */
/*    and multiplication of floating-point numbers, and predicates for       */
/*    robustly performing the orientation and incircle tests used in         */
/*    computational geometry.  The algorithms and underlying theory are      */
/*    described in Jonathan Richard Shewchuk.  "Adaptive Precision Floating- */
/*    Point Arithmetic and Fast Robust Geometric Predicates."  Technical     */
/*    Report CMU-CS-96-140, School of Computer Science, Carnegie Mellon      */
/*    University, Pittsburgh, Pennsylvania, May 1996.  (Submitted to         */
/*    Discrete & Computational Geometry.)                                    */
/*                                                                           */
/*  This file, the paper listed above, and other information are available   */
/*    from the Web page http://www.cs.cmu.edu/~quake/robust.html .           */
/*                                                                           */
/*****************************************************************************/

#ifndef PREDICATES_H
#define PREDICATES_H

#pragma warning (disable:4244)

#define INEXACT

#include "gpudt.h"

//********* Geometric primitives begin here                           *********
//**                                                                         **
//**                                                                         **
//
//* The adaptive exact arithmetic geometric predicates implemented herein are *
//*   described in detail in my paper, "Adaptive Precision Floating-Point     *
//*   Arithmetic and Fast Robust Geometric Predicates."  See the header for a *
//*   full citation.                                                          *
//
//* Which of the following two methods of finding the absolute values is      *
//*   fastest is compiler-dependent.  A few compilers can inline and optimize *
//*   the fabs() call; but most will incur the overhead of a function call,   *
//*   which is disastrously slow.  A faster way on IEEE machines might be to  *
//*   mask the appropriate bit, but that's difficult to do in C without       *
//*   forcing the value to be stored to memory (rather than be kept in the    *
//*   register to which the optimizer assigned it).                           *

#define Absolute(a)  ((a) >= 0.0 ? (a) : -(a))
// #define Absolute(a)  fabs(a)

//* Many of the operations are broken up into two pieces, a main part that    *
//*   performs an approximate operation, and a "tail" that computes the       *
//*   roundoff error of that operation.                                       *
//*                                                                           *
//* The operations Fast_Two_Sum(), Fast_Two_Diff(), Two_Sum(), Two_Diff(),    *
//*   Split(), and Two_Product() are all implemented as described in the      *
//*   reference.  Each of these macros requires certain variables to be       *
//*   defined in the calling routine.  The variables `bvirt', `c', `abig',    *
//*   `_i', `_j', `_k', `_l', `_m', and `_n' are declared `INEXACT' because   *
//*   they store the result of an operation that may incur roundoff error.    *
//*   The input parameter `x' (or the highest numbered `x_' parameter) must   *
//*   also be declared `INEXACT'.                                             *

#define Fast_Two_Sum_Tail(a, b, x, y) \
    bvirt = x - a; \
    y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
    x = (REAL) (a + b); \
    Fast_Two_Sum_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y) \
    bvirt = (REAL) (x - a); \
    avirt = x - bvirt; \
    bround = b - bvirt; \
    around = a - avirt; \
    y = around + bround

#define Two_Sum(a, b, x, y) \
    x = (REAL) (a + b); \
    Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y) \
    bvirt = (REAL) (a - x); \
    avirt = x + bvirt; \
    bround = bvirt - b; \
    around = a - avirt; \
    y = around + bround

#define Two_Diff(a, b, x, y) \
    x = (REAL) (a - b); \
    Two_Diff_Tail(a, b, x, y)

#define Split(a, ahi, alo) \
    c = (REAL) (gpudt_splitter * a); \
    abig = (REAL) (c - a); \
    ahi = c - abig; \
    alo = a - ahi

#define Two_Product_Tail(a, b, x, y) \
    Split(a, ahi, alo); \
    Split(b, bhi, blo); \
    err1 = x - (ahi * bhi); \
    err2 = err1 - (alo * bhi); \
    err3 = err2 - (ahi * blo); \
    y = (alo * blo) - err3

#define Two_Product(a, b, x, y) \
    x = (REAL) (a * b); \
    Two_Product_Tail(a, b, x, y)

// Two_Product_Presplit() is Two_Product() where one of the inputs has       
//   already been split.  Avoids redundant splitting.                        

#define Two_Product_Presplit(a, b, bhi, blo, x, y) \
    x = (REAL) (a * b); \
    Split(a, ahi, alo); \
    err1 = x - (ahi * bhi); \
    err2 = err1 - (alo * bhi); \
    err3 = err2 - (ahi * blo); \
    y = (alo * blo) - err3

// Square() can be done more quickly than Two_Product().                     

#define Square_Tail(a, x, y) \
    Split(a, ahi, alo); \
    err1 = x - (ahi * ahi); \
    err3 = err1 - ((ahi + ahi) * alo); \
    y = (alo * alo) - err3

#define Square(a, x, y) \
    x = (REAL) (a * a); \
    Square_Tail(a, x, y)

// Macros for summing expansions of various fixed lengths.  These are all    
//   unrolled versions of Expansion_Sum().                                   

#define Two_One_Sum(a1, a0, b, x2, x1, x0) \
    Two_Sum(a0, b , _i, x0); \
    Two_Sum(a1, _i, x2, x1)

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
    Two_Diff(a0, b , _i, x0); \
    Two_Sum( a1, _i, x2, x1)

#define Two_Two_Sum(a1, a0, b1, b0, x3, x2, x1, x0) \
    Two_One_Sum(a1, a0, b0, _j, _0, x0); \
    Two_One_Sum(_j, _0, b1, x3, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
    Two_One_Diff(a1, a0, b0, _j, _0, x0); \
    Two_One_Diff(_j, _0, b1, x3, x2, x1)

// Macro for multiplying a two-component expansion by a single component.    

#define Two_One_Product(a1, a0, b, x3, x2, x1, x0) \
    Split(b, bhi, blo); \
    Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
    Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
    Two_Sum(_i, _0, _k, x1); \
    Fast_Two_Sum(_j, _k, x3, x2)

//*****************************************************************************
//* FUNCTIONS                                                                 *
//*****************************************************************************
void gpudt_exactinit();

REAL counterclockwise(gpudtVertex *pa, gpudtVertex *pb, gpudtVertex *pc);

REAL incircle(gpudtVertex *pa, gpudtVertex *pb, gpudtVertex *pc, gpudtVertex *pd);

#endif