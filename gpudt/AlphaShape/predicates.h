/*
Author: Srinivasan Kidambi Sridharan, Ashwin Nanjappa, Cao Thanh Tung

Copyright (c) 2012, School of Computing, National University of Singapore. 
All rights reserved.

If you use GAlpha 1.0 and you like it or have comments on its usefulness etc., we 
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

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

/*****************************************************************************/
/*                                                                           */
/*  Routines for Arbitrary Precision Floating-gpudtVertex Arithmetic               */
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
/*    and multiplication of floating-gpudtVertex numbers, and predicates for       */
/*    robustly performing the orientation and incircle tests used in         */
/*    computational geometry.  The algorithms and underlying theory are      */
/*    described in Jonathan Richard Shewchuk.  "Adaptive Precision Floating- */
/*    gpudtVertex Arithmetic and Fast Robust Geometric Predicates."  Technical     */
/*    Report CMU-CS-96-140, School of Computer Science, Carnegie Mellon      */
/*    University, Pittsburgh, Pennsylvania, May 1996.  (Submitted to         */
/*    Discrete & Computational Geometry.)                                    */
/*                                                                           */
/*  This file, the paper listed above, and other information are available   */
/*    from the Web page http://www.cs.cmu.edu/~quake/robust.html .           */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/*                                                                           */
/*  Using this code:                                                         */
/*                                                                           */
/*  First, read the short or long version of the paper (from the Web page    */
/*    above).                                                                */
/*                                                                           */
/*  Be sure to call exactinit() once, before calling any of the arithmetic   */
/*    functions or geometric predicates.  Also be sure to turn on the        */
/*    optimizer when compiling this file.                                    */
/*                                                                           */
/*                                                                           */
/*  Several geometric predicates are defined.  Their parameters are all      */
/*    gpudtVertexs.  Each gpudtVertex is an array of two or three floating-gpudtVertex         */
/*    numbers.  The geometric predicates, described in the papers, are       */
/*                                                                           */
/*    orient2d(pa, pb, pc)                                                   */
/*    orient2dfast(pa, pb, pc)                                               */
/*    orient3d(pa, pb, pc, pd)                                               */
/*    orient3dfast(pa, pb, pc, pd)                                           */
/*    incircle(pa, pb, pc, pd)                                               */
/*    incirclefast(pa, pb, pc, pd)                                           */
/*    insphere(pa, pb, pc, pd, pe)                                           */
/*    inspherefast(pa, pb, pc, pd, pe)                                       */
/*                                                                           */
/*  Those with suffix "fast" are approximate, non-robust versions.  Those    */
/*    without the suffix are adaptive precision, robust versions.  There     */
/*    are also versions with the suffices "exact" and "slow", which are      */
/*    non-adaptive, exact arithmetic versions, which I use only for timings  */
/*    in my arithmetic papers.                                               */
/*                                                                           */
/*                                                                           */
/*  An expansion is represented by an array of floating-gpudtVertex numbers,       */
/*    sorted from smallest to largest magnitude (possibly with interspersed  */
/*    zeros).  The length of each expansion is stored as a separate integer, */
/*    and each arithmetic function returns an integer which is the length    */
/*    of the expansion it created.                                           */
/*                                                                           */
/*  Several arithmetic functions are defined.  Their parameters are          */
/*                                                                           */
/*    e, f           Input expansions                                        */
/*    elen, flen     Lengths of input expansions (must be >= 1)              */
/*    h              Output expansion                                        */
/*    b              Input scalar                                            */
/*                                                                           */
/*  The arithmetic functions are                                             */
/*                                                                           */
/*    grow_expansion(elen, e, b, h)                                          */
/*    grow_expansion_zeroelim(elen, e, b, h)                                 */
/*    expansion_sum(elen, e, flen, f, h)                                     */
/*    expansion_sum_zeroelim1(elen, e, flen, f, h)                           */
/*    expansion_sum_zeroelim2(elen, e, flen, f, h)                           */
/*    fast_expansion_sum(elen, e, flen, f, h)                                */
/*    fast_expansion_sum_zeroelim(elen, e, flen, f, h)                       */
/*    linear_expansion_sum(elen, e, flen, f, h)                              */
/*    linear_expansion_sum_zeroelim(elen, e, flen, f, h)                     */
/*    scale_expansion(elen, e, b, h)                                         */
/*    scale_expansion_zeroelim(elen, e, b, h)                                */
/*    compress(elen, e, h)                                                   */
/*                                                                           */
/*  All of these are described in the long version of the paper; some are    */
/*    described in the short version.  All return an integer that is the     */
/*    length of h.  Those with suffix _zeroelim perform zero elimination,    */
/*    and are recommended over their counterparts.  The procedure            */
/*    fast_expansion_sum_zeroelim() (or linear_expansion_sum_zeroelim() on   */
/*    processors that do not use the round-to-even tiebreaking rule) is      */
/*    recommended over expansion_sum_zeroelim().  Each procedure has a       */
/*    little note next to it (in the code below) that tells you whether or   */
/*    not the output expansion may be the same array as one of the input     */
/*    expansions.                                                            */
/*                                                                           */
/*                                                                           */
/*  If you look around below, you'll also find macros for a bunch of         */
/*    simple unrolled arithmetic operations, and procedures for printing     */
/*    expansions (commented out because they don't work with all C           */
/*    compilers) and for generating random floating-gpudtVertex numbers whose      */
/*    significand bits are all random.  Most of the macros have undocumented */
/*    requirements that certain of their parameters should not be the same   */
/*    variable; for safety, better to make sure all the parameters are       */
/*    distinct variables.  Feel free to send email to jrs@cs.cmu.edu if you  */
/*    have questions.                                                        */
/*                                                                           */
/*****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

	/* On some machines, the exact arithmetic routines might be defeated by the  */
	/*   use of internal extended precision floating-gpudtVertex registers.  Sometimes */
	/*   this problem can be fixed by defining certain values to be volatile,    */
	/*   thus forcing them to be stored to memory and rounded off.  This isn't   */
	/*   a great solution, though, as it slows the arithmetic down.              */
	/*                                                                           */
	/* To try this out, write "#define INEXACT volatile" below.  Normally,       */
	/*   however, INEXACT should be defined to be nothing.  ("#define INEXACT".) */

#define INEXACT                          /* Nothing */
	/* #define INEXACT volatile */

#define MAX_N 200
#define REAL double	                     /* float or double */
#define MAX(a,b) (a>b?a:b)
#define REALPRINT doubleprint
#define REALRAND doublerand
#define NARROWRAND narrowdoublerand
#define UNIFORMRAND uniformdoublerand

	/* Which of the following two methods of finding the absolute values is      */
	/*   fastest is compiler-dependent.  A few compilers can inline and optimize */
	/*   the fabs() call; but most will incur the overhead of a function call,   */
	/*   which is disastrously slow.  A faster way on IEEE machines might be to  */
	/*   mask the appropriate bit, but that's difficult to do in C.              */

#define Absolute(a)  ((a) >= 0.0 ? (a) : -(a))
	/* #define Absolute(a)  fabs(a) */

	/* Many of the operations are broken up into two pieces, a main part that    */
	/*   performs an approximate operation, and a "tail" that computes the       */
	/*   roundoff error of that operation.                                       */
	/*                                                                           */
	/* The operations Fast_Two_Sum(), Fast_Two_Diff(), Two_Sum(), Two_Diff(),    */
	/*   Split(), and Two_Product() are all implemented as described in the      */
	/*   reference.  Each of these macros requires certain variables to be       */
	/*   defined in the calling routine.  The variables `bvirt', `c', `abig',    */
	/*   `_i', `_j', `_k', `_l', `_m', and `_n' are declared `INEXACT' because   */
	/*   they store the result of an operation that may incur roundoff error.    */
	/*   The input parameter `x' (or the highest numbered `x_' parameter) must   */
	/*   also be declared `INEXACT'.                                             */

#define Fast_Two_Sum_Tail(a, b, x, y) \
	bvirt = x - a; \
	y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
	x = (REAL) (a + b); \
	Fast_Two_Sum_Tail(a, b, x, y)

#define Fast_Two_Diff_Tail(a, b, x, y) \
	bvirt = a - x; \
	y = bvirt - b

#define Fast_Two_Diff(a, b, x, y) \
	x = (REAL) (a - b); \
	Fast_Two_Diff_Tail(a, b, x, y)

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
	c = (REAL) (splitter * a); \
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

	/* Two_Product_Presplit() is Two_Product() where one of the inputs has       */
	/*   already been split.  A  voids redundant splitting.                        */

#define Two_Product_Presplit(a, b, bhi, blo, x, y) \
	x = (REAL) (a * b); \
	Split(a, ahi, alo); \
	err1 = x - (ahi * bhi); \
	err2 = err1 - (alo * bhi); \
	err3 = err2 - (ahi * blo); \
	y = (alo * blo) - err3

	/* Two_Product_2Presplit() is Two_Product() where both of the inputs have    */
	/*   already been split.  A  voids redundant splitting.                        */

#define Two_Product_2Presplit(a, ahi, alo, b, bhi, blo, x, y) \
	x = (REAL) (a * b); \
	err1 = x - (ahi * bhi); \
	err2 = err1 - (alo * bhi); \
	err3 = err2 - (ahi * blo); \
	y = (alo * blo) - err3

	/* Square() can be done more quickly than Two_Product().                     */

#define Square_Tail(a, x, y) \
	Split(a, ahi, alo); \
	err1 = x - (ahi * ahi); \
	err3 = err1 - ((ahi + ahi) * alo); \
	y = (alo * alo) - err3

#define Square(a, x, y) \
	x = (REAL) (a * a); \
	Square_Tail(a, x, y)

	/* Macros for summing expansions of various fixed lengths.  These are all    */
	/*   unrolled versions of Expansion_Sum().                                   */

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

#define Four_One_Sum(a3, a2, a1, a0, b, x4, x3, x2, x1, x0) \
	Two_One_Sum(a1, a0, b , _j, x1, x0); \
	Two_One_Sum(a3, a2, _j, x4, x3, x2)

#define Four_Two_Sum(a3, a2, a1, a0, b1, b0, x5, x4, x3, x2, x1, x0) \
	Four_One_Sum(a3, a2, a1, a0, b0, _k, _2, _1, _0, x0); \
	Four_One_Sum(_k, _2, _1, _0, b1, x5, x4, x3, x2, x1)

#define Four_Four_Sum(a3, a2, a1, a0, b4, b3, b1, b0, x7, x6, x5, x4, x3, x2, \
	x1, x0) \
	Four_Two_Sum(a3, a2, a1, a0, b1, b0, _l, _2, _1, _0, x1, x0); \
	Four_Two_Sum(_l, _2, _1, _0, b4, b3, x7, x6, x5, x4, x3, x2)

#define Eight_One_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b, x8, x7, x6, x5, x4, \
	x3, x2, x1, x0) \
	Four_One_Sum(a3, a2, a1, a0, b , _j, x3, x2, x1, x0); \
	Four_One_Sum(a7, a6, a5, a4, _j, x8, x7, x6, x5, x4)

#define Eight_Two_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b1, b0, x9, x8, x7, \
	x6, x5, x4, x3, x2, x1, x0) \
	Eight_One_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b0, _k, _6, _5, _4, _3, _2, \
	_1, _0, x0); \
	Eight_One_Sum(_k, _6, _5, _4, _3, _2, _1, _0, b1, x9, x8, x7, x6, x5, x4, \
	x3, x2, x1)

#define Eight_Four_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b4, b3, b1, b0, x11, \
	x10, x9, x8, x7, x6, x5, x4, x3, x2, x1, x0) \
	Eight_Two_Sum(a7, a6, a5, a4, a3, a2, a1, a0, b1, b0, _l, _6, _5, _4, _3, \
	_2, _1, _0, x1, x0); \
	Eight_Two_Sum(_l, _6, _5, _4, _3, _2, _1, _0, b4, b3, x11, x10, x9, x8, \
	x7, x6, x5, x4, x3, x2)

	/* Macros for multiplying expansions of various fixed lengths.               */

#define Two_One_Product(a1, a0, b, x3, x2, x1, x0) \
	Split(b, bhi, blo); \
	Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
	Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
	Two_Sum(_i, _0, _k, x1); \
	Fast_Two_Sum(_j, _k, x3, x2)

#define Four_One_Product(a3, a2, a1, a0, b, x7, x6, x5, x4, x3, x2, x1, x0) \
	Split(b, bhi, blo); \
	Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
	Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
	Two_Sum(_i, _0, _k, x1); \
	Fast_Two_Sum(_j, _k, _i, x2); \
	Two_Product_Presplit(a2, b, bhi, blo, _j, _0); \
	Two_Sum(_i, _0, _k, x3); \
	Fast_Two_Sum(_j, _k, _i, x4); \
	Two_Product_Presplit(a3, b, bhi, blo, _j, _0); \
	Two_Sum(_i, _0, _k, x5); \
	Fast_Two_Sum(_j, _k, x7, x6)

#define Two_Two_Product(a1, a0, b1, b0, x7, x6, x5, x4, x3, x2, x1, x0) \
	Split(a0, a0hi, a0lo); \
	Split(b0, bhi, blo); \
	Two_Product_2Presplit(a0, a0hi, a0lo, b0, bhi, blo, _i, x0); \
	Split(a1, a1hi, a1lo); \
	Two_Product_2Presplit(a1, a1hi, a1lo, b0, bhi, blo, _j, _0); \
	Two_Sum(_i, _0, _k, _1); \
	Fast_Two_Sum(_j, _k, _l, _2); \
	Split(b1, bhi, blo); \
	Two_Product_2Presplit(a0, a0hi, a0lo, b1, bhi, blo, _i, _0); \
	Two_Sum(_1, _0, _k, x1); \
	Two_Sum(_2, _k, _j, _1); \
	Two_Sum(_l, _j, _m, _2); \
	Two_Product_2Presplit(a1, a1hi, a1lo, b1, bhi, blo, _j, _0); \
	Two_Sum(_i, _0, _n, _0); \
	Two_Sum(_1, _0, _i, x2); \
	Two_Sum(_2, _i, _k, _1); \
	Two_Sum(_m, _k, _l, _2); \
	Two_Sum(_j, _n, _k, _0); \
	Two_Sum(_1, _0, _j, x3); \
	Two_Sum(_2, _j, _i, _1); \
	Two_Sum(_l, _i, _m, _2); \
	Two_Sum(_1, _k, _i, x4); \
	Two_Sum(_2, _i, _k, x5); \
	Two_Sum(_m, _k, x7, x6)

	/* An expansion of length two can be squared more quickly than finding the   */
	/*   product of two different expansions of length two, and the result is    */
	/*   guaranteed to have no more than six (rather than eight) components.     */

#define Two_Square(a1, a0, x5, x4, x3, x2, x1, x0) \
	Square(a0, _j, x0); \
	_0 = a0 + a0; \
	Two_Product(a1, _0, _k, _1); \
	Two_One_Sum(_k, _1, _j, _l, _2, x1); \
	Square(a1, _j, _1); \
	Two_Two_Sum(_j, _1, _l, _2, x5, x4, x3, x2)

	struct Predicates 
	{		
		REAL splitter;     /* = 2^ceiling(p / 2) + 1.  Used to split floats in half. */
		REAL epsilon;                /* = 2^(-p).  Used to estimate roundoff errors. */
		/* A set of coefficients used to calculate maximum roundoff errors.          */
		REAL resulterrbound;
		REAL ccwerrboundA, ccwerrboundB, ccwerrboundC;
		REAL o3derrboundA, o3derrboundB, o3derrboundC;
		REAL iccerrboundA, iccerrboundB, iccerrboundC;
		REAL isperrboundA, isperrboundB, isperrboundC;
		REAL a2trierrbound, a2triedgeerrbound, a2edgeErrorbound;
		REAL inballerrbound2;

		REAL *CPUtemp2a, *CPUtemp2b;								// 136130
		REAL *CPUtemp2;												// 272260
		int CPUtemp2alen, CPUtemp2blen, CPUtemp2len;			
		REAL *CPUrunningSum1, *CPUrunningSum2;							// 1633536
		REAL *CPUP1, *CPUP2;		

		/*****************************************************************************/
		/*                                                                           */
		/*  exactinit()   Initialize the variables used for exact arithmetic.        */
		/*                                                                           */
		/*  `epsilon' is the largest power of two such that 1.0 + epsilon = 1.0 in   */
		/*  floating-gpudtVertex arithmetic.  `epsilon' bounds the relative roundoff       */
		/*  error.  It is used for floating-gpudtVertex error analysis.                    */
		/*                                                                           */
		/*  `splitter' is used to split floating-gpudtVertex numbers into two half-        */
		/*  length significands for exact multiplication.                            */
		/*                                                                           */
		/*  I imagine that a highly optimizing compiler might be too smart for its   */
		/*  own good, and somehow cause this routine to fail, if it pretends that    */
		/*  floating-gpudtVertex arithmetic is too much like real arithmetic.              */
		/*                                                                           */
		/*  Don't change this routine unless you fully understand it.                */
		/*                                                                           */
		/*****************************************************************************/

		__device__ __host__ void exactinit()
		{
			REAL half;
			REAL check, lastcheck;
			int every_other;

			every_other = 1;
			half = 0.5;
			epsilon = 1.0;
			splitter = 1.0;
			check = 1.0;
			/* Repeatedly divide `epsilon' by two until it is too small to add to    */
			/*   one without causing roundoff.  (Also check if the sum is equal to   */
			/*   the previous sum, for machines that round up instead of using exact */
			/*   rounding.  Not that this library will work on such machines anyway. */
			do {
				lastcheck = check;
				epsilon *= half;
				if (every_other) {
					splitter *= 2.0;
				}
				every_other = !every_other;
				check = 1.0 + epsilon;
			} while ((check != 1.0) && (check != lastcheck));
			splitter += 1.0;

			/* Error bounds for orientation and incircle tests. */
			resulterrbound = (3.0 + 8.0 * epsilon) * epsilon;
			ccwerrboundA = (3.0 + 16.0 * epsilon) * epsilon;
			ccwerrboundB = (2.0 + 12.0 * epsilon) * epsilon;
			ccwerrboundC = (9.0 + 64.0 * epsilon) * epsilon * epsilon;
			o3derrboundA = (7.0 + 56.0 * epsilon) * epsilon;
			o3derrboundB = (3.0 + 28.0 * epsilon) * epsilon;
			o3derrboundC = (26.0 + 288.0 * epsilon) * epsilon * epsilon;
			iccerrboundA = (10.0 + 96.0 * epsilon) * epsilon;
			iccerrboundB = (4.0 + 48.0 * epsilon) * epsilon;
			iccerrboundC = (44.0 + 576.0 * epsilon) * epsilon * epsilon;
			isperrboundA = (16.0 + 224.0 * epsilon) * epsilon;
			isperrboundB = (5.0 + 72.0 * epsilon) * epsilon;
			isperrboundC = (71.0 + 1408.0 * epsilon) * epsilon * epsilon;

			/* Error bounds for alpha shapes */
			a2trierrbound = ( 173.0 + 320.0 * epsilon) * epsilon;
			a2triedgeerrbound = ( 144.0 + 256.0 * epsilon) * epsilon;
			a2edgeErrorbound = ( 100.0 + 640.0 * epsilon) * epsilon;	

			inballerrbound2 = ( 160.0 + 32.0 * epsilon) * epsilon;	
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  grow_expansion()   Add a scalar to an expansion.                         */
		/*                                                                           */
		/*  Sets h = e + b.  See the long version of my paper for details.           */
		/*                                                                           */
		/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
		/*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    */
		/*  properties as well.  (That is, if e has one of these properties, so      */
		/*  will h.)                                                                 */
		/*                                                                           */
		/*****************************************************************************/

		__device__ __host__ int grow_expansion(int elen,REAL *e,REAL b,REAL *h)                /* e and h can be the same. */
		{
			REAL Q;
			INEXACT REAL Qnew;
			int eindex;
			REAL enow;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;

			Q = b;
			for (eindex = 0; eindex < elen; eindex++) {
				enow = e[eindex];
				Two_Sum(Q, enow, Qnew, h[eindex]);
				Q = Qnew;
			}
			h[eindex] = Q;
			return eindex + 1;
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  grow_expansion_zeroelim()   Add a scalar to an expansion, eliminating    */
		/*                              zero components from the output expansion.   */
		/*                                                                           */
		/*  Sets h = e + b.  See the long version of my paper for details.           */
		/*                                                                           */
		/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
		/*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    */
		/*  properties as well.  (That is, if e has one of these properties, so      */
		/*  will h.)                                                                 */
		/*                                                                           */
		/*****************************************************************************/

		__device__ __host__ int grow_expansion_zeroelim(int elen,REAL *e,REAL b,REAL *h)       /* e and h can be the same. */
		{
			REAL Q, hh;
			INEXACT REAL Qnew;
			int eindex, hindex;
			REAL enow;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;

			hindex = 0;
			Q = b;
			for (eindex = 0; eindex < elen; eindex++) {
				enow = e[eindex];
				Two_Sum(Q, enow, Qnew, hh);
				Q = Qnew;
				if (hh != 0.0) {
					h[hindex++] = hh;
				}
			}
			if ((Q != 0.0) || (hindex == 0)) {
				h[hindex++] = Q;
			}
			return hindex;
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  expansion_sum()   Sum two expansions.                                    */
		/*                                                                           */
		/*  Sets h = e + f.  See the long version of my paper for details.           */
		/*                                                                           */
		/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
		/*  with IEEE 754), maintains the nonadjacent property as well.  (That is,   */
		/*  if e has one of these properties, so will h.)  Does NOT maintain the     */
		/*  strongly nonoverlapping property.                                        */
		/*                                                                           */
		/*****************************************************************************/

		__device__ __host__ int expansion_sum(int elen,REAL *e,int flen,REAL *f,REAL *h)
			/* e and h can be the same, but f and h cannot. */
		{
			REAL Q;
			INEXACT REAL Qnew;
			int findex, hindex, hlast;
			REAL hnow;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;

			Q = f[0];
			for (hindex = 0; hindex < elen; hindex++) {
				hnow = e[hindex];
				Two_Sum(Q, hnow, Qnew, h[hindex]);
				Q = Qnew;
			}
			h[hindex] = Q;
			hlast = hindex;
			for (findex = 1; findex < flen; findex++) {
				Q = f[findex];
				for (hindex = findex; hindex <= hlast; hindex++) {
					hnow = h[hindex];
					Two_Sum(Q, hnow, Qnew, h[hindex]);
					Q = Qnew;
				}
				h[++hlast] = Q;
			}
			return hlast + 1;
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  expansion_sum_zeroelim1()   Sum two expansions, eliminating zero         */
		/*                              components from the output expansion.        */
		/*                                                                           */
		/*  Sets h = e + f.  See the long version of my paper for details.           */
		/*                                                                           */
		/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
		/*  with IEEE 754), maintains the nonadjacent property as well.  (That is,   */
		/*  if e has one of these properties, so will h.)  Does NOT maintain the     */
		/*  strongly nonoverlapping property.                                        */
		/*                                                                           */
		/*****************************************************************************/

		__device__ __host__ int expansion_sum_zeroelim1(int elen,REAL *e,int flen,REAL *f,REAL *h)
			/* e and h can be the same, but f and h cannot. */
		{
			REAL Q;
			INEXACT REAL Qnew;
			int index, findex, hindex, hlast;
			REAL hnow;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;

			Q = f[0];
			for (hindex = 0; hindex < elen; hindex++) {
				hnow = e[hindex];
				Two_Sum(Q, hnow, Qnew, h[hindex]);
				Q = Qnew;
			}
			h[hindex] = Q;
			hlast = hindex;
			for (findex = 1; findex < flen; findex++) {
				Q = f[findex];
				for (hindex = findex; hindex <= hlast; hindex++) {
					hnow = h[hindex];
					Two_Sum(Q, hnow, Qnew, h[hindex]);
					Q = Qnew;
				}
				h[++hlast] = Q;
			}
			hindex = -1;
			for (index = 0; index <= hlast; index++) {
				hnow = h[index];
				if (hnow != 0.0) {
					h[++hindex] = hnow;
				}
			}
			if (hindex == -1) {
				return 1;
			} else {
				return hindex + 1;
			}
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  expansion_sum_zeroelim2()   Sum two expansions, eliminating zero         */
		/*                              components from the output expansion.        */
		/*                                                                           */
		/*  Sets h = e + f.  See the long version of my paper for details.           */
		/*                                                                           */
		/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
		/*  with IEEE 754), maintains the nonadjacent property as well.  (That is,   */
		/*  if e has one of these properties, so will h.)  Does NOT maintain the     */
		/*  strongly nonoverlapping property.                                        */
		/*                                                                           */
		/*****************************************************************************/

		__device__ __host__ int expansion_sum_zeroelim2(int elen,REAL *e,int flen,REAL *f,REAL *h)
			/* e and h can be the same, but f and h cannot. */
		{
			REAL Q, hh;
			INEXACT REAL Qnew;
			int eindex, findex, hindex, hlast;
			REAL enow;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;

			hindex = 0;
			Q = f[0];
			for (eindex = 0; eindex < elen; eindex++) {
				enow = e[eindex];
				Two_Sum(Q, enow, Qnew, hh);
				Q = Qnew;
				if (hh != 0.0) {
					h[hindex++] = hh;
				}
			}
			h[hindex] = Q;
			hlast = hindex;
			for (findex = 1; findex < flen; findex++) {
				hindex = 0;
				Q = f[findex];
				for (eindex = 0; eindex <= hlast; eindex++) {
					enow = h[eindex];
					Two_Sum(Q, enow, Qnew, hh);
					Q = Qnew;
					if (hh != 0) {
						h[hindex++] = hh;
					}
				}
				h[hindex] = Q;
				hlast = hindex;
			}
			return hlast + 1;
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  fast_expansion_sum()   Sum two expansions.                               */
		/*                                                                           */
		/*  Sets h = e + f.  See the long version of my paper for details.           */
		/*                                                                           */
		/*  If round-to-even is used (as with IEEE 754), maintains the strongly      */
		/*  nonoverlapping property.  (That is, if e is strongly nonoverlapping, h   */
		/*  will be also.)  Does NOT maintain the nonoverlapping or nonadjacent      */
		/*  properties.                                                              */
		/*                                                                           */
		/*****************************************************************************/		

		__device__ __host__ int fast_expansion_sum(int elen,REAL *e,int flen,REAL *f,REAL *h)           /* h cannot be e or f. */
		{
			REAL Q;
			INEXACT REAL Qnew;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;
			int eindex, findex, hindex;
			REAL enow, fnow;

			enow = e[0];
			fnow = f[0];
			eindex = findex = 0;
			if ((fnow > enow) == (fnow > -enow)) {
				Q = enow;
				enow = e[++eindex];
			} else {
				Q = fnow;
				fnow = f[++findex];
			}
			hindex = 0;
			if ((eindex < elen) && (findex < flen)) {
				if ((fnow > enow) == (fnow > -enow)) {
					Fast_Two_Sum(enow, Q, Qnew, h[0]);
					enow = e[++eindex];
				} else {
					Fast_Two_Sum(fnow, Q, Qnew, h[0]);
					fnow = f[++findex];
				}
				Q = Qnew;
				hindex = 1;
				while ((eindex < elen) && (findex < flen)) {
					if ((fnow > enow) == (fnow > -enow)) {
						Two_Sum(Q, enow, Qnew, h[hindex]);
						enow = e[++eindex];
					} else {
						Two_Sum(Q, fnow, Qnew, h[hindex]);
						fnow = f[++findex];
					}
					Q = Qnew;
					hindex++;
				}
			}
			while (eindex < elen) {
				Two_Sum(Q, enow, Qnew, h[hindex]);
				enow = e[++eindex];
				Q = Qnew;
				hindex++;
			}
			while (findex < flen) {
				Two_Sum(Q, fnow, Qnew, h[hindex]);
				fnow = f[++findex];
				Q = Qnew;
				hindex++;
			}
			h[hindex] = Q;
			return hindex + 1;
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  fast_expansion_sum_zeroelim()   Sum two expansions, eliminating zero     */
		/*                                  components from the output expansion.    */
		/*                                                                           */
		/*  Sets h = e + f.  See the long version of my paper for details.           */
		/*                                                                           */
		/*  If round-to-even is used (as with IEEE 754), maintains the strongly      */
		/*  nonoverlapping property.  (That is, if e is strongly nonoverlapping, h   */
		/*  will be also.)  Does NOT maintain the nonoverlapping or nonadjacent      */
		/*  properties.                                                              */
		/*                                                                           */
		/*****************************************************************************/

		int cpu_fast_expansion_sum_zeroelim(int elen,REAL *e,int flen,REAL *f,REAL *h)  /* h cannot be e or f. */
		{
			REAL Q;
			INEXACT REAL Qnew;
			INEXACT REAL hh;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;
			int eindex, findex, hindex;
			REAL enow, fnow;

			enow = e[0];
			fnow = f[0];
			eindex = findex = 0;
			if ((fnow > enow) == (fnow > -enow)) {
				Q = enow;
				enow = e[++eindex];
			} else {
				Q = fnow;
				fnow = f[++findex];
			}
			hindex = 0;
			if ((eindex < elen) && (findex < flen)) {
				if ((fnow > enow) == (fnow > -enow)) {
					Fast_Two_Sum(enow, Q, Qnew, hh);
					enow = e[++eindex];
				} else {
					Fast_Two_Sum(fnow, Q, Qnew, hh);
					fnow = f[++findex];
				}
				Q = Qnew;
				if (hh != 0.0) {
					h[hindex++] = hh;					
				}				
				while ((eindex < elen) && (findex < flen)) {
					if ((fnow > enow) == (fnow > -enow)) {
						Two_Sum(Q, enow, Qnew, hh);
						enow = e[++eindex];
					} else {
						Two_Sum(Q, fnow, Qnew, hh);
						fnow = f[++findex];
					}
					Q = Qnew;
					if (hh != 0.0) {
						h[hindex++] = hh;						
					}
				}
			}
			while (eindex < elen) {
				Two_Sum(Q, enow, Qnew, hh);
				enow = e[++eindex];
				Q = Qnew;
				if (hh != 0.0) {
					h[hindex++] = hh;					
				}
			}
			while (findex < flen) {
				Two_Sum(Q, fnow, Qnew, hh);
				fnow = f[++findex];
				Q = Qnew;
				if (hh != 0.0) {
					h[hindex++] = hh;					
				}
			}
			if ((Q != 0.0) || (hindex == 0)) {
				h[hindex++] = Q;				
			}
			return hindex;
		}

		__device__ __host__ int fast_expansion_sum_zeroelim(int elen,REAL *e,int flen,REAL *f,REAL *h)  /* h cannot be e or f. */
		{
			REAL Q;
			INEXACT REAL Qnew;
			INEXACT REAL hh;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;
			int eindex, findex, hindex;
			REAL enow, fnow;

			enow = e[0];
			fnow = f[0];
			eindex = findex = 0;
			if ((fnow > enow) == (fnow > -enow)) {
				Q = enow;
				enow = e[++eindex];
			} else {
				Q = fnow;
				fnow = f[++findex];
			}
			hindex = 0;
			if ((eindex < elen) && (findex < flen)) {
				if ((fnow > enow) == (fnow > -enow)) {
					Fast_Two_Sum(enow, Q, Qnew, hh);
					enow = e[++eindex];
				} else {
					Fast_Two_Sum(fnow, Q, Qnew, hh);
					fnow = f[++findex];
				}
				Q = Qnew;
				if (hh != 0.0) {
					h[hindex++] = hh;
					if(hindex == MAX_N-1)	return -2;				//added
				}
				if(hindex == MAX_N-1)	return -2;					//added
				while ((eindex < elen) && (findex < flen)) {
					if ((fnow > enow) == (fnow > -enow)) {
						Two_Sum(Q, enow, Qnew, hh);
						enow = e[++eindex];
					} else {
						Two_Sum(Q, fnow, Qnew, hh);
						fnow = f[++findex];
					}
					Q = Qnew;
					if (hh != 0.0) {
						h[hindex++] = hh;
						if(hindex == MAX_N-1)	return -2;			//added
					}
				}
			}
			while (eindex < elen) {
				Two_Sum(Q, enow, Qnew, hh);
				enow = e[++eindex];
				Q = Qnew;
				if (hh != 0.0) {
					h[hindex++] = hh;
					if(hindex == MAX_N-1)	return -2;				//added
				}
			}
			while (findex < flen) {
				Two_Sum(Q, fnow, Qnew, hh);
				fnow = f[++findex];
				Q = Qnew;
				if (hh != 0.0) {
					h[hindex++] = hh;
					if(hindex == MAX_N-1)	return -2;				//added
				}
			}
			if ((Q != 0.0) || (hindex == 0)) {
				h[hindex++] = Q;
				if(hindex == MAX_N-1)	return -2;					//added
			}
			return hindex;
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  linear_expansion_sum()   Sum two expansions.                             */
		/*                                                                           */
		/*  Sets h = e + f.  See either version of my paper for details.             */
		/*                                                                           */
		/*  Maintains the nonoverlapping property.  (That is, if e is                */
		/*  nonoverlapping, h will be also.)                                         */
		/*                                                                           */
		/*****************************************************************************/

		__device__ __host__ int linear_expansion_sum(int elen,REAL *e,int flen,REAL *f,REAL *h)         /* h cannot be e or f. */
		{
			REAL Q, q;
			INEXACT REAL Qnew;
			INEXACT REAL R;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;
			int eindex, findex, hindex;
			REAL enow, fnow;
			REAL g0;

			enow = e[0];
			fnow = f[0];
			eindex = findex = 0;
			if ((fnow > enow) == (fnow > -enow)) {
				g0 = enow;
				enow = e[++eindex];
			} else {
				g0 = fnow;
				fnow = f[++findex];
			}
			if ((eindex < elen) && ((findex >= flen)
				|| ((fnow > enow) == (fnow > -enow)))) {
					Fast_Two_Sum(enow, g0, Qnew, q);
					enow = e[++eindex];
			} else {
				Fast_Two_Sum(fnow, g0, Qnew, q);
				fnow = f[++findex];
			}
			Q = Qnew;
			for (hindex = 0; hindex < elen + flen - 2; hindex++) {
				if ((eindex < elen) && ((findex >= flen)
					|| ((fnow > enow) == (fnow > -enow)))) {
						Fast_Two_Sum(enow, q, R, h[hindex]);
						enow = e[++eindex];
				} else {
					Fast_Two_Sum(fnow, q, R, h[hindex]);
					fnow = f[++findex];
				}
				Two_Sum(Q, R, Qnew, q);
				Q = Qnew;
			}
			h[hindex] = q;
			h[hindex + 1] = Q;
			return hindex + 2;
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  linear_expansion_sum_zeroelim()   Sum two expansions, eliminating zero   */
		/*                                    components from the output expansion.  */
		/*                                                                           */
		/*  Sets h = e + f.  See either version of my paper for details.             */
		/*                                                                           */
		/*  Maintains the nonoverlapping property.  (That is, if e is                */
		/*  nonoverlapping, h will be also.)                                         */
		/*                                                                           */
		/*****************************************************************************/

		__device__ __host__ int linear_expansion_sum_zeroelim(int elen,REAL *e,int flen,REAL *f,REAL *h)/* h cannot be e or f. */
		{
			REAL Q, q, hh;
			INEXACT REAL Qnew;
			INEXACT REAL R;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;
			int eindex, findex, hindex;
			int count;
			REAL enow, fnow;
			REAL g0;

			enow = e[0];
			fnow = f[0];
			eindex = findex = 0;
			hindex = 0;
			if ((fnow > enow) == (fnow > -enow)) {
				g0 = enow;
				enow = e[++eindex];
			} else {
				g0 = fnow;
				fnow = f[++findex];
			}
			if ((eindex < elen) && ((findex >= flen)
				|| ((fnow > enow) == (fnow > -enow)))) {
					Fast_Two_Sum(enow, g0, Qnew, q);
					enow = e[++eindex];
			} else {
				Fast_Two_Sum(fnow, g0, Qnew, q);
				fnow = f[++findex];
			}
			Q = Qnew;
			for (count = 2; count < elen + flen; count++) {
				if ((eindex < elen) && ((findex >= flen)
					|| ((fnow > enow) == (fnow > -enow)))) {
						Fast_Two_Sum(enow, q, R, hh);
						enow = e[++eindex];
				} else {
					Fast_Two_Sum(fnow, q, R, hh);
					fnow = f[++findex];
				}
				Two_Sum(Q, R, Qnew, q);
				Q = Qnew;
				if (hh != 0) {
					h[hindex++] = hh;
				}
			}
			if (q != 0) {
				h[hindex++] = q;
			}
			if ((Q != 0.0) || (hindex == 0)) {
				h[hindex++] = Q;
			}
			return hindex;
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  scale_expansion()   Multiply an expansion by a scalar.                   */
		/*                                                                           */
		/*  Sets h = be.  See either version of my paper for details.                */
		/*                                                                           */
		/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
		/*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    */
		/*  properties as well.  (That is, if e has one of these properties, so      */
		/*  will h.)                                                                 */
		/*                                                                           */
		/*****************************************************************************/

		__device__ __host__ int scale_expansion(int elen,REAL *e,REAL b,REAL *h)            /* e and h cannot be the same. */
		{
			INEXACT REAL Q;
			INEXACT REAL sum;
			INEXACT REAL product1;
			REAL product0;
			int eindex, hindex;
			REAL enow;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;
			INEXACT REAL c;
			INEXACT REAL abig;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;

			Split(b, bhi, blo);
			Two_Product_Presplit(e[0], b, bhi, blo, Q, h[0]);
			hindex = 1;
			for (eindex = 1; eindex < elen; eindex++) {
				enow = e[eindex];
				Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
				Two_Sum(Q, product0, sum, h[hindex]);
				hindex++;
				Two_Sum(product1, sum, Q, h[hindex]);
				hindex++;
			}
			h[hindex] = Q;
			return elen + elen;
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  scale_expansion_zeroelim()   Multiply an expansion by a scalar,          */
		/*                               eliminating zero components from the        */
		/*                               output expansion.                           */
		/*                                                                           */
		/*  Sets h = be.  See either version of my paper for details.                */
		/*                                                                           */
		/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
		/*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    */
		/*  properties as well.  (That is, if e has one of these properties, so      */
		/*  will h.)                                                                 */
		/*                                                                           */
		/*****************************************************************************/

		int cpu_scale_expansion_zeroelim(int elen,REAL *e,REAL b,REAL *h)   /* e and h cannot be the same. */
		{
			INEXACT REAL Q, sum;
			REAL hh;
			INEXACT REAL product1;
			REAL product0;
			int eindex, hindex;
			REAL enow;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;
			INEXACT REAL c;
			INEXACT REAL abig;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;

			Split(b, bhi, blo);
			Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
			hindex = 0;
			if (hh != 0) {
				h[hindex++] = hh;				
			}
			for (eindex = 1; eindex < elen; eindex++) {
				enow = e[eindex];
				Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
				Two_Sum(Q, product0, sum, hh);
				if (hh != 0) {
					h[hindex++] = hh;					
				}
				Fast_Two_Sum(product1, sum, Q, hh);
				if (hh != 0) {
					h[hindex++] = hh;					
				}
			}
			if ((Q != 0.0) || (hindex == 0)) {
				h[hindex++] = Q;				
			}
			return hindex;
		}

		__device__ __host__ int scale_expansion_zeroelim(int elen,REAL *e,REAL b,REAL *h)   /* e and h cannot be the same. */
		{
			INEXACT REAL Q, sum;
			REAL hh;
			INEXACT REAL product1;
			REAL product0;
			int eindex, hindex;
			REAL enow;
			INEXACT REAL bvirt;
			REAL avirt, bround, around;
			INEXACT REAL c;
			INEXACT REAL abig;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;

			Split(b, bhi, blo);
			Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
			hindex = 0;
			if (hh != 0) {
				h[hindex++] = hh;
				if(hindex == MAX_N-1)	return -2;					//added
			}
			for (eindex = 1; eindex < elen; eindex++) {
				enow = e[eindex];
				Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
				Two_Sum(Q, product0, sum, hh);
				if (hh != 0) {
					h[hindex++] = hh;
					if(hindex == MAX_N-1)	return -2;				//added
				}
				Fast_Two_Sum(product1, sum, Q, hh);
				if (hh != 0) {
					h[hindex++] = hh;
					if(hindex == MAX_N-1)	return -2;				//added
				}
			}
			if ((Q != 0.0) || (hindex == 0)) {
				h[hindex++] = Q;
				if(hindex == MAX_N-1)	return -2;					//added
			}
			return hindex;
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  compress()   Compress an expansion.                                      */
		/*                                                                           */
		/*  See the long version of my paper for details.                            */
		/*                                                                           */
		/*  Maintains the nonoverlapping property.  If round-to-even is used (as     */
		/*  with IEEE 754), then any nonoverlapping expansion is converted to a      */
		/*  nonadjacent expansion.                                                   */
		/*                                                                           */
		/*****************************************************************************/

		__device__ __host__ int compress(int elen,REAL *e,REAL *h)                         /* e and h may be the same. */
		{
			REAL Q, q;
			INEXACT REAL Qnew;
			int eindex, hindex;
			INEXACT REAL bvirt;
			REAL enow, hnow;
			int top, bottom;

			bottom = elen - 1;
			Q = e[bottom];
			for (eindex = elen - 2; eindex >= 0; eindex--) {
				enow = e[eindex];
				Fast_Two_Sum(Q, enow, Qnew, q);
				if (q != 0) {
					h[bottom--] = Qnew;
					Q = q;
				} else {
					Q = Qnew;
				}
			}
			top = 0;
			for (hindex = bottom + 1; hindex < elen; hindex++) {
				hnow = h[hindex];
				Fast_Two_Sum(hnow, Q, Qnew, q);
				if (q != 0) {
					h[top++] = q;
				}
				Q = Qnew;
			}
			h[top] = Q;
			return top + 1;
		}

		/*****************************************************************************/
		/*                                                                           */
		/*  estimate()   Produce a one-word estimate of an expansion's value.        */
		/*                                                                           */
		/*  See either version of my paper for details.                              */
		/*                                                                           */
		/*****************************************************************************/

		__device__ __host__ REAL estimate(int elen,REAL *e)
		{
			REAL Q;
			int eindex;

			Q = e[0];
			for (eindex = 1; eindex < elen; eindex++) {
				Q += e[eindex];
			}
			return Q;
		}		

		/*****************************************************************************/
		/*                                                                           */
		/*  Sub-Routines for 2D Alpha Shapes								         */				
		/*                                                                           */
		/*****************************************************************************/		

		__device__ __host__ double determinant_3(gpudtVertex& pa, gpudtVertex& pb, gpudtVertex& pc)
		{
			double term1 = pa.x * (pb.y - pc.y);
			double term2 = pa.y * (pb.x - pc.x);
			double term3 = pb.x * pc.y - pb.y * pc.x;

			return (term1 - term2 + term3);
		}

		__device__ __host__ void shift(REAL &x1, REAL &y1, REAL &x2, REAL &y2, REAL &x3, REAL &y3)
		{			
			x2 = x2 - x1;
			y2 = y2 - y1;
			x3 = x3 - x1;
			y3 = y3 - y1;
			x1 = 0;
			y1 = 0;
		}

		__device__ __host__ double birth_time(gpudtVertex& pa, gpudtVertex& pb, gpudtVertex& pc)
		{
			REAL ax, ay, bx, by, cx, cy;
			REAL a, b, c;
			REAL ar1;
			REAL alpha, den;			

			// Points
			REAL x1,x2,x3,y1,y2,y3;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;
			x3 = pc.x; y3 = pc.y;			
			//shift(x1,y1,x2,y2,x3,y3);

			ax = x1 - x2;
			ay = y1 - y2;
			
			bx = x2 - x3;
			by = y2 - y3;

			cx = x3 - x1;
			cy = y3 - y1;

			ax = ax * ax;
			ay = ay * ay;
			bx = bx * bx;
			by = by * by;
			cx = cx * cx;
			cy = cy * cy;

			a = ax + ay;
			b = bx + by;
			c = cx + cy;

			ar1 = x2 * y3 - x3 * y2 - x1 * y3 + x3 * y1 + x1 * y2 - x2 * y1;			
			den = 4 * ar1 * ar1;
			alpha = (a * b * c) / den;
			
			return alpha;
		}

		__device__ __host__ double birth_time(gpudtVertex& pa, gpudtVertex& pb)
		{
			REAL alpha_f = 0;
			
			REAL x1,x2,y1,y2;
			REAL ax, ay;
			REAL a;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;

			ax = x1 - x2;
			ay = y1 - y2;
			ax = ax * ax;
			ay = ay * ay;
			a = ax + ay;
									
			alpha_f = a / 4;

			return alpha_f;
		}		

		__device__ __host__ bool isAttached(gpudtVertex& i, gpudtVertex& j, gpudtVertex& k, bool print)
		{
			REAL d1, d2, diff;

			d1 = ((i.x - j.x)*(i.x - j.x) + (i.y - j.y)*(i.y - j.y));
			d2 = (i.x + j.x - 2*k.x)*(i.x + j.x - 2*k.x) + (i.y + j.y - 2*k.y)*(i.y + j.y - 2*k.y);
			diff = d1 - d2;			

			REAL permanent = d1 + ( (Absolute(i.x + j.x) + 2*Absolute(k.x)) * (Absolute(i.x + j.x) + 2*Absolute(k.x)) + 
									(Absolute(i.y + j.y) + 2*Absolute(k.y)) * (Absolute(i.y + j.y) + 2*Absolute(k.y)) );
			REAL errbound = inballerrbound2 * permanent;

			if ((diff > errbound) || (-diff > errbound)) {
				return (diff > 0);
			}

			/*if(print)
				printf("\nSlow computation\n");*/
			return isAttachedSlow(i,j,k);
		}

		__device__ __host__ bool isAttachedSlow(gpudtVertex& i, gpudtVertex& j, gpudtVertex& k)
		{
			// For comments, gpudtVertex pa will be referred as (x1,y1), pb as (x2,y2) .. and pf as (x6,y6)			
			
			// Variables needed for Sum, Multiplication
			INEXACT REAL bvirt;
			REAL avirt, bround, around;			
			INEXACT REAL abig;
			REAL c;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;
			INEXACT REAL _i, _j, _k, _l;
			REAL _0, _1, _2;			

			// Variable to locally multiply and keep a running sum upto now
			REAL x1,x2,x3,y1,y2,y3;			
			x1 = i.x; y1 = i.y;
			x2 = j.x; y2 = j.y;
			x3 = k.x; y3 = k.y;			

			REAL ax1, ay1;
			REAL ax0, ay0;
			REAL ax_sq[6], ay_sq[6], a_sq[12];
			
			REAL temp2a[36], temp2b[36], temp2c[36];			
			REAL temp2[36];										
			int temp2alen, temp2clen, temp2len;			
			REAL runningSum1[36], runningSum2[36];			
			
			int rSumlen1 = 0, rSumlen2 = 0;
			runningSum1[0] = 0;

			// abracadabra
			// Compute ( (xi-xj)^2 + (yi-yj)^2 ) = p1			
			Two_Diff(x1,x2,ax1,ax0);																// Compute (x1-x2)			
			Two_Diff(y1,y2,ay1,ay0);																// Compute (y1-y2)		
			Two_Square(ax1, ax0, ax_sq[5], ax_sq[4], ax_sq[3], ax_sq[2], ax_sq[1], ax_sq[0]);		// Compute (x1-x2)^2			
			Two_Square(ay1, ay0, ay_sq[5], ay_sq[4], ay_sq[3], ay_sq[2], ay_sq[1], ay_sq[0]);		// Compute (y1-y2)^2
			int a_sq_length = fast_expansion_sum_zeroelim(6, ax_sq, 6, ay_sq, a_sq);				// Compute ((x1-x2)^2 + (y1-y2)^2)	[max 12 floats]			

			// Compute 4*( (xi + xj - 2*xk)^2 + (yi + yj - 2*yk)^2 )	= p2			
			Two_Sum(x1, x2, temp2a[1], temp2a[0]);
			Two_One_Diff(temp2a[1], temp2a[0], 2*x3, temp2b[2], temp2b[1], temp2b[0]);
			temp2alen = scale_expansion_zeroelim(3, temp2b, x1, temp2a);
			temp2clen = scale_expansion_zeroelim(3, temp2b, x2, temp2c);
			rSumlen1 = fast_expansion_sum_zeroelim(temp2alen, temp2a, temp2clen, temp2c, runningSum1);
			temp2alen = scale_expansion_zeroelim(3, temp2b, -2*x3, temp2a);
			temp2len = fast_expansion_sum_zeroelim(rSumlen1, runningSum1, temp2alen, temp2a, temp2);

			Two_Sum(y1, y2, temp2a[1], temp2a[0]);
			Two_One_Diff(temp2a[1], temp2a[0], 2*y3, temp2b[2], temp2b[1], temp2b[0]);
			temp2alen = scale_expansion_zeroelim(3, temp2b, y1, temp2a);
			temp2clen = scale_expansion_zeroelim(3, temp2b, y2, temp2c);
			rSumlen1 = fast_expansion_sum_zeroelim(temp2alen, temp2a, temp2clen, temp2c, runningSum1);
			temp2alen = scale_expansion_zeroelim(3, temp2b, -2*y3, temp2a);
			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1, temp2alen, temp2a, runningSum2);

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2, temp2len, temp2, runningSum1);
			
			// Compute ( (xi-xj)^2 + (yi-yj)^2 ) - ( (xi + xj - 2*xk)^2 + (yi + yj - 2*yk)^2 )
			for(int i=0; i<rSumlen1; i++)	runningSum1[i] = -runningSum1[i];
			rSumlen2 = fast_expansion_sum_zeroelim(a_sq_length, a_sq, rSumlen1, runningSum1, runningSum2);

			return (runningSum2[rSumlen2-1] >= 0);
		}
		
		__device__ __host__ int convert(double diff)
		{
			if(diff > 0)
				return 1;
			else if(diff < 0)
				return -1;
			return 0;
		}				

		__device__ __host__ REAL orient2dslow(gpudtVertex& pa,gpudtVertex& pb,gpudtVertex& pc)
		{
			INEXACT REAL acx, acy, bcx, bcy;
			REAL acxtail, acytail;
			REAL bcxtail, bcytail;
			REAL negate, negatetail;
			REAL axby[8], bxay[8];
			INEXACT REAL axby7, bxay7;
			REAL deter[16];
			int deterlen;

			INEXACT REAL bvirt;
			REAL avirt, bround, around;
			INEXACT REAL c;
			INEXACT REAL abig;
			REAL a0hi, a0lo, a1hi, a1lo, bhi, blo;
			REAL err1, err2, err3;
			INEXACT REAL _i, _j, _k, _l, _m, _n;
			REAL _0, _1, _2;

			Two_Diff(pa.x, pc.x, acx, acxtail);
			Two_Diff(pa.y, pc.y, acy, acytail);
			Two_Diff(pb.x, pc.x, bcx, bcxtail);
			Two_Diff(pb.y, pc.y, bcy, bcytail);

			Two_Two_Product(acx, acxtail, bcy, bcytail,
				axby7, axby[6], axby[5], axby[4],
				axby[3], axby[2], axby[1], axby[0]);
			axby[7] = axby7;
			negate = -acy;
			negatetail = -acytail;
			Two_Two_Product(bcx, bcxtail, negate, negatetail,
				bxay7, bxay[6], bxay[5], bxay[4],
				bxay[3], bxay[2], bxay[1], bxay[0]);
			bxay[7] = bxay7;

			deterlen = fast_expansion_sum_zeroelim(8, axby, 8, bxay, deter);

			return deter[deterlen - 1];
		}

		__device__ __host__ REAL orient2dexact(REAL *pa,REAL *pb,REAL *pc)
		{
			INEXACT REAL axby1, axcy1, bxcy1, bxay1, cxay1, cxby1;
			REAL axby0, axcy0, bxcy0, bxay0, cxay0, cxby0;
			REAL aterms[4], bterms[4], cterms[4];
			INEXACT REAL aterms3, bterms3, cterms3;
			REAL v[8], w[12];
			int vlength, wlength;

			INEXACT REAL bvirt;
			REAL avirt, bround, around;
			INEXACT REAL c;
			INEXACT REAL abig;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;
			INEXACT REAL _i, _j;
			REAL _0;

			Two_Product(pa[0], pb[1], axby1, axby0);
			Two_Product(pa[0], pc[1], axcy1, axcy0);
			Two_Two_Diff(axby1, axby0, axcy1, axcy0,
				aterms3, aterms[2], aterms[1], aterms[0]);
			aterms[3] = aterms3;

			Two_Product(pb[0], pc[1], bxcy1, bxcy0);
			Two_Product(pb[0], pa[1], bxay1, bxay0);
			Two_Two_Diff(bxcy1, bxcy0, bxay1, bxay0,
				bterms3, bterms[2], bterms[1], bterms[0]);
			bterms[3] = bterms3;

			Two_Product(pc[0], pa[1], cxay1, cxay0);
			Two_Product(pc[0], pb[1], cxby1, cxby0);
			Two_Two_Diff(cxay1, cxay0, cxby1, cxby0,
				cterms3, cterms[2], cterms[1], cterms[0]);
			cterms[3] = cterms3;

			vlength = fast_expansion_sum_zeroelim(4, aterms, 4, bterms, v);
			wlength = fast_expansion_sum_zeroelim(vlength, v, 4, cterms, w);

			return w[wlength - 1];
		}

		// Triangle1 > Triangle2
		__device__ __host__ int compareFast(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pc, gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf, int print)
		{
			REAL ax, ay, bx, by, cx, cy;
			REAL a, b, c;
			REAL ar1, ar2;
			REAL alpha1, alpha2, diff;
			REAL permanent1, permanent2, permanent;
			REAL errbound;

			// Points
			REAL x1,x2,x3,y1,y2,y3;
			REAL x4,x5,x6,y4,y5,y6;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;
			x3 = pc.x; y3 = pc.y;
			x4 = pd.x; y4 = pd.y;
			x5 = pe.x; y5 = pe.y;
			x6 = pf.x; y6 = pf.y;

			ax = x1 - x2;
			ay = y1 - y2;
			
			bx = x2 - x3;
			by = y2 - y3;

			cx = x3 - x1;
			cy = y3 - y1;

			ax = ax * ax;
			ay = ay * ay;
			bx = bx * bx;
			by = by * by;
			cx = cx * cx;
			cy = cy * cy;

			a = ax + ay;
			b = bx + by;
			c = cx + cy;

			//ar1 = x2 * y3 - x3 * y2 - x1 * y3 + x3 * y1 + x1 * y2 - x2 * y1;	
			//ar2 = x5 * y6 - x6 * y5 - x4 * y6 + x6 * y4 + x4 * y5 - x5 * y4;	
			ar1 = orient2dslow(pa, pb, pc);
			ar2 = orient2dslow(pd, pe, pf); 

			alpha1 = (a * b * c) / (ar1 * ar1);

			/*permanent1 = Absolute(x6 * y5) + Absolute(x5 * y6) + 
						 Absolute(x4 * y6) + Absolute(x6 * y4) + 
						 Absolute(x4 * y5) + Absolute(x5 * y4);
			permanent1 = a * b * c * permanent1 * permanent1;*/
			/*permanent1 = Absolute(x2 * y3) + Absolute(x3 * y2) + 
						 Absolute(x1 * y3) + Absolute(x3 * y1) + 
						 Absolute(x1 * y2) + Absolute(x2 * y1);*/
			permanent1 = (a * b * c) / (ar1 * ar1);


			ax = x4 - x5;
			ay = y4 - y5;
			
			bx = x5 - x6;
			by = y5 - y6;

			cx = x6 - x4;
			cy = y6 - y4;

			ax = ax * ax;
			ay = ay * ay;
			bx = bx * bx;
			by = by * by;
			cx = cx * cx;
			cy = cy * cy;

			a = ax + ay;
			b = bx + by;
			c = cx + cy;
								
			alpha2 = (a * b * c) / (ar2 * ar2);

			/*permanent2 = Absolute(x2 * y3) + Absolute(x3 * y2) + 
						 Absolute(x1 * y3) + Absolute(x3 * y1) + 
						 Absolute(x1 * y2) + Absolute(x2 * y1);*/
			/*permanent2 = Absolute(x6 * y5) + Absolute(x5 * y6) + 
						 Absolute(x4 * y6) + Absolute(x6 * y4) + 
						 Absolute(x4 * y5) + Absolute(x5 * y4);*/
			permanent2 = (a * b * c) / (ar2 * ar2);

			
			diff = alpha1 - alpha2;			
			permanent = permanent1 + permanent2;			
			 
			errbound = a2trierrbound * permanent;
			if ((diff > errbound) || (-diff > errbound)) {
				return convert(diff);
			}
									
			return -2;
		}	

		// Compare Triangle > Edge
		__device__ __host__ int compareFast(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pc, gpudtVertex &pd, gpudtVertex &pe, int print)
		{
			REAL ax, ay, bx, by, cx, cy;
			REAL a, b, c;
			REAL ar1;
			REAL alpha1, alpha2, diff;
			REAL permanent1, permanent2, permanent;
			REAL errbound;

			// Points
			REAL x1,x2,x3,y1,y2,y3;
			REAL x4,x5,y4,y5;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;
			x3 = pc.x; y3 = pc.y;
			x4 = pd.x; y4 = pd.y;
			x5 = pe.x; y5 = pe.y;
			
			ax = x1 - x2;
			ay = y1 - y2;
			
			bx = x2 - x3;
			by = y2 - y3;

			cx = x3 - x1;
			cy = y3 - y1;

			ax = ax * ax;
			ay = ay * ay;
			bx = bx * bx;
			by = by * by;
			cx = cx * cx;
			cy = cy * cy;

			a = ax + ay;
			b = bx + by;
			c = cx + cy;
				
			REAL pr = (a * b * c);

			//ar1 = x2 * y3 - x3 * y2 - x1 * y3 + x3 * y1 + x1 * y2 - x2 * y1;
			ar1 = orient2dslow(pa, pb, pc);
			alpha1 = pr / (ar1*ar1);

			/*permanent1 = Absolute(pc.x * pb.y) + Absolute(pb.x * pc.y) + 
						 Absolute(pa.x * pc.y) + Absolute(pc.x * pa.y) + 
						 Absolute(pa.x * pb.y) + Absolute(pb.x * pa.y);*/
			permanent1 = alpha1;

			ax = x4 - x5;
			ay = y4 - y5;
			ax = ax * ax;
			ay = ay * ay;
			a = ax + ay;
				
			//ar1 = determinant_3(pa,pb,pc);			
			alpha2 = a /** ar1 * ar1*/;

			/*permanent2 = Absolute(pc.x * pb.y) + Absolute(pb.x * pc.y) + 
						 Absolute(pa.x * pc.y) + Absolute(pc.x * pa.y) + 
						 Absolute(pa.x * pb.y) + Absolute(pb.x * pa.y);*/			
			permanent2 = a;

			
			diff = alpha1 - alpha2;		
			permanent = permanent1 + permanent2;			

			errbound = a2triedgeerrbound * permanent;
			if ((diff > errbound) || (-diff > errbound)) {
				return convert(diff);
			}
			
			return -2;
		}

		// Compare Edge > Triangle
		__device__ __host__ int compareFast(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf, int cfe, int print)
		{			
			REAL ax, ay, bx, by, cx, cy;
			REAL a, b, c;
			REAL ar2;
			REAL alpha1, alpha2, diff;
			REAL permanent1, permanent2, permanent;
			REAL errbound;

			// Points
			REAL x1,x2,y1,y2;
			REAL x4,x5,x6,y4,y5,y6;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;			
			x4 = pd.x; y4 = pd.y;
			x5 = pe.x; y5 = pe.y;
			x6 = pf.x; y6 = pf.y;

			ax = x1 - x2;
			ay = y1 - y2;
			ax = ax * ax;
			ay = ay * ay;
			a = ax + ay;

			//ar2 = x5 * y6 - x6 * y5 - x4 * y6 + x6 * y4 + x4 * y5 - x5 * y4;
			ar2 = orient2dslow(pd,pe,pf);			
			alpha1 = a /** ar2 * ar2*/;

			/*permanent1 = Absolute(pe.x * pf.y) + Absolute(pf.x * pe.y) + 
						 Absolute(pd.x * pf.y) + Absolute(pf.x * pd.y) + 
						 Absolute(pd.x * pe.y) + Absolute(pe.x * pd.y);*/			
			permanent1 = /*permanent1 * permanent1 **/ a;

			ax = x4 - x5;
			ay = y4 - y5;
			
			bx = x5 - x6;
			by = y5 - y6;

			cx = x6 - x4;
			cy = y6 - y4;

			ax = ax * ax;
			ay = ay * ay;
			bx = bx * bx;
			by = by * by;
			cx = cx * cx;
			cy = cy * cy;

			a = ax + ay;
			b = bx + by;
			c = cx + cy;
			
			REAL pr = (a * b * c);

			alpha2 = pr / (ar2 * ar2);
			/*permanent2 = Absolute(pe.x * pf.y) + Absolute(pf.x * pe.y) + 
						 Absolute(pd.x * pf.y) + Absolute(pf.x * pd.y) + 
						 Absolute(pd.x * pe.y) + Absolute(pe.x * pd.y);*/
			permanent2 = alpha2;
			
			diff = alpha1 - alpha2;			
			permanent = permanent1 + permanent2;			

			errbound = a2triedgeerrbound * permanent;
			if ((diff > errbound) || (-diff > errbound)) {
				return convert(diff);
			}
			
			return -2;
		}

		// Compare Edge1 > Edge2
		__device__ __host__ int compareFast(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pd, gpudtVertex &pe, int print)
		{
			REAL ax, ay;
			REAL a;			
			REAL alpha1, alpha2, diff;
			REAL permanent;
			REAL errbound;

			// Points
			REAL x1,x2,y1,y2;
			REAL x4,x5,y4,y5;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;			
			x4 = pd.x; y4 = pd.y;
			x5 = pe.x; y5 = pe.y;

			ax = x1 - x2;
			ay = y1 - y2;
			ax = ax * ax;
			ay = ay * ay;
			a = ax + ay;
			
			alpha1 = a;		

			ax = x4 - x5;
			ay = y4 - y5;
			ax = ax * ax;
			ay = ay * ay;
			a = ax + ay;
						
			alpha2 = a;

			diff = alpha1 - alpha2;			
			permanent = alpha1 + alpha2;					

			errbound = a2edgeErrorbound * permanent;
			if ((diff > errbound) || (-diff > errbound)) {
				return convert(diff);
			}
			
			return -2;
		}

		__device__ __host__ int alpha2dtri(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pc, gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf, REAL *a_sq,
									PrecisionData p_d, int tid)
		{
			// For comments, gpudtVertex pa will be referred as (x1,y1), pb as (x2,y2) .. and pf as (x6,y6)			

			// Variables needed for Sum, Multiplication
			INEXACT REAL bvirt;
			REAL avirt, bround, around;			
			INEXACT REAL abig;
			REAL c;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;
			INEXACT REAL _i, _j, _k, _l;
			REAL _0, _1, _2;			

			// Variable to locally multiply and keep a running sum upto now
			REAL x1,x2,x3,y1,y2,y3;
			REAL x4,x5,x6,y4,y5,y6;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;
			x3 = pc.x; y3 = pc.y;
			x4 = pd.x; y4 = pd.y;
			x5 = pe.x; y5 = pe.y;
			x6 = pf.x; y6 = pf.y;			

			REAL ax1, ay1;
			REAL ax0, ay0;
			REAL ax_sq[6], ay_sq[6];
								
			int temp2alen, temp2blen;
			int rSumlen1 = 0, rSumlen2 = 0;			

			REAL *temp2a = p_d.temp2a._array + 200*tid;
			REAL *temp2b = p_d.temp2b._array + 200*tid;
			REAL *runningSum1 = p_d.runningSum1._array + 200*tid;
			REAL *runningSum2 = p_d.runningSum2._array + 200*tid;			

			REAL tx, ty, tz, ta, tb, tc;
			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) = p1			
			Two_Diff(x1,x2,ax1,ax0);																// Compute (x1-x2)			
			Two_Diff(y1,y2,ay1,ay0);																// Compute (y1-y2)		
			Two_Square(ax1, ax0, tx, ty, tz, ta, tb, tc);											// Compute (x1-x2)^2						
			ax_sq[5] = tx; ax_sq[4] = ty; ax_sq[3] = tz; ax_sq[2] = ta; ax_sq[1] = tb; ax_sq[0] = tc;
			Two_Square(ay1, ay0, tx, ty, tz, ta, tb, tc);											// Compute (y1-y2)^2			
			ay_sq[5] = tx; ay_sq[4] = ty; ay_sq[3] = tz; ay_sq[2] = ta; ay_sq[1] = tb; ay_sq[0] = tc;
			int a_sq_length = fast_expansion_sum_zeroelim(6, ax_sq, 6, ay_sq, a_sq);					// Compute ((x1-x2)^2 + (y1-y2)^2)	[max 12 floats]			

			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) * ( x2^2 -2*x2*x3 + x3^2 + y2^2 - 2*y1*y2 + y3^2 )	= p2
			REAL op1, op2;
			op1 = x2;
			op2 = x3;			
																		
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2^2				[12f*2 = 24f]
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op1, temp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				

			for(int ij=0; ij<temp2blen; ij++)
				runningSum2[ij] = temp2b[ij];
			rSumlen2 = temp2blen;			
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2*x3				[12f*2 = 24f]
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -2*op2, temp2b);					// p1*(x2^x3 tail)		[12f*2 = 24f]												

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);				// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]							
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op2, temp2a);					// p1*x3^2				[12f*2 = 24f]			
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op2, temp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]							

			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,							// [48f+97f = 145f][48f+242f=290f]
													temp2blen, temp2b, runningSum2);				// p1*x2^2 - p1*2*x2*x3 + p1*x3^2															

			op1 = y2;
			op2 = y3;
																		
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2^2				[12f*2 = 24f]
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op1, temp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
														temp2blen, temp2b, runningSum1);			// Store p1*x2^2		[48f+0f = 48f][48f+145f=193f]				
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2*x3				[12f*2 = 24f]
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -2*op2, temp2b);					// p1*(x2^x3 tail)		[12f*2 = 24f]												

			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);				// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]				
			if(rSumlen2 < 0)	return -2;
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op2, temp2a);					// p1*x3^2				[12f*2 = 24f]			
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op2, temp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]							

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,							// [48f+97f = 145f][48f+242f=290f]
													temp2blen, temp2b, runningSum1);				// p1*x2^2 - p1*2*x2*x3 + p1*x3^2				
			if(rSumlen1 < 0)	return -2;			

			for(int j=0; j<rSumlen1; j++)	a_sq[j] = runningSum1[j];
			a_sq_length = rSumlen1;			
			rSumlen1 = 0; rSumlen2 = 0;			

			// Compute p2 * ( x3^2 -2*x3*x1 + x1^2 + y3^2 - 2*y3*y1 + y1^2 )	= p3
			op1 = x3;
			op2 = x1;			
																		
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2^2				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op1, temp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				
			if(temp2blen < 0)	return -2;

			for(int ij=0; ij<temp2blen; ij++)
				runningSum2[ij] = temp2b[ij];
			rSumlen2 = temp2blen;				
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2*x3				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -2*op2, temp2b);					// p1*(x2^x3 tail)		[12f*2 = 24f]												
			if(temp2blen < 0)	return -2;

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);				// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]				
			if(rSumlen1 < 0)	return -2;
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op2, temp2a);					// p1*x3^2				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op2, temp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]				
			if(temp2blen < 0)	return -2;

			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,							// [48f+97f = 145f][48f+242f=290f]
													temp2blen, temp2b, runningSum2);				// p1*x2^2 - p1*2*x2*x3 + p1*x3^2				
			if(rSumlen2 < 0)	return -2;								

			op1 = y3;
			op2 = y1;
																		
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2^2				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op1, temp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				
			if(temp2blen < 0)	return -2;

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
														temp2blen, temp2b, runningSum1);			// Store p1*x2^2		[48f+0f = 48f][48f+145f=193f]				
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2*x3				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -2*op2, temp2b);					// p1*(x2^x3 tail)		[12f*2 = 24f]												
			if(temp2blen < 0)	return -2;

			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);				// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]				
			if(rSumlen2 < 0)	return -2;
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op2, temp2a);					// p1*x3^2				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op2, temp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]				
			if(temp2blen < 0)	return -2;

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,							// [48f+97f = 145f][48f+242f=290f]
													temp2blen, temp2b, runningSum1);				// p1*x2^2 - p1*2*x2*x3 + p1*x3^2				
			if(rSumlen1 < 0)	return -2;			

			for(int j=0; j<rSumlen1; j++)	a_sq[j] = runningSum1[j];
			a_sq_length = rSumlen1;			
			
			rSumlen1 = 0; rSumlen2 = 0;

			// Compute p3 * (x2*y3 - x3*y2 - x1*y3 + x3*y1 + x1*y2 - x2*y1) = p4				
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x5, temp2a);					// p3*x2*y3				[5762f*2 = 11344f][68064f*2=136,128f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y6, temp2b);					// p3*(x2*y3 tail)		[5762f*2 = 11344f][68064f*2=136,128f]				
			if(temp2blen < 0)	return -2;

			for(int ij=0; ij<temp2blen; ij++)
				runningSum2[ij] = temp2b[ij];
			rSumlen2 = temp2blen;
			
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x6, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y5, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);
			if(rSumlen1 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x4, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y6, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);
			if(rSumlen2 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x6, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y4, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);
			if(rSumlen1 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x4, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y5, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);
			if(rSumlen2 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x5, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y4, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);				
			if(rSumlen1 < 0)	return -2;

			for(int j=0; j<rSumlen1; j++)	a_sq[j] = runningSum1[j];
			a_sq_length = rSumlen1;	
			
			rSumlen1 = 0; rSumlen2 = 0;

			// Compute p3 * (x2*y3 - x3*y2 - x1*y3 + x3*y1 + x1*y2 - x2*y1) = p4				
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x5, temp2a);					// p3*x2*y3				[5762f*2 = 11344f][68064f*2=136,128f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y6, temp2b);					// p3*(x2*y3 tail)		[5762f*2 = 11344f][68064f*2=136,128f]				
			if(temp2blen < 0)	return -2;

			for(int ij=0; ij<temp2blen; ij++)
				runningSum2[ij] = temp2b[ij];
			rSumlen2 = temp2blen;
			
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x6, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y5, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);
			if(rSumlen1 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x4, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y6, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);
			if(rSumlen2 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x6, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y4, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);
			if(rSumlen1 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x4, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y5, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);
			if(rSumlen2 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x5, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y4, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);				
			if(rSumlen1 < 0)	return -2;

			for(int j=0; j<rSumlen1; j++)	a_sq[j] = runningSum1[j];
			a_sq_length = rSumlen1;	

			return a_sq_length;
		}
		
		__device__ __host__ int alpha2dTriEdge(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pc, REAL *a_sq,
										PrecisionData p_d, int tid)
		{
			// For comments, gpudtVertex pa will be referred as (x1,y1), pb as (x2,y2) .. and pf as (x6,y6)			

			// Variables needed for Sum, Multiplication
			INEXACT REAL bvirt;
			REAL avirt, bround, around;			
			INEXACT REAL abig;
			REAL c;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;
			INEXACT REAL _i, _j, _k, _l;
			REAL _0, _1, _2;			

			// Variable to locally multiply and keep a running sum upto now
			REAL x1,x2,x3,y1,y2,y3;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;
			x3 = pc.x; y3 = pc.y;

			REAL ax1, ay1;
			REAL ax0, ay0;
			REAL ax_sq[6], ay_sq[6];
									
			int temp2alen, temp2blen;			
			int rSumlen1 = 0, rSumlen2 = 0;

			REAL *temp2a = p_d.temp2a._array + 200*tid;
			REAL *temp2b = p_d.temp2b._array + 200*tid;
			REAL *runningSum1 = p_d.runningSum1._array + 200*tid;
			REAL *runningSum2 = p_d.runningSum2._array + 200*tid;			
			
			REAL tx, ty, tz, ta, tb, tc;
			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) = p1			
			Two_Diff(x1,x2,ax1,ax0);																// Compute (x1-x2)			
			Two_Diff(y1,y2,ay1,ay0);																// Compute (y1-y2)		
			Two_Square(ax1, ax0, tx, ty, tz, ta, tb, tc);											// Compute (x1-x2)^2						
			ax_sq[5] = tx; ax_sq[4] = ty; ax_sq[3] = tz; ax_sq[2] = ta; ax_sq[1] = tb; ax_sq[0] = tc;
			Two_Square(ay1, ay0, tx, ty, tz, ta, tb, tc);											// Compute (y1-y2)^2			
			ay_sq[5] = tx; ay_sq[4] = ty; ay_sq[3] = tz; ay_sq[2] = ta; ay_sq[1] = tb; ay_sq[0] = tc;
			int a_sq_length = fast_expansion_sum_zeroelim(6, ax_sq, 6, ay_sq, a_sq);					// Compute ((x1-x2)^2 + (y1-y2)^2)	[max 12 floats]			

			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) * ( x2^2 -2*x2*x3 + x3^2 + y2^2 - 2*y1*y2 + y3^2 )	= p2						
			REAL op1, op2;
			op1 = x2;
			op2 = x3;			
																		
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2^2				[12f*2 = 24f]
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op1, temp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				

			for(int ij=0; ij<temp2blen; ij++)
				runningSum2[ij] = temp2b[ij];
			rSumlen2 = temp2blen;
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2*x3				[12f*2 = 24f]
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -2*op2, temp2b);					// p1*(x2^x3 tail)		[12f*2 = 24f]												

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);				// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]							
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op2, temp2a);					// p1*x3^2				[12f*2 = 24f]			
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op2, temp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]							

			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,							// [48f+97f = 145f][48f+242f=290f]
													temp2blen, temp2b, runningSum2);				// p1*x2^2 - p1*2*x2*x3 + p1*x3^2							

			op1 = y2;
			op2 = y3;
																		
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2^2				[12f*2 = 24f]
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op1, temp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
														temp2blen, temp2b, runningSum1);			// Store p1*x2^2		[48f+0f = 48f][48f+145f=193f]				
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2*x3				[12f*2 = 24f]
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -2*op2, temp2b);					// p1*(x2^x3 tail)		[12f*2 = 24f]												

			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);				// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]				
			if(rSumlen2 < 0)	return -2;
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op2, temp2a);					// p1*x3^2				[12f*2 = 24f]			
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op2, temp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]							

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,							// [48f+97f = 145f][48f+242f=290f]
													temp2blen, temp2b, runningSum1);				// p1*x2^2 - p1*2*x2*x3 + p1*x3^2				
			if(rSumlen1 < 0)	return -2;			

			for(int j=0; j<rSumlen1; j++)	a_sq[j] = runningSum1[j];
			a_sq_length = rSumlen1;
			runningSum1[0] = 0;
			rSumlen1 = 0; rSumlen2 = 0;			

			// Compute p2 * ( x3^2 -2*x3*x1 + x1^2 + y3^2 - 2*y3*y1 + y1^2 )	= p3
			op1 = x3;
			op2 = x1;			
																		
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2^2				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op1, temp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				
			if(temp2blen < 0)	return -2;

			for(int ij=0; ij<temp2blen; ij++)
				runningSum2[ij] = temp2b[ij];
			rSumlen2 = temp2blen;				
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2*x3				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -2*op2, temp2b);					// p1*(x2^x3 tail)		[12f*2 = 24f]												
			if(temp2blen < 0)	return -2;

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);				// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]				
			if(rSumlen1 < 0)	return -2;
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op2, temp2a);					// p1*x3^2				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op2, temp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]				
			if(temp2blen < 0)	return -2;

			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,							// [48f+97f = 145f][48f+242f=290f]
													temp2blen, temp2b, runningSum2);				// p1*x2^2 - p1*2*x2*x3 + p1*x3^2				
			if(rSumlen2 < 0)	return -2;								

			op1 = y3;
			op2 = y1;
																		
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2^2				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op1, temp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				
			if(temp2blen < 0)	return -2;

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
														temp2blen, temp2b, runningSum1);			// Store p1*x2^2		[48f+0f = 48f][48f+145f=193f]				
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op1, temp2a);					// p1*x2*x3				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -2*op2, temp2b);					// p1*(x2^x3 tail)		[12f*2 = 24f]												
			if(temp2blen < 0)	return -2;

			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);				// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]				
			if(rSumlen2 < 0)	return -2;
										
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, op2, temp2a);					// p1*x3^2				[12f*2 = 24f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, op2, temp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]				
			if(temp2blen < 0)	return -2;

			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,							// [48f+97f = 145f][48f+242f=290f]
													temp2blen, temp2b, runningSum1);				// p1*x2^2 - p1*2*x2*x3 + p1*x3^2				
			if(rSumlen1 < 0)	return -2;			

			for(int j=0; j<rSumlen1; j++)	a_sq[j] = runningSum1[j];
			a_sq_length = rSumlen1;			

			return a_sq_length;
		}
		
		__device__ __host__ int alpha2dTriEdge(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf, REAL *a_sq,
									PrecisionData p_d, int tid)
		{			
			// For comments, gpudtVertex pa will be referred as (x1,y1), pb as (x2,y2) .. and pf as (x6,y6)			

			// Variables needed for Sum, Multiplication
			INEXACT REAL bvirt;
			REAL avirt, bround, around;			
			INEXACT REAL abig;
			REAL c;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;
			INEXACT REAL _i, _j, _k, _l;
			REAL _0, _1, _2;			

			// Variable to locally multiply and keep a running sum upto now
			REAL x1,x2,y1,y2;
			REAL x4,x5,x6,y4,y5,y6;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;			
			x4 = pd.x; y4 = pd.y;
			x5 = pe.x; y5 = pe.y;
			x6 = pf.x; y6 = pf.y;

			REAL ax1, ay1;
			REAL ax0, ay0;
			REAL ax_sq[6], ay_sq[6];
						
			int temp2alen, temp2blen;			
			int rSumlen1 = 0, rSumlen2 = 0;

			REAL *temp2a = p_d.temp2a._array + 200*tid;
			REAL *temp2b = p_d.temp2b._array + 200*tid;
			REAL *runningSum1 = p_d.runningSum1._array + 200*tid;
			REAL *runningSum2 = p_d.runningSum2._array + 200*tid;			
			
			REAL tx, ty, tz, ta, tb, tc;
			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) = p1			
			Two_Diff(x1,x2,ax1,ax0);																// Compute (x1-x2)			
			Two_Diff(y1,y2,ay1,ay0);																// Compute (y1-y2)		
			Two_Square(ax1, ax0, tx, ty, tz, ta, tb, tc);											// Compute (x1-x2)^2						
			ax_sq[5] = tx; ax_sq[4] = ty; ax_sq[3] = tz; ax_sq[2] = ta; ax_sq[1] = tb; ax_sq[0] = tc;
			Two_Square(ay1, ay0, tx, ty, tz, ta, tb, tc);											// Compute (y1-y2)^2			
			ay_sq[5] = tx; ay_sq[4] = ty; ay_sq[3] = tz; ay_sq[2] = ta; ay_sq[1] = tb; ay_sq[0] = tc;
			int a_sq_length = fast_expansion_sum_zeroelim(6, ax_sq, 6, ay_sq, a_sq);					// Compute ((x1-x2)^2 + (y1-y2)^2)	[max 12 floats]		
						
			rSumlen1 = 0; rSumlen2 = 0;

			// Compute p3 * (x2*y3 - x3*y2 - x1*y3 + x3*y1 + x1*y2 - x2*y1) = p4				
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x5, temp2a);					// p3*x2*y3				[5762f*2 = 11344f][68064f*2=136,128f]			
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y6, temp2b);					// p3*(x2*y3 tail)		[5762f*2 = 11344f][68064f*2=136,128f]							

			for(int ij=0; ij<temp2blen; ij++)
				runningSum2[ij] = temp2b[ij];
			rSumlen2 = temp2blen;
			
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x6, temp2a);							
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y5, temp2b);							
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);			

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x4, temp2a);							
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y6, temp2b);							
			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);			

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x6, temp2a);							
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y4, temp2b);							
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);			

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x4, temp2a);							
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y5, temp2b);							
			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);
			if(rSumlen2 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x5, temp2a);							
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y4, temp2b);							
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);				
			if(rSumlen1 < 0)	return -2;

			for(int j=0; j<rSumlen1; j++)	a_sq[j] = runningSum1[j];
			a_sq_length = rSumlen1;	
			
			rSumlen1 = 0; rSumlen2 = 0;

			// Compute p3 * (x2*y3 - x3*y2 - x1*y3 + x3*y1 + x1*y2 - x2*y1) = p4				
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x5, temp2a);					// p3*x2*y3				[5762f*2 = 11344f][68064f*2=136,128f]
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y6, temp2b);					// p3*(x2*y3 tail)		[5762f*2 = 11344f][68064f*2=136,128f]				
			if(temp2blen < 0)	return -2;
			
			for(int ij=0; ij<temp2blen; ij++)
				runningSum2[ij] = temp2b[ij];
			rSumlen2 = temp2blen;
			
			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x6, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y5, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);
			if(rSumlen1 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x4, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y6, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);
			if(rSumlen2 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x6, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y4, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);
			if(rSumlen1 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x4, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, y5, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen2 = fast_expansion_sum_zeroelim(rSumlen1, runningSum1,
													temp2blen, temp2b, runningSum2);
			if(rSumlen2 < 0)	return -2;

			temp2alen = scale_expansion_zeroelim(a_sq_length, a_sq, x5, temp2a);				
			if(temp2alen < 0)	return -2;
			temp2blen = scale_expansion_zeroelim(temp2alen, temp2a, -y4, temp2b);				
			if(temp2blen < 0)	return -2;
			rSumlen1 = fast_expansion_sum_zeroelim(rSumlen2, runningSum2,
													temp2blen, temp2b, runningSum1);				
			if(rSumlen1 < 0)	return -2;

			for(int j=0; j<rSumlen1; j++)	a_sq[j] = runningSum1[j];
			a_sq_length = rSumlen1;			

			return a_sq_length;			
		}

		__device__ __host__ int alpha2dEdge(gpudtVertex &pa, gpudtVertex &pb, REAL *a_sq, PrecisionData p_d, int tid)
		{
			// For comments, gpudtVertex pa will be referred as (x1,y1), pb as (x2,y2) .. and pf as (x6,y6)			
			INEXACT REAL bvirt;
			REAL avirt, bround, around;			
			INEXACT REAL abig;
			REAL c;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;
			INEXACT REAL _i, _j, _k, _l;
			REAL _0, _1, _2;

			REAL ax1, ay1;
			REAL ax0, ay0;
			REAL ax_sq[6], ay_sq[6];

			REAL x1,x2,y1,y2;			
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;			
			
			REAL tx, ty, tz, ta, tb, tc;
			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) = p1			
			Two_Diff(x1,x2,ax1,ax0);																// Compute (x1-x2)			
			Two_Diff(y1,y2,ay1,ay0);																// Compute (y1-y2)		
			Two_Square(ax1, ax0, tx, ty, tz, ta, tb, tc);											// Compute (x1-x2)^2						
			ax_sq[5] = tx; ax_sq[4] = ty; ax_sq[3] = tz; ax_sq[2] = ta; ax_sq[1] = tb; ax_sq[0] = tc;
			Two_Square(ay1, ay0, tx, ty, tz, ta, tb, tc);											// Compute (y1-y2)^2			
			ay_sq[5] = tx; ay_sq[4] = ty; ay_sq[3] = tz; ay_sq[2] = ta; ay_sq[1] = tb; ay_sq[0] = tc;
			int a_sq_length = fast_expansion_sum_zeroelim(6, ax_sq, 6, ay_sq, a_sq);					// Compute ((x1-x2)^2 + (y1-y2)^2)	[max 12 floats]	
						
			return a_sq_length;
		}		

		// To compare edge > Triangle
		__device__ __host__ int compareSlow(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf, int x, 
									PrecisionData p_d, int tid)
		{						
			REAL *P1 = p_d.P1._array + 200*tid;
			REAL *P2 = p_d.P2._array + 200*tid;
			REAL *ans = p_d.runningSum1._array + 200*tid;			
			
			int P1_len = alpha2dTriEdge(pa,pb,pd,pe,pf, P1, p_d,tid);
			int P2_len = alpha2dTriEdge(pd,pe,pf, P2, p_d,tid);			

			if( P1_len == -2 && P2_len == -2 ) return -4;
			else if( P1_len == -2 ) return -2;
			else if( P2_len == -2 ) return -3;

			for(int i=0; i<P2_len; i++) 
			{
				P2[i] = -P2[i];				
			}
			int ans_len = fast_expansion_sum_zeroelim(P1_len, P1, P2_len, P2, ans);

			if( ans[ ans_len-1 ] > 0 )
				return 1;
			else if( ans[ ans_len-1 ] == 0 )
				return 0;
			else
				return -1;			
		}

		// To compare if Triangle > edge
		__device__ __host__ int compareSlow(gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf, gpudtVertex &pa, gpudtVertex &pb,
									PrecisionData p_d, int tid)
		{		
			REAL *P1 = p_d.P1._array + 200*tid;
			REAL *P2 = p_d.P2._array + 200*tid;
			REAL *ans = p_d.runningSum1._array + 200*tid;						

			int P1_len = alpha2dTriEdge(pd,pe,pf, P1, p_d,tid);
			int P2_len = alpha2dTriEdge(pa,pb,pd,pe,pf, P2, p_d,tid);			

			if( P1_len == -2 && P2_len == -2 ) return -4;
			else if( P1_len == -2 ) return -2;
			else if( P2_len == -2 ) return -3;

			for(int i=0; i<P2_len; i++) 
			{
				P2[i] = -P2[i];				
			}		
			int ans_len = fast_expansion_sum_zeroelim(P1_len, P1, P2_len, P2, ans);

			if( ans[ ans_len-1 ] > 0 )
				return 1;
			else if( ans[ ans_len-1 ] == 0 )
				return 0;
			else
				return -1;			
		}

		// To Compare if Triangle1 > Triangle  
		__device__ __host__ int compareSlow(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pc, gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf,
									PrecisionData p_d, int tid)
		{				 			
			REAL *P1 = p_d.P1._array + 200*tid;
			REAL *P2 = p_d.P2._array + 200*tid;
			REAL *ans = p_d.runningSum1._array + 200*tid;			

			int P1_len = alpha2dtri(pa,pb,pc,pd,pe,pf, P1, p_d,tid);
			int P2_len = alpha2dtri(pd,pe,pf,pa,pb,pc, P2, p_d,tid);			

			if( P1_len == -2 && P2_len == -2 ) return -4;
			else if( P1_len == -2 ) return -2;
			else if( P2_len == -2 ) return -3;

			for(int i=0; i<P2_len; i++) 
			{
				P2[i] = -P2[i];				
			}	
			int ans_len = fast_expansion_sum_zeroelim(P1_len, P1, P2_len, P2, ans);

			if( ans[ ans_len-1 ] > 0 )
				return 1;
			else if( ans[ ans_len-1 ] == 0 )
				return 0;
			else
				return -1;			
		}

		// To Compare if Edge1 > Edge2
		__device__ __host__ int compareSlow(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pc, gpudtVertex &pd,
									PrecisionData p_d, int tid)
		{						
			REAL *P1 = p_d.P1._array + 200*tid;
			REAL *P2 = p_d.P2._array + 200*tid;
			REAL *ans = p_d.runningSum1._array + 200*tid;						

			int P1_len = alpha2dEdge(pa,pb, P1, p_d,tid);
			int P2_len = alpha2dEdge(pc,pd, P2, p_d,tid);

			for(int i=0; i<P2_len; i++) 
			{
				P2[i] = -P2[i];				
			}		
			int ans_len = fast_expansion_sum_zeroelim(P1_len, P1, P2_len, P2, ans);			

			if( ans[ ans_len-1 ] == 0 )
				return 0;
			else if( ans[ ans_len-1 ] > 0 )
				return 1;
			else
				return -1;				
		}

		/* CPU Functions : To compare two simplexes precisely for 2D Alpha Shapes */

		int CPUalpha2dtri(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pc, gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf, REAL *a_sq)
		{
			// For comments, gpudtVertex pa will be referred as (x1,y1), pb as (x2,y2) .. and pf as (x6,y6)			

			// Variables needed for Sum, Multiplication
			INEXACT REAL bvirt;
			REAL avirt, bround, around;			
			INEXACT REAL abig;
			REAL c;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;
			INEXACT REAL _i, _j, _k, _l;
			REAL _0, _1, _2;			

			// Variable to locally multiply and keep a running sum upto now
			REAL x1,x2,x3,y1,y2,y3;
			REAL x4,x5,x6,y4,y5,y6;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;
			x3 = pc.x; y3 = pc.y;
			x4 = pd.x; y4 = pd.y;
			x5 = pe.x; y5 = pe.y;
			x6 = pf.x; y6 = pf.y;
			//shift(x1,y1,x2,y2,x3,y3);

			REAL ax1, ay1;
			REAL ax0, ay0;
			REAL ax_sq[6], ay_sq[6];		
			
			int rSumlen1 = 1, rSumlen2 = 0;			
			REAL tmp;

			for(int kp=0; kp<200; kp++)	{ CPUrunningSum1[kp] = 0; }

			// abracadabra
			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) = p1			
			Two_Diff(x1,x2,ax1,ax0);																// Compute (x1-x2)			
			Two_Diff(y1,y2,ay1,ay0);																// Compute (y1-y2)		
			Two_Square(ax1, ax0, tmp, ax_sq[4], ax_sq[3], ax_sq[2], ax_sq[1], ax_sq[0]);		// Compute (x1-x2)^2			
			ax_sq[5] = tmp;
			Two_Square(ay1, ay0, tmp, ay_sq[4], ay_sq[3], ay_sq[2], ay_sq[1], ay_sq[0]);		// Compute (y1-y2)^2
			ay_sq[5] = tmp;
			int a_sq_length = cpu_fast_expansion_sum_zeroelim(6, ax_sq, 6, ay_sq, a_sq);				// Compute ((x1-x2)^2 + (y1-y2)^2)	[max 12 floats]			

			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) * ( x2^2 -2*x2*x3 + x3^2 + y2^2 - 2*y1*y2 + y3^2 )	= p2
			REAL op1, op2;
			for(int i=0; i<2; i++)
			{		
				if(!i)	op1 = x2, op2 = x3;
				else	op1 = y2, op2 = y3;
																		
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op1, CPUtemp2a);					// p1*x2^2				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, op1, CPUtemp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				

				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,
															CPUtemp2blen, CPUtemp2b, CPUrunningSum2);			// Store p1*x2^2		[48f+0f = 48f][48f+145f=193f]				
											
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op1, CPUtemp2a);					// p1*x2*x3				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, -2*op2, CPUtemp2b);		// p1*(x2^x3 tail)		[12f*2 = 24f]					

				rSumlen1 = cpu_fast_expansion_sum_zeroelim(rSumlen2, CPUrunningSum2,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum1);					// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]				
											
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op2, CPUtemp2a);					// p1*x3^2				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, op2, CPUtemp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]				

				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,							// [48f+97f = 145f][48f+242f=290f]
														CPUtemp2blen, CPUtemp2b, CPUrunningSum2);				// p1*x2^2 - p1*2*x2*x3 + p1*x3^2				
				
				for(int j=0; j<rSumlen2; j++)	CPUrunningSum1[j] = CPUrunningSum2[j];
				rSumlen1 = rSumlen2;
			}
			
			for(int j=0; j<rSumlen1; j++)	a_sq[j] = CPUrunningSum1[j];
			a_sq_length = rSumlen1;
			for(int kp=0; kp<rSumlen1; kp++)	{ CPUrunningSum1[kp] = 0; }
			rSumlen1 = 0; rSumlen2 = 0;				

			// Compute p2 * ( x3^2 -2*x3*x1 + x1^2 + y3^2 - 2*y3*y1 + y1^2 )	= p3
			for(int i=0; i<2; i++)
			{
				if(!i)	op1 = x3, op2 = x1;
				else	op1 = y3, op2 = y1;

				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op1, CPUtemp2a);					// p1*x2^2				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, op1, CPUtemp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				

				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,
															CPUtemp2blen, CPUtemp2b, CPUrunningSum2);			// Store p1*x2^2		[48f+0f = 48f][48f+145f=193f]				
											
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op1, CPUtemp2a);					// p1*x2*x3				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, -2*op2, CPUtemp2b);		// p1*(x2^x3 tail)		[12f*2 = 24f]					

				rSumlen1 = cpu_fast_expansion_sum_zeroelim(rSumlen2, CPUrunningSum2,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum1);				// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]				
											
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op2, CPUtemp2a);					// p1*x3^2				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, op2, CPUtemp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]				

				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,							// [48f+97f = 145f][48f+242f=290f]
														CPUtemp2blen, CPUtemp2b, CPUrunningSum2);					// p1*x2^2 - p1*2*x2*x3 + p1*x3^2				
				
				for(int j=0; j<rSumlen2; j++)	CPUrunningSum1[j] = CPUrunningSum2[j];
				rSumlen1 = rSumlen2;
			}

			for(int j=0; j<rSumlen1; j++)	a_sq[j] = CPUrunningSum1[j];
			a_sq_length = rSumlen1;

			for(int i=0; i<2; i++)
			{
				for(int kp=0; kp<rSumlen1; kp++)	{ CPUrunningSum1[kp] = 0; }
				rSumlen1 = 0; rSumlen2 = 0;

				// Compute p3 * (x2*y3 - x3*y2 - x1*y3 + x3*y1 + x1*y2 - x2*y1) = p4				
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x5, CPUtemp2a);					// p3*x2*y3				[5762f*2 = 11344f][68064f*2=136,128f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, y6, CPUtemp2b);					// p3*(x2*y3 tail)		[5762f*2 = 11344f][68064f*2=136,128f]				
				rSumlen2 = cpu_fast_expansion_sum_zeroelim(CPUtemp2blen, CPUtemp2b, 
																rSumlen1, CPUrunningSum1, CPUrunningSum2);	// Store p3*x2*y3				
				
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x6, CPUtemp2a);				
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, -y5, CPUtemp2b);				
				rSumlen1 = cpu_fast_expansion_sum_zeroelim(rSumlen2, CPUrunningSum2,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum1);

				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x4, CPUtemp2a);				
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, -y6, CPUtemp2b);				
				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum2);			

				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x6, CPUtemp2a);				
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, y4, CPUtemp2b);				
				rSumlen1 = cpu_fast_expansion_sum_zeroelim(rSumlen2, CPUrunningSum2,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum1);				

				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x4, CPUtemp2a);				
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, y5, CPUtemp2b);				
				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum2);

				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x5, CPUtemp2a);				
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, -y4, CPUtemp2b);				
				rSumlen1 = cpu_fast_expansion_sum_zeroelim(rSumlen2, CPUrunningSum2,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum1);				

				for(int j=0; j<rSumlen1; j++)	a_sq[j] = CPUrunningSum1[j];
				a_sq_length = rSumlen1;				
			}
						
			return a_sq_length;
		}
		
		int CPUalpha2dTriEdge(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pc, REAL *a_sq)
		{
			// For comments, gpudtVertex pa will be referred as (x1,y1), pb as (x2,y2) .. and pf as (x6,y6)			

			// Variables needed for Sum, Multiplication
			INEXACT REAL bvirt;
			REAL avirt, bround, around;			
			INEXACT REAL abig;
			REAL c;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;
			INEXACT REAL _i, _j, _k, _l;
			REAL _0, _1, _2;			

			// Variable to locally multiply and keep a running sum upto now
			REAL x1,x2,x3,y1,y2,y3;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;
			x3 = pc.x; y3 = pc.y;

			REAL ax1, ay1;
			REAL ax0, ay0;
			REAL ax_sq[6], ay_sq[6];			
			
			int rSumlen1 = 1, rSumlen2 = 0;
			for(int i=0; i<200; i++)	{ CPUrunningSum1[i] = 0; CPUrunningSum2[i] = 0; }
			REAL tmp;

			// abracadabra
			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) = p1			
			Two_Diff(x1,x2,ax1,ax0);																// Compute (x1-x2)			
			Two_Diff(y1,y2,ay1,ay0);																// Compute (y1-y2)		
			Two_Square(ax1, ax0, tmp, ax_sq[4], ax_sq[3], ax_sq[2], ax_sq[1], ax_sq[0]);		// Compute (x1-x2)^2			
			ax_sq[5] = tmp;
			Two_Square(ay1, ay0, tmp, ay_sq[4], ay_sq[3], ay_sq[2], ay_sq[1], ay_sq[0]);		// Compute (y1-y2)^2
			ay_sq[5] = tmp;
			int a_sq_length = cpu_fast_expansion_sum_zeroelim(6, ax_sq, 6, ay_sq, a_sq);				// Compute ((x1-x2)^2 + (y1-y2)^2)	[max 12 floats]				

			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) * ( x2^2 -2*x2*x3 + x3^2 + y2^2 - 2*y1*y2 + y3^2 )	= p2
			REAL op1, op2;			
			for(int i=0; i<2; i++)
			{		
				if(!i)	op1 = x2, op2 = x3;
				else	op1 = y2, op2 = y3;
																		
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op1, CPUtemp2a);					// p1*x2^2				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, op1, CPUtemp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				

				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,
															CPUtemp2blen, CPUtemp2b, CPUrunningSum2);			// Store p1*x2^2		[48f+0f = 48f][48f+145f=193f]				
											
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op1, CPUtemp2a);					// p1*x2*x3				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, -2*op2, CPUtemp2b);		// p1*(x2^x3 tail)		[12f*2 = 24f]					

				rSumlen1 = cpu_fast_expansion_sum_zeroelim(rSumlen2, CPUrunningSum2,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum1);					// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]				
											
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op2, CPUtemp2a);					// p1*x3^2				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, op2, CPUtemp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]				

				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,							// [48f+97f = 145f][48f+242f=290f]
														CPUtemp2blen, CPUtemp2b, CPUrunningSum2);					// p1*x2^2 - p1*2*x2*x3 + p1*x3^2				
				
				for(int j=0; j<rSumlen2; j++)	CPUrunningSum1[j] = CPUrunningSum2[j];
				rSumlen1 = rSumlen2;
			}
			
			for(int j=0; j<rSumlen1; j++)	a_sq[j] = CPUrunningSum1[j];
			a_sq_length = rSumlen1;
			for(int kp=0; kp<rSumlen1; kp++)	{ CPUrunningSum1[kp] = 0; }			
			rSumlen1 = 0; rSumlen2 = 0;			

			// Compute p2 * ( x3^2 -2*x3*x1 + x1^2 + y3^2 - 2*y3*y1 + y1^2 )	= p3
			for(int i=0; i<2; i++)
			{
				if(!i)	op1 = x3, op2 = x1;
				else	op1 = y3, op2 = y1;

				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op1, CPUtemp2a);					// p1*x2^2				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, op1, CPUtemp2b);					// p1*(x2^2 tail)		[12f*2 = 24f]				

				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,
															CPUtemp2blen, CPUtemp2b, CPUrunningSum2);			// Store p1*x2^2		[48f+0f = 48f][48f+145f=193f]				
											
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op1, CPUtemp2a);					// p1*x2*x3				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, -2*op2, CPUtemp2b);		// p1*(x2^x3 tail)		[12f*2 = 24f]					

				rSumlen1 = cpu_fast_expansion_sum_zeroelim(rSumlen2, CPUrunningSum2,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum1);				// p1*x2^2 - p1*2*x2*x3 [49f+48f = 97f][49f+193f=242]				
											
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, op2, CPUtemp2a);					// p1*x3^2				[12f*2 = 24f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, op2, CPUtemp2b);					// p1*(x3^2 tail)		[12f*2 = 24f]				

				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,							// [48f+97f = 145f][48f+242f=290f]
														CPUtemp2blen, CPUtemp2b, CPUrunningSum2);					// p1*x2^2 - p1*2*x2*x3 + p1*x3^2				
				
				for(int j=0; j<rSumlen2; j++)	CPUrunningSum1[j] = CPUrunningSum2[j];
				rSumlen1 = rSumlen2;
			}

			for(int j=0; j<rSumlen1; j++)	a_sq[j] = CPUrunningSum1[j];
			a_sq_length = rSumlen1;	
						
			return a_sq_length;
		}
		
		int CPUalpha2dTriEdge(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf, REAL *a_sq)
		{			
			// For comments, gpudtVertex pa will be referred as (x1,y1), pb as (x2,y2) .. and pf as (x6,y6)			

			// Variables needed for Sum, Multiplication
			INEXACT REAL bvirt;
			REAL avirt, bround, around;			
			INEXACT REAL abig;
			REAL c;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;
			INEXACT REAL _i, _j, _k, _l;
			REAL _0, _1, _2;			

			// Variable to locally multiply and keep a running sum upto now
			REAL x1,x2,y1,y2;
			REAL x4,x5,x6,y4,y5,y6;
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;			
			x4 = pd.x; y4 = pd.y;
			x5 = pe.x; y5 = pe.y;
			x6 = pf.x; y6 = pf.y;

			REAL ax1, ay1;
			REAL ax0, ay0;
			REAL ax_sq[6], ay_sq[6];
			
			int rSumlen1 = 0, rSumlen2 = 0;
			for(int i=0; i<200; i++)	CPUrunningSum1[i] = 0, CPUrunningSum2[i] = 0;
			REAL tmp;

			// abracadabra
			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) = p1			
			Two_Diff(x1,x2,ax1,ax0);																// Compute (x1-x2)			
			Two_Diff(y1,y2,ay1,ay0);																// Compute (y1-y2)		
			Two_Square(ax1, ax0, tmp, ax_sq[4], ax_sq[3], ax_sq[2], ax_sq[1], ax_sq[0]);		// Compute (x1-x2)^2			
			ax_sq[5] = tmp;
			Two_Square(ay1, ay0, tmp, ay_sq[4], ay_sq[3], ay_sq[2], ay_sq[1], ay_sq[0]);		// Compute (y1-y2)^2
			ay_sq[5] = tmp;
			int a_sq_length = cpu_fast_expansion_sum_zeroelim(6, ax_sq, 6, ay_sq, a_sq);				// Compute ((x1-x2)^2 + (y1-y2)^2)	[max 12 floats]	
			
			for(int i=0; i<2; i++)
			{
				for(int kp=0; kp<rSumlen1; kp++)	{ CPUrunningSum1[kp] = 0; }
				rSumlen1 = 0; rSumlen2 = 0;

				// Compute p3 * (x2*y3 - x3*y2 - x1*y3 + x3*y1 + x1*y2 - x2*y1) = p4				
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x5, CPUtemp2a);					// p3*x2*y3				[5762f*2 = 11344f][68064f*2=136,128f]
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, y6, CPUtemp2b);			// p3*(x2*y3 tail)		[5762f*2 = 11344f][68064f*2=136,128f]				
				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1, 
														CPUtemp2blen, CPUtemp2b, CPUrunningSum2);				// Store p3*x2*y3				
				
				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x6, CPUtemp2a);				
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, -y5, CPUtemp2b);				
				rSumlen1 = cpu_fast_expansion_sum_zeroelim(rSumlen2, CPUrunningSum2,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum1);

				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x4, CPUtemp2a);				
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, -y6, CPUtemp2b);				
				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum2);			

				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x6, CPUtemp2a);				
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, y4, CPUtemp2b);				
				rSumlen1 = cpu_fast_expansion_sum_zeroelim(rSumlen2, CPUrunningSum2,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum1);				

				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x4, CPUtemp2a);				
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, y5, CPUtemp2b);				
				rSumlen2 = cpu_fast_expansion_sum_zeroelim(rSumlen1, CPUrunningSum1,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum2);

				CPUtemp2alen = cpu_scale_expansion_zeroelim(a_sq_length, a_sq, x5, CPUtemp2a);				
				CPUtemp2blen = cpu_scale_expansion_zeroelim(CPUtemp2alen, CPUtemp2a, -y4, CPUtemp2b);				
				rSumlen1 = cpu_fast_expansion_sum_zeroelim(rSumlen2, CPUrunningSum2,
														CPUtemp2blen, CPUtemp2b, CPUrunningSum1);				

				for(int j=0; j<rSumlen1; j++)	a_sq[j] = CPUrunningSum1[j];
				a_sq_length = rSumlen1;				
			}
						
			return a_sq_length;			
		}

		int CPUalpha2dEdge(gpudtVertex &pa, gpudtVertex &pb, REAL *a_sq)
		{
			// For comments, gpudtVertex pa will be referred as (x1,y1), pb as (x2,y2) .. and pf as (x6,y6)			
			INEXACT REAL bvirt;
			REAL avirt, bround, around;			
			INEXACT REAL abig;
			REAL c;
			REAL ahi, alo, bhi, blo;
			REAL err1, err2, err3;
			INEXACT REAL _i, _j, _k, _l;
			REAL _0, _1, _2;

			REAL ax1, ay1;
			REAL ax0, ay0;
			REAL ax_sq[6], ay_sq[6];

			REAL x1,x2,y1,y2;			
			x1 = pa.x; y1 = pa.y;
			x2 = pb.x; y2 = pb.y;	
			
			REAL tmp;

			// abracadabra
			// Compute ( (x1-x2)^2 + (y1-y2)^2 ) = p1			
			Two_Diff(x1,x2,ax1,ax0);																// Compute (x1-x2)			
			Two_Diff(y1,y2,ay1,ay0);																// Compute (y1-y2)		
			Two_Square(ax1, ax0, tmp, ax_sq[4], ax_sq[3], ax_sq[2], ax_sq[1], ax_sq[0]);		// Compute (x1-x2)^2			
			ax_sq[5] = tmp;
			Two_Square(ay1, ay0, tmp, ay_sq[4], ay_sq[3], ay_sq[2], ay_sq[1], ay_sq[0]);		// Compute (y1-y2)^2
			ay_sq[5] = tmp;
			int a_sq_length = cpu_fast_expansion_sum_zeroelim(6, ax_sq, 6, ay_sq, a_sq);				// Compute ((x1-x2)^2 + (y1-y2)^2)	[max 12 floats]	
			
			return a_sq_length;
		}		

		// To compare edge > Triangle
		int CPUcompareSlow(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf, int x)
		{						
			for(int i=0; i<200; i++)	CPUP1[i] = 0, CPUP2[i] = 0;			
			
			int P1_len = CPUalpha2dTriEdge(pa,pb,pd,pe,pf,CPUP1);
			int P2_len = CPUalpha2dTriEdge(pd,pe,pf,CPUP2);			

			for(int i=0; i<P2_len; i++) 
			{
				CPUP2[i] = -CPUP2[i];			
				CPUrunningSum1[i] = 0;
			}
			int ans_len = fast_expansion_sum_zeroelim(P1_len, CPUP1, P2_len, CPUP2, CPUrunningSum1);			

			if( CPUrunningSum1[ ans_len-1 ] == 0 )
				return 0;
			else if( CPUrunningSum1[ ans_len-1 ] > 0 )
				return 1;
			else
				return -1;

			//int p1Index = P1_len-1, p2Index = P2_len-1;			
			//while(p1Index >= 0 && p2Index >= 0)
			//{
			//	REAL circumRadius1 = CPUP1[ p1Index-- ];
			//	REAL circumRadius2 = CPUP2[ p2Index-- ];				

			//	if( circumRadius1 > circumRadius2 )					
			//		return 1;
			//	else if( circumRadius1 < circumRadius2 )					
			//		return -1;
			//}
			//
			//if(p1Index >= 0)				
			//	return 1;

			//else if(p2Index >= 0)
			//	return -1;

			//// Total for runningSum1, runningSum2, a_sq = [1,633,536 * 3] = 4,900,608 = 4,900,608 * 8 Bytes = 39 MB
			//return 0;
		}

		// To compare if Triangle > edge
		int CPUcompareSlow(gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf, gpudtVertex &pa, gpudtVertex &pb)
		{						
			for(int i=0; i<200; i++)	CPUP1[i] = 0, CPUP2[i] = 0;			

			int P1_len = CPUalpha2dTriEdge(pd,pe,pf,CPUP1);
			int P2_len = CPUalpha2dTriEdge(pa,pb,pd,pe,pf,CPUP2);			

			for(int i=0; i<P2_len; i++) 
			{
				CPUP2[i] = -CPUP2[i];			
				CPUrunningSum1[i] = 0;
			}
			int ans_len = fast_expansion_sum_zeroelim(P1_len, CPUP1, P2_len, CPUP2, CPUrunningSum1);			

			if( CPUrunningSum1[ ans_len-1 ] == 0 )
				return 0;
			else if( CPUrunningSum1[ ans_len-1 ] > 0 )
				return 1;
			else
				return -1;

			/*int p1Index = P1_len-1, p2Index = P2_len-1;			
			while(p1Index >= 0 && p2Index >= 0)
			{
				REAL circumRadius1 = CPUP1[ p1Index-- ];
				REAL circumRadius2 = CPUP2[ p2Index-- ];				

				if( circumRadius1 > circumRadius2 )					
					return 1;
				else if( circumRadius1 < circumRadius2 )					
					return -1;
			}
			
			if(p1Index >= 0)				
				return 1;

			else if(p2Index >= 0)
				return -1;

			return 0;*/
		}

		// To Compare if Triangle1 > Triangle  
		int CPUcompareSlow(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pc, gpudtVertex &pd, gpudtVertex &pe, gpudtVertex &pf)
		{				 			
			for(int i=0; i<200; i++)	CPUP1[i] = 0, CPUP2[i] = 0;

			int P1_len = CPUalpha2dtri(pa,pb,pc,pd,pe,pf,CPUP1);
			int P2_len = CPUalpha2dtri(pd,pe,pf,pa,pb,pc,CPUP2);			

			for(int i=0; i<P2_len; i++) 
			{
				CPUP2[i] = -CPUP2[i];			
				CPUrunningSum1[i] = 0;
			}
			int ans_len = fast_expansion_sum_zeroelim(P1_len, CPUP1, P2_len, CPUP2, CPUrunningSum1);			

			if( CPUrunningSum1[ ans_len-1 ] == 0 )
				return 0;
			else if( CPUrunningSum1[ ans_len-1 ] > 0 )
				return 1;
			else
				return -1;

			/*int p1Index = P1_len-1, p2Index = P2_len-1;			
			while(p1Index >= 0 && p2Index >= 0)
			{
				REAL circumRadius1 = CPUP1[ p1Index-- ];
				REAL circumRadius2 = CPUP2[ p2Index-- ];				
				
				if( circumRadius1 > circumRadius2 )					
					return 1;				
				else if( circumRadius1 < circumRadius2 )					
					return -1;
			}
			
			if(p1Index >= 0)	
				return 1;

			else if(p2Index >= 0)
				return -1;
			
			return 0;*/
		}

		// To Compare if Edge1 > Edge2
		int CPUcompareSlow(gpudtVertex &pa, gpudtVertex &pb, gpudtVertex &pc, gpudtVertex &pd)
		{			
			REAL P1[ 200 ], P2[ 200 ];	// 1633536
			for(int i=0; i<200; i++)	P1[i] = 0, P2[i] = 0;			

			int P1_len = CPUalpha2dEdge(pa,pb,P1);
			int P2_len = CPUalpha2dEdge(pc,pd,P2);			

			for(int i=0; i<P2_len; i++) 
			{
				CPUP2[i] = -CPUP2[i];			
				CPUrunningSum1[i] = 0;
			}
			int ans_len = fast_expansion_sum_zeroelim(P1_len, CPUP1, P2_len, CPUP2, CPUrunningSum1);			

			if( CPUrunningSum1[ ans_len-1 ] == 0 )
				return 0;
			else if( CPUrunningSum1[ ans_len-1 ] > 0 )
				return 1;
			else
				return -1;

			/*int p1Index = P1_len-1, p2Index = P2_len-1;			
			while(p1Index >= 0 && p2Index >= 0)
			{
				REAL circumRadius1 = P1[ p1Index-- ];
				REAL circumRadius2 = P2[ p2Index-- ];				

				if( circumRadius1 > circumRadius2 )		
					return 1;
				else if( circumRadius1 < circumRadius2 )		
					return -1;
			}

			if(p1Index >= 0)	
				return 1;

			else if(p2Index >= 0)
				return -1;

			return 0;*/
		}
	};

#ifdef __cplusplus
}
#endif