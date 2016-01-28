/*
Author: Cao Thanh Tung
Date: 15/03/2011

File Name: cuda.h

Header file with declarations of all CUDA steps, including some configuration.

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

#ifndef __CUDA_H__
#define __CUDA_H__

#include "../gpudt.h"

#ifdef SINGLE_PRECISION
    #define REAL2           float2
    #define MAKE_REAL2      make_float2
#else
    #define REAL2           double2
    #define MAKE_REAL2      make_double2
#endif

#define BYTE            char

#define STRIPE          64

extern "C" void cudaAllocation();
extern "C" void cudaDeallocation(); 
extern "C" void cudaInitialize();
extern "C" void cudaFinalize();
extern "C" void cudaDiscreteVoronoiDiagram();
extern "C" void cudaReconstruction(); 
extern "C" void cudaShifting(); 
extern "C" void cudaMissing(); 
extern "C" void cudaConstraint(); 
extern "C" void cudaFixBoundary();
extern "C" int cudaFlipping(int **suspective);
extern "C" void cudaExactInit(); 

#endif
