/*
Author: Cao Thanh Tung
Date: 15/03/2011

File Name: common.h

Some common CUDA functions

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

#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>

// Define CUDA_CHECK_ERROR to enable CUDA error checking feature
#define CUDA_CHECK_ERROR

#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
#define cutilCheckError()            __cudaCheckError   (__FILE__, __LINE__)



inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
    do {
        if( cudaSuccess != err) {
            fprintf(stderr, "cudaSafeCall() Runtime API error in file <%s>, line %i : %s.\n",
                    file, line, cudaGetErrorString( err) );
            exit(-1);
        }
    } while (0);
#endif
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
    do {
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) {
            fprintf(stderr, "cutilCheckMsg() CUTIL CUDA error in file <%s>, line %i : %s.\n",
                    file, line, cudaGetErrorString( err) );
            exit(-1);
        }
		// More careful checking will be to perform a cudaThreadSynchronize(). 
		// This however will reduce the performance. 
        //err = cudaThreadSynchronize();
        //if( cudaSuccess != err) {
        //    fprintf(stderr, "cutilCheckMsg cudaThreadSynchronize error in file <%s>, line %i : %s.\n",
        //            file, line, cudaGetErrorString( err) );
        //    exit(-1);
        //}
    } while (0);
#endif
}

#endif