/*
Author: Cao Thanh Tung
Date: 26/01/2010

File Name: pba2DHost.cu

===============================================================================

Copyright (c) 2010, School of Computing, National University of Singapore. 
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/pba.html

If you use PBA and you like it or have comments on its usefulness etc., we 
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

#include <device_functions.h>

#include "cuda.h"

// Parameters for CUDA kernel executions
#define BLOCKX		16
#define BLOCKY		16
#define BLOCKSIZE	64
#define TILE_DIM	32
#define BLOCK_ROWS	8

extern "C" void pba2DCompute(int phase1Band, int phase2Band, int phase3Band); 

/****** Global Variables *******/
extern short2 **pingpongColor;       // Two textures used to compute 2D Voronoi Diagram
extern int texSize;					 // Texture size, shared with cudaVoronoi

texture<short2> pbaTexColor; 
texture<short2> pbaTexLinks; 

/********* Kernels ********/
#include "pba2DKernel.h"

// In-place transpose a squared texture. 
// Block orders are modified to optimize memory access. 
// Point coordinates are also swapped. 
void pba2DTranspose(short2 *texture)
{
    dim3 block(TILE_DIM, BLOCK_ROWS); 
    dim3 grid(texSize / TILE_DIM, texSize / TILE_DIM); 

    cudaBindTexture(0, pbaTexColor, texture); 
    kernelTranspose<<< grid, block >>>(texture, texSize); 
    cudaUnbindTexture(pbaTexColor); 
}

// Phase 1 of PBA. m1 must divides texture size
void pba2DPhase1(int m1) 
{
    dim3 block = dim3(BLOCKSIZE);   
    dim3 grid = dim3(texSize / block.x, m1); 

    // Flood vertically in their own bands
    cudaBindTexture(0, pbaTexColor, pingpongColor[0]); 
    kernelFloodDown<<< grid, block >>>(pingpongColor[1], texSize, texSize / m1); 
    cudaUnbindTexture(pbaTexColor); 

    cudaBindTexture(0, pbaTexColor, pingpongColor[1]); 
    kernelFloodUp<<< grid, block >>>(pingpongColor[1], texSize, texSize / m1); 

    // Passing information between bands
    grid = dim3(texSize / block.x, m1); 
    kernelPropagateInterband<<< grid, block >>>(pingpongColor[0], texSize, texSize / m1); 

    cudaBindTexture(0, pbaTexLinks, pingpongColor[0]); 
    kernelUpdateVertical<<< grid, block >>>(pingpongColor[1], texSize, m1, texSize / m1); 
    cudaUnbindTexture(pbaTexLinks); 
    cudaUnbindTexture(pbaTexColor); 
}

// Phase 2 of PBA. m2 must divides texture size
void pba2DPhase2(int m2) 
{
    // Compute proximate points locally in each band
    dim3 block = dim3(BLOCKSIZE);   
    dim3 grid = dim3(texSize / block.x, m2); 
    cudaBindTexture(0, pbaTexColor, pingpongColor[1]); 
    kernelProximatePoints<<< grid, block >>>(pingpongColor[0], texSize, texSize / m2); 

    cudaBindTexture(0, pbaTexLinks, pingpongColor[0]); 
    kernelCreateForwardPointers<<< grid, block >>>(pingpongColor[0], texSize, texSize / m2); 

    // Repeatly merging two bands into one
    for (int noBand = m2; noBand > 1; noBand /= 2) {
        grid = dim3(texSize / block.x, noBand / 2); 
        kernelMergeBands<<< grid, block >>>(pingpongColor[0], texSize, texSize / noBand); 
    }

    // Replace the forward link with the X coordinate of the seed to remove
    // the need of looking at the other texture. We need it for coloring.
    grid = dim3(texSize / block.x, texSize); 
    kernelDoubleToSingleList<<< grid, block >>>(pingpongColor[0], texSize); 
    cudaUnbindTexture(pbaTexLinks); 
    cudaUnbindTexture(pbaTexColor); 
}

// Phase 3 of PBA. m3 must divides texture size
void pba2DPhase3(int m3) 
{
    dim3 block = dim3(BLOCKSIZE / m3, m3); 
    dim3 grid = dim3(texSize / block.x); 
    cudaBindTexture(0, pbaTexColor, pingpongColor[0]); 
    kernelColor<<< grid, block >>>(pingpongColor[1], texSize); 
    cudaUnbindTexture(pbaTexColor); 
}

void pba2DCompute(int phase1Band, int phase2Band, int phase3Band)
{
    // Vertical sweep
    pba2DPhase1(phase1Band); 

    pba2DTranspose(pingpongColor[1]); 

    // Horizontal coloring
    pba2DPhase2(phase2Band); 

    // Color the rows. 
    pba2DPhase3(phase3Band); 

    pba2DTranspose(pingpongColor[1]); 
}
