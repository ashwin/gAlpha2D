/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifndef _SCAN_BEST_KERNEL_CU_
#define _SCAN_BEST_KERNEL_CU_

#include <stdio.h>
#include "common.h"

// Define this to more rigorously avoid bank conflicts, 
// even at the lower (root) levels of the tree
// Note that due to the higher addressing overhead, performance 
// is lower with ZERO_BANK_CONFLICTS enabled.  It is provided
// as an example.
//#define ZERO_BANK_CONFLICTS 

// 16 banks on G80
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

///////////////////////////////////////////////////////////////////////////////
// Work-efficient compute implementation of scan, one thread per 2 elements
// Work-efficient: O(log(n)) steps, and O(n) adds.
// Also shared storage efficient: Uses n + n/NUM_BANKS shared memory -- no ping-ponging
// Also avoids most bank conflicts using single-element offsets every NUM_BANKS elements.
//
// In addition, If ZERO_BANK_CONFLICTS is defined, uses 
//     n + n/NUM_BANKS + n/(NUM_BANKS*NUM_BANKS) 
// shared memory. If ZERO_BANK_CONFLICTS is defined, avoids ALL bank conflicts using 
// single-element offsets every NUM_BANKS elements, plus additional single-element offsets 
// after every NUM_BANKS^2 elements.
//
// Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums 
// and Their Applications", or Prins and Chatterjee PRAM course notes:
// http://www.cs.unc.edu/~prins/Classes/203/Handouts/pram.pdf
// 
// This work-efficient version is based on the algorithm presented in Guy Blelloch's
// excellent paper "Prefix sums and their applications".
// http://www-2.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html
//
// Pro: Work Efficient, very few bank conflicts (or zero if ZERO_BANK_CONFLICTS is defined)
// Con: More instructions to compute bank-conflict-free shared memory addressing,
// and slightly more shared memory storage used.
//

template <bool isNP2>
__device__ void loadSharedChunkFromMem(int *s_data,
                                       const int *g_idata, 
                                       int n, int baseIndex,
                                       int& ai, int& bi, 
                                       int& mem_ai, int& mem_bi, 
                                       int& bankOffsetA, int& bankOffsetB)
{
    int thid = threadIdx.x;
    mem_ai = baseIndex + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;

    // compute spacing to avoid bank conflicts
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Cache the computational window in shared memory
    // pad values beyond n with zeros
    s_data[ai + bankOffsetA] = g_idata[mem_ai]; 
    
    if (isNP2) // compile-time decision
    {
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
    }
    else
    {
        s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
    }
}

template <bool isNP2>
__device__ void storeSharedChunkToMem(int* g_odata, 
                                      const int* s_data,
                                      int n, 
                                      int ai, int bi, 
                                      int mem_ai, int mem_bi,
                                      int bankOffsetA, int bankOffsetB)
{
    __syncthreads();

    // write results to global memory
    g_odata[mem_ai] = s_data[ai + bankOffsetA]; 
    if (isNP2) // compile-time decision
    {
        if (bi < n)
            g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
    else
    {
        g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
}

template <bool storeSum>
__device__ void clearLastElement(int* s_data, 
                                 int *g_blockSums, 
                                 int blockIndex)
{
    if (threadIdx.x == 0)
    {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);
        
        if (storeSum) // compile-time decision
        {
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        }

        // zero the last element in the scan so it will propagate back to the front
        s_data[index] = 0;
    }
}



__device__ unsigned int buildSum(int *s_data)
{
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;
    
    // build the sum in place up the tree
    for (int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)      
        {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_data[bi] += s_data[ai];
        }

        stride *= 2;
    }

    return stride;
}

__device__ void scanRootToLeaves(int *s_data, unsigned int stride)
{
     unsigned int thid = threadIdx.x;

    // traverse down the tree building the scan in place
    for (int d = 1; d <= blockDim.x; d *= 2)
    {
        stride >>= 1;

        __syncthreads();

        if (thid < d)
        {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

template <bool storeSum>
__device__ void prescanBlock(int *data, int blockIndex, int *blockSums)
{
    int stride = buildSum(data);               // build the sum in place up the tree
    clearLastElement<storeSum>(data, blockSums, 
                               (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanRootToLeaves(data, stride);            // traverse down tree to build the scan 
}

template <bool storeSum, bool isNP2>
__global__ void prescan(int *g_odata, 
                        const int *g_idata, 
                        int *g_blockSums, 
                        int n, 
                        int blockIndex, 
                        int baseIndex)
{
    int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
    extern __shared__ int s_data[];

    // load data into shared memory
    loadSharedChunkFromMem<isNP2>(s_data, g_idata, n, 
                                  (baseIndex == 0) ? 
                                  __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex,
                                  ai, bi, mem_ai, mem_bi, 
                                  bankOffsetA, bankOffsetB); 
    // scan the data in each block
    prescanBlock<storeSum>(s_data, blockIndex, g_blockSums); 
    // write results to device memory
    storeSharedChunkToMem<isNP2>(g_odata, s_data, n, 
                                 ai, bi, mem_ai, mem_bi, 
                                 bankOffsetA, bankOffsetB);  
}


__global__ void uniformAdd(int *g_data, 
                           int *uniforms, 
                           int n, 
                           int blockOffset, 
                           int baseIndex)
{
    __shared__ int uni;
    if (threadIdx.x == 0)
        uni = uniforms[blockIdx.x + blockOffset];
    
    unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x; 

    __syncthreads();
    
    // note two adds per thread
    g_data[address]              += uni;
    g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}


__global__ void uniformAdd1(int *g_data, 
                           int *uniforms, 
                           int n, 
                           int blockOffset, 
                           int baseIndex)
{
    __shared__ int uni;
    if (threadIdx.x == 0)
        uni = uniforms[blockIdx.x + blockOffset];
    
    unsigned int address = __mul24(blockIdx.x, blockDim.x) + baseIndex + threadIdx.x; 

    __syncthreads();
    
    // note one add per thread
    g_data[address] += uni;
}


inline bool 
isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

inline int 
floorPow2(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((float)n);
#else
    // method 1
    // int nf = (int)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127); 
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

#define SCAN_BLOCK 256

int** g_scanBlockSums;
unsigned int g_numEltsAllocated = 0;
unsigned int g_numLevelsAllocated = 0;

void preallocBlockSums(unsigned int maxNumElements)
{
    g_numEltsAllocated = maxNumElements;

    unsigned int blockSize = SCAN_BLOCK; // max size of the thread blocks
    unsigned int numElts = maxNumElements;

    int level = 0;

    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (int**) malloc(level * sizeof(int*));
    g_numLevelsAllocated = level;
    
    numElts = maxNumElements;
    level = 0;
    
    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) 
        {
            cutilSafeCall( cudaMalloc((void**) &g_scanBlockSums[level++], numBlocks * sizeof(int)) );
        }
        numElts = numBlocks;
    } while (numElts > 1);
}

void deallocBlockSums()
{
    for (unsigned int i = 0; i < g_numLevelsAllocated; i++)
    {
        cutilSafeCall( cudaFree(g_scanBlockSums[i]) );
    }

    free((void**)g_scanBlockSums);

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}


void prescanArrayRecursive(int *outArray, 
                           const int *inArray, 
                           int numElements, 
                           int level)
{
    unsigned int blockSize = SCAN_BLOCK; // max size of the thread blocks
    unsigned int numBlocks = 
        max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = 
        numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);    
        
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = 
            sizeof(int) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = 
        sizeof(int) * (numEltsPerBlock + extraSpace);

#ifdef DEBUG
    if (numBlocks > 1)
    {
        assert(g_numEltsAllocated >= numElements);
    }
#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);

    // execute the scan
    if (numBlocks > 1)
    {
        prescan<true, false><<< grid, threads, sharedMemSize >>>(outArray, 
                                                                 inArray, 
                                                                 g_scanBlockSums[level],
                                                                 numThreads * 2, 0, 0);

        if (np2LastBlock)
        {
            prescan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>>
                (outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, 
                 numBlocks - 1, numElements - numEltsLastBlock);
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be sdded to each block to 
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive(g_scanBlockSums[level], 
                              g_scanBlockSums[level], 
                              numBlocks, 
                              level+1);

        uniformAdd<<< grid, threads >>>(outArray, 
                                        g_scanBlockSums[level], 
                                        numElements - numEltsLastBlock, 
                                        0, 0);
        
        if (np2LastBlock)
        {
            uniformAdd1<<< 1, numEltsLastBlock >>>(outArray, 
                                                     g_scanBlockSums[level], 
                                                     numEltsLastBlock, 
                                                     numBlocks - 1, 
                                                     numElements - numEltsLastBlock);
        }
    }
    else if (isPowerOfTwo(numElements))
    {
        prescan<false, false><<< grid, threads, sharedMemSize >>>(outArray, inArray,
                                                                  0, numThreads * 2, 0, 0);
    }
    else
    {
         prescan<false, true><<< grid, threads, sharedMemSize >>>(outArray, inArray, 
                                                                  0, numElements, 0, 0);
    }
}

void prescanArray(int *outArray, int *inArray, int numElements)
{
    prescanArrayRecursive(outArray, inArray, numElements, 0);
}

#endif // #ifndef _SCAN_BEST_KERNEL_CU_

