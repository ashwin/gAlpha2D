/*
Author: Cao Thanh Tung
Date: 15/03/2011

File Name: kernelVoronoi.cu

This file include all CUDA kernel code used in cudaVoronoi.cu

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

// Fill an array with some value
__global__ void kernelFillShort(short2* arr, short value, int log2Width) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    arr[(y << log2Width) + x] = make_short2(value, value); 
}

__global__ void kernelMapPointsToTexture(int n, REAL2 *points, short2 *texture, 
                                         int *pattern, int log2Width)
{
    int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;     

    if (id >= n)
        return ; 

    REAL2 p = points[id]; 

    // Map
    int x = int(floor(p.x)) + 1; 
    int y = int(floor(p.y)) + 1; 

    int pos = (y << log2Width) + x; 

    texture[pos] = make_short2(x, y); 
    pattern[pos] = id; 
}

#define SIGN(x)         ((x) > 0 ? 1 : ((x) < 0 ? -1 : 0))
#define EQUAL(u, v)     ((u.x == v.x) && (u.y == v.y))
#define NEQUAL(u, v)    ((u.x != v.x) || (u.y != v.y))
#define FETCH(x, y)		tex1Dfetch(texColor, ((y) << log2Width) + (x))

// Kernel to mark all islands and their neighbors
__global__ void kernelMarkIsland(short2 *islandMap, int *islandMark, int texSize,int log2Width)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx < 1 || ty < 1 || tx >= texSize - 1 || ty >= texSize - 1)
        return ; 
	int id = (ty << log2Width) + tx; 

	if(islandMap[id].x == texSize)	{
		
		for(int i = tx-1; i <= tx+1; i++)
			for(int j = ty-1; j <= ty+1; j++)
				if(i>0 && j>0 && i<texSize-1 && j<texSize-1)
				{
					islandMark[j*texSize+i] = 1;
				}
	}	
}

__global__ void kernelCollectIsland(short2 *islandMap, int *islandMark, int *compact, short2 *output, int texSize, int log2Width, int *cflag)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
	if (tx < 1 || ty < 1 || tx >= texSize - 1 || ty >= texSize - 1)
        return ;
	int id = (ty << log2Width) + tx; 
	
	if(islandMark[id]==1)
	{
		
		int address = compact[id];
		output[2*address].x = tx;
		output[2*address].y = ty;
		output[2*address+1].x = islandMap[id].x;
		output[2*address+1].y = islandMap[id].y;
		
	}

}
__global__ void kernelRecolorIsland(short2 *islandMap, short2 *output, int log2Width, int islands)
{
    int x = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	
	if(x >= islands)
		return;
	int tx = output[2*x].x;
	int ty = output[2*x].y;

	int address = (ty << log2Width) + tx; 
	islandMap[address] = output[2*x+1];
	
}
// Kernel to detect all islands
__global__ void kernelIslandDetection(short2 *output, int texSize, 
                                      int log2Width, int *cflag)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx < 1 || ty < 1 || tx >= texSize - 1 || ty >= texSize - 1)
        return ; 

    int id = (ty << log2Width) + tx; 

    //********************************************************************************
    // Check whether this pixel is connected to the isolated pixel
    //********************************************************************************
    short2 coords0, coords1, coords2;

    // Get The Center Pixel
    short2 centerPixel = tex1Dfetch(texColor, id);
	
	bool isIsolated = true; 

    // Test for Isolation
    int2 deltaPos = make_int2(SIGN(centerPixel.x - tx), SIGN(centerPixel.y - ty));
    // If one of the (maximum) three vertices has the same color with the center pixel, 
    // it is not isolated    

    if (deltaPos.x != 0) {
        coords0 = FETCH(tx + deltaPos.x, ty); 

        if (EQUAL(coords0, centerPixel))
            isIsolated = false;        
    }

    if (deltaPos.y != 0) {
        coords1 = FETCH(tx, ty + deltaPos.y); 

        if (EQUAL(coords1, centerPixel))
            isIsolated = false; 
       
        if (deltaPos.x != 0) {
            coords2 = FETCH(tx + deltaPos.x, ty + deltaPos.y); 

            if (EQUAL(coords2, centerPixel))
                isIsolated = false;            
        }
    }

    if (isIsolated && (deltaPos.x != 0 || deltaPos.y != 0)) {
		if(centerPixel.x != texSize)
		{
			*cflag = 1;			
		}
		centerPixel = make_short2(texSize,texSize);       			
    }

    output[id] = centerPixel; 
}

// Kernel to mark all real voronoi vertices
// Also, we propagate the indices of the sites to replace coordinate info
// in other pixels
__global__ void kernelFindRealVoronoiVertices(int *outputColor, short2 *outputPattern, 
                                              int texSize, int log2Width)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx < 1 || ty < 1 || tx >= texSize - 1 || ty >= texSize - 1)
        return ; 

    int id = (ty << log2Width) + tx; 

    short2 coords0, coords1, coords2, coords3; 
    int color_num = 4, pattern = 4;

    // center pixels
    coords0 = tex1Dfetch(texColor, id); 
    coords1 = tex1Dfetch(texColor, id + 1); 
    coords3 = tex1Dfetch(texColor, id + texSize); 
    coords2 = tex1Dfetch(texColor, id + texSize + 1); 

    // One neck property
    if (EQUAL(coords0, coords2) || EQUAL(coords1, coords3)) {
        pattern = -1; 
        tx = MAX_SHORT; 
    }
    else
    {
        if (EQUAL(coords0, coords1))
        {
            color_num--;
            pattern = 0;
        }
        if (EQUAL(coords1, coords2))
        {
            color_num--;
            pattern = 1;
        }
        if (EQUAL(coords2, coords3))
        {
            color_num--;
            pattern = 2;
        }
        if (EQUAL(coords3, coords0))
        {
            color_num--;
            pattern = 3;
        }
        if (color_num < 3) {
            pattern = -1; 
            tx = MAX_SHORT; 
        }
    }

    // Propagate the index of the sites
    outputColor[id] = tex1Dfetch(texInt, (coords0.y << log2Width) + coords0.x); 
    outputPattern[id] = make_short2(pattern, tx); 
}

// JFA 1D flooding
__global__ void kernelFlood1D(short2 *output, int texSize, int log2Width, int step)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x; 
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    int id = (ty << log2Width) + tx; 

    short2 input; 

    input = tex1Dfetch(texPattern, id); 

    if (input.y == MAX_SHORT && tx + step < texSize - 1)
        input = tex1Dfetch(texPattern, id + step); 

    output[id] = input; 
}

