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

#include "Alpha_Shapes_2D.h"
#include <thrust/remove.h>
#include <memory.h>
#include <time.h>


#define MIN(a,b) (a<b?a:b)
#define MAX(a,b) (a>b?a:b)
#define e_vp(a) d->pDVec._array[a]

typedef thrust::device_vector<REAL>::iterator       dli;
typedef thrust::device_vector<int>::iterator		ili;
typedef thrust::device_vector<Alpha>::iterator      ali;
typedef thrust::device_vector<backRef>::iterator    bli;
typedef thrust::device_vector<int>::iterator        fli;
typedef thrust::tuple<ali,fli>						IteratorTuple;
typedef thrust::tuple<dli,bli>						SpectrumTuple;
typedef thrust::tuple<ili,ili>						IntegerTuple;
typedef thrust::zip_iterator<IteratorTuple>         ZipIterator;
typedef	thrust::zip_iterator<SpectrumTuple>         ZipSpectrum;
typedef thrust::zip_iterator<IntegerTuple>          ZipInteger;

Alpha_Shapes_2D::Alpha_Shapes_2D(PGPUDTPARAMS pIpt, PGPUDTOUTPUT pDelaunay) : data_initialized(true), pDT(pDelaunay), pInput(pIpt)
{
	cudaFuncSetCacheConfig(computeEdgeTriMap, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(computeTriEdgeCount, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(computeTriangleRo, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(computeEdgeRo, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(processAttached, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(processSpectrum, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(UpdateBackRefs, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(processEdges, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(edgeIntervals, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(computeVertexIntervals, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(computeMasterList, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(markSpectrum, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(sorT, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(pre_sort, cudaFuncCachePreferL1);
}

Alpha_Shapes_2D::Alpha_Shapes_2D() : pDT(NULL), pInput(NULL), data_initialized(false)
{
}

class MasterListDelete
{
public:

	__device__ bool operator() (const thrust::tuple<Alpha, int>& t) const
	{				
		return (thrust::get<1>(t) == 0);
	}
};

class FlagDelete
{
public:

	__device__ bool operator() (const thrust::tuple<int, int>& t) const
	{				
		return (thrust::get<0>(t) == 0);
	}
};

class SpectrumDelete
{
public:

	__device__ bool operator() (const thrust::tuple<REAL, backRef>& t) const
	{				
		return (thrust::get<1>(t).my_index == -1);
	}
};

void Alpha_Shapes_2D::clear_all()
{
	point_list.clear();
	edge_list.clear();
	tri_list.clear();

	master_list.clear();
	spectrum.clear();

	point_listD.clear();
	edge_listD.clear();
	tri_listD.clear();

	master_listD.clear();
	spectrumD.clear();
}

void Alpha_Shapes_2D::computeEdges()
{	
	int noOfBlocks = 0, noOfThreads = 0;
	noOfBlocks = (pDT->nTris+511)/512; 
	noOfThreads = (pDT->nTris>512)?512:pDT->nTris;

	DeviceData d(pDVec, triDVec, ptTriMap, point_listD, edge_listD, tri_listD, master_listD, spectrumD, br, tri_edge_countD, tri_edge_indicesD, ps, num_edges);
	
	computeTriEdgeCount <<< noOfBlocks, noOfThreads >>> (d);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "Error computeTriEdgeCount!\n" );

	thrust::exclusive_scan(tri_edge_countD.begin(), tri_edge_countD.end(), tri_edge_indicesD.begin());	
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "Error Exclusive scan for edges!\n" );

	num_edges = tri_edge_indicesD[pDT->nTris-1] + tri_edge_countD[pDT->nTris-1];
	printf("\nNumber of edges: %d\n", num_edges);

	edge_tri_map.resize(num_edges);

	computeEdgeTriMap <<< noOfBlocks, noOfThreads >>> (d, edge_tri_map);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "Error computeTriEdgeCount!\n" );
}

bool Alpha_Shapes_2D::init_alpha_shapes()
{
	if(!data_initialized)
		return false;

	clear_all();

	/* Initialize data structures in host : Start */	
	alpha = 0;
	/* Initialize data structures in host : End */

	/* Initialize data structures in GPU : Start	*/
	vector<gpudtVertex> pts(pInput->points, pInput->points + pInput->nPoints);
	pDVec = pts;

	vector<gpudtTriangle> tris(pDT->triangles, pDT->triangles + pDT->nTris);
	triDVec = tris;

	/* Compute point to triangle map : Start */
	ptTriMap.resize(pInput->nPoints, -1);
	
	int size = pDT->nTris;
	int noOfBlocks = (size+511)/512; 
	int noOfThreads = (size>512)?512:size;

	// Todo: Change ptTriMap computation to do continuous writes
	computePtTriMap <<< noOfBlocks, noOfThreads >>> (pDVec, triDVec, ptTriMap);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "\nError computing point-triangle map!\n" );
	/* Compute point to triangle map End */
		
	tri_listD.resize(pDT->nTris);
	tri_edge_countD.resize(pDT->nTris);
	tri_edge_indicesD.resize(pDT->nTris);

	// Allocate memory for predicates
	cudaMalloc(&ps,sizeof(Predicates));
	initPredicate <<< 1,1 >>> (ps);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "\nError initializing predicates in Alpha Shapes!\n" );
	/* Initialize data structures in GPU : End */

	/* Compute Edge Data structure */	
	computeEdges();

	return true;
}

void Alpha_Shapes_2D::computeAlphaShapes()
{
	clock_t tv1[2];
	tv1[0] = clock();
	init_alpha_shapes();
	tv1[1] = clock();
	printf("Alpha Shapes init: %.4fs\n", (tv1[1]-tv1[0])/(REAL)CLOCKS_PER_SEC);	
		
	tv1[0] = clock();
	compute_ro();	
	tv1[1] = clock();
	printf("Alpha Shapes birth time computation: %.4fs\n", (tv1[1]-tv1[0])/(REAL)CLOCKS_PER_SEC);	

	tv1[0] = clock();
	sort_spectrum();
	tv1[1] = clock();
	printf("Alpha Shapes sorting spectrum: %.4fs\n", (tv1[1]-tv1[0])/(REAL)CLOCKS_PER_SEC);	

	tv1[0] = clock();
	compute_intervals();
	tv1[1] = clock();
	printf("Alpha Shapes computing face intervals: %.4fs\n", (tv1[1]-tv1[0])/(REAL)CLOCKS_PER_SEC);	

	tv1[0] = clock();
	compute_master_list();
	tv1[1] = clock();
	printf("Alpha Shapes computing and sorting master list: %.4fs\n", (tv1[1]-tv1[0])/(REAL)CLOCKS_PER_SEC);	
}

void Alpha_Shapes_2D::compute_ro()
{
	int spectrum_size = num_edges + pDT->nTris;	
	br.resize(spectrum_size + 1);
	spectrumD.resize(spectrum_size + 1);
	edge_listD.resize(num_edges);

	DeviceData d(pDVec, triDVec, ptTriMap, point_listD, edge_listD, tri_listD, master_listD, spectrumD, br, tri_edge_countD, tri_edge_indicesD, ps, num_edges);

	int size = pDT->nTris;
	int noOfBlocks = (size+127)/128; 
	int noOfThreads = (size>128)?128:size;

	//printf("blocks = %d, threads = %d\n", noOfBlocks, noOfThreads);

	computeTriangleRo <<< noOfBlocks, noOfThreads >>> (d);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "\nError computing ro for triangle!\n" );	

	size = num_edges;
	noOfBlocks = (size+127)/128; 
	noOfThreads = (size>128)?128:size;

	//printf("blocks = %d, threads = %d\n", noOfBlocks, noOfThreads);

	computeEdgeRo <<< noOfBlocks, noOfThreads >>> (d, edge_tri_map);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "\nError computing ro for edge!\n" );	

	edge_tri_map.clear();	

	/*REAL valhaha = spectrumD[ spectrumD.size()-1 ];
	printf( "\nBefore removing attached edges = %d, %.10lf\n", spectrumD.size(), valhaha);*/	

	processAttached <<< noOfBlocks, noOfThreads >>> (d);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "\nError processing attached edges!\n" );

	spectrumD.erase(thrust::remove_if(spectrumD.begin(), spectrumD.end(), LessThanZero()), spectrumD.end());
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "\nError erasing attached edge entries in spectrum!\n" );	

	br.erase(thrust::remove_if(br.begin(), br.end(), BackRefRemove()), br.end());
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "\nError erasing attached edge entries in back reference!\n" );		

	/*valhaha = spectrumD[ spectrumD.size()-1 ];
	printf( "\nAfterremoving attached edges = %d, %.10lf\n", spectrumD.size(), valhaha);*/		
}

void PrintMemInfo()
{	
	REAL free_m,total_m;
	size_t tot, free;

	cudaSetDevice(0);
	cudaMemGetInfo(&free, &tot);

	free_m  =(size_t)free/1048576.0 ;
	total_m =(size_t)tot/1048576.0;
	printf("\nFree: %d MB, Total: %d MB\n", (int)free_m, (int)total_m);	
}

__global__ void setit(int *isSortD)
{
	*isSortD = 1;
}

void Alpha_Shapes_2D::sort_spectrum()
{	
	int size;
	int noOfBlocks; 
	int noOfThreads;
	int *isSortD, isSortH = 1, round = 1;	

	tri_edge_countD.clear();
	
	thrust::sort_by_key(spectrumD.begin()+1, spectrumD.end(), br.begin()+1);	

	/*spectrum = spectrumD;
	for(int i=0; i<spectrum.size(); i++)
	{
		if( spectrum[i] > 0.03725 && spectrum[i] < 0.0372501)
			printf("spectrum[%d] = %.30lf\n", i, spectrum[i]);		
	}*/

	// Sorting Phase 2
	DeviceData d(pDVec, triDVec, ptTriMap, point_listD, edge_listD, tri_listD, master_listD, spectrumD, br, tri_edge_countD, tri_edge_indicesD, ps, num_edges);
	DeviceData *dd;
	cudaMalloc(&dd, sizeof(d));
	cudaMalloc(&isSortD, sizeof(int));
	cudaMemcpy(dd, &d, sizeof(d), cudaMemcpyHostToDevice);		
	cudaMemcpy(isSortD, &isSortH, sizeof(int), cudaMemcpyHostToDevice);

	int specSize = spectrumD.size();
	//int gpu_marker_size = ceil((double)specSize/32.0);
	flag.resize(specSize, 0);	

	size = specSize;
	noOfBlocks = (size+255)/256; 
	noOfThreads = (size>256)?256:size;

	/*ShortDVec markerOrder(size, -1);
	IntDVec markerTid(size);
	
	thrust::sequence(markerTid.begin(), markerTid.end());
	pre_mark <<< noOfBlocks, noOfThreads >>> (dd, markerOrder);
	sort_by_key(markerOrder.begin(), markerOrder.end(), markerTid.begin());*/

	BoolDVec gpu_marker(specSize, false);
	markSpectrum <<< noOfBlocks, noOfThreads >>> (dd, gpu_marker/*, markerTid*/);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "Error marking spectrum for exact check!\n" );

	/*markerOrder.clear();
	markerTid.clear();*/
	
	size = specSize;		
	size = MIN(size, 70000);	
	noOfBlocks = (size+127)/128; 
	noOfThreads = (size>128)?128:size;

	int noOfChunks = ceil((double)specSize/(2.0*size));
	DoubleDVec P1, P2, temp2a, temp2b, runningSum1, runningSum2;
	BoolDVec isSortedD(noOfChunks, false);
	BoolDVec previous(noOfChunks, true);
	ShortDVec ordering(size, -1);
	IntDVec threadTid(size, 0);

	P1.resize(200*(size+1));
	P2.resize(200*(size+1));
	temp2a.resize(200*(size+1));
	temp2b.resize(200*(size+1));
	runningSum1.resize(200*(size+1));	
	runningSum2.resize(200*(size+1));
	PrecisionData pd(P1, P2, temp2a, temp2b, runningSum1, runningSum2);	
	
	//PrintMemInfo();
	REAL tv1[2];				
	do
	{
		//printf("Round %d started\n", round);
		tv1[0] = clock();
		isSortH = 1;
		cudaMemcpy(isSortD, &isSortH, sizeof(int), cudaMemcpyHostToDevice);
		cudaThreadSynchronize();
		if ( cudaSuccess != cudaGetLastError() )
				printf( "Error memcpy7!\n" );		
		
		int chunk = 0, chunkNo = 0;
		while( chunk < specSize ) 
		{
			if( previous[chunkNo] != isSortedD[chunkNo] )
			{	
				if( round < 4 )
				{
					thrust::fill(ordering.begin(), ordering.end(), -1);
					thrust::sequence(threadTid.begin(), threadTid.end());
					pre_sort <<< noOfBlocks, noOfThreads >>> (dd, 1, ordering, threadTid, gpu_marker, size, chunk);
					sort_by_key(ordering.begin(), ordering.end(), threadTid.begin());								
				}

				sorT <<< noOfBlocks, noOfThreads >>> (dd, 1, isSortD, threadTid, flag, gpu_marker, isSortedD, chunkNo, pd, size, chunk, round);		
				cudaError_t err = cudaThreadSynchronize();
				if ( cudaSuccess != cudaGetLastError() )
				{
						printf("cutilCheckMsg cudaThreadSynchronize error: %s.\n", cudaGetErrorString( err) );
						break;
				}				
			}
			chunk += (2*size);
			chunkNo++;
		}
		if( round != 1 )
			previous = isSortedD;		
		thrust::fill(isSortedD.begin(), isSortedD.end(), false);

		chunk = 0;
		chunkNo = 0;		
		while( chunk < specSize ) 
		{	
			if( previous[chunkNo] != isSortedD[chunkNo] )
			{
				if(round < 4)
				{
					thrust::fill(ordering.begin(), ordering.end(), -1);
					thrust::sequence(threadTid.begin(), threadTid.end());
					pre_sort <<< noOfBlocks, noOfThreads >>> (dd, 1, ordering, threadTid, gpu_marker, size, chunk);
					sort_by_key(ordering.begin(), ordering.end(), threadTid.begin());
				}

				//tv1[0] = clock();
				sorT <<< noOfBlocks, noOfThreads >>> (dd, 2, isSortD, threadTid, flag, gpu_marker, isSortedD, chunkNo, pd, size, chunk, round);		
				cudaError_t err = cudaThreadSynchronize();
				if ( cudaSuccess != cudaGetLastError() )
				{
						printf("cutilCheckMsg cudaThreadSynchronize error: %s.\n", cudaGetErrorString( err) );
						break;
				}
				//tv1[1] = clock();				
				/*if( round < 3 )
					printf("GPU Phase 2.2: %.4fs\n", (tv1[1]-tv1[0])/(REAL)CLOCKS_PER_SEC);	*/			
			}
			chunk += (2*size);
			chunkNo++;
		}
		previous = isSortedD;
		thrust::fill(isSortedD.begin(), isSortedD.end(), false);

		cudaMemcpy(&isSortH, isSortD, sizeof(int), cudaMemcpyDeviceToHost);
		
		//tv1[1] = clock();
		//printf("Phase 2 Sorting Rounnd %d: %.4fs\n", round++, (tv1[1]-tv1[0])/(REAL)CLOCKS_PER_SEC);

		if( round == 3 )
			thrust::sequence(threadTid.begin(), threadTid.end());
		round++;
		//printf("Round %d ended\n\n", round++);		
	}while(isSortH == 0);			

	cudaFree(dd);
	cudaFree(isSortD);

	P1.clear();
	P2.clear();
	temp2a.clear();
	temp2b.clear();
	runningSum1.clear();
	runningSum2.clear();
	previous.clear();
	isSortedD.clear();

	//PrintMemInfo();
	// Sorting Phase 3
	CPU_insertion_sort(specSize);
	flag.clear();
	gpu_marker.clear();

	size = specSize;
	noOfBlocks = (size+511)/512; 
	noOfThreads = (size>512)?512:size;	

	processSpectrum <<< noOfBlocks, noOfThreads >>> (d);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "Error processSpectrum kernel in sort_spectrum!\n" );

	DoubleDVec tempSpectrumD(spectrumD);
	BackRefDVec tempbrD(br);

	ZipSpectrum begin( thrust::make_tuple( tempSpectrumD.begin(), tempbrD.begin() ) );
    ZipSpectrum end  ( thrust::make_tuple( tempSpectrumD.end(),   tempbrD.end() ) );
	ZipSpectrum output_end = thrust::remove_if(begin, end, SpectrumDelete());
	int offset = output_end - begin;

	tempSpectrumD.erase(tempSpectrumD.begin() + offset, tempSpectrumD.end());
	tempbrD.erase(tempbrD.begin() + offset, tempbrD.end());

	size = tempbrD.size();
	noOfBlocks = (size+511)/512; 
	noOfThreads = (size>512)?512:size;

	UpdateBackRefs <<< noOfBlocks, noOfThreads >>> (d, tempbrD);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "Error processSpectrum kernel in sort_spectrum!\n" );

	printf("Final spectrum size %d, rounds = %d\n", tempSpectrumD.size(), round);	
	spectrumD = tempSpectrumD;
	br = tempbrD;
	tempSpectrumD.clear();
	tempbrD.clear();

	size = num_edges;
	noOfBlocks = (size+511)/512; 
	noOfThreads = (size>512)?512:size;

	/*setit <<< 1,1 >>> (isSortD);
	fastCheck <<< noOfBlocks, noOfThreads >>> (dd, isSortD);
	cudaMemcpy(&isSortH, isSortD, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Value: %d\n", isSortH);*/
}

void Alpha_Shapes_2D::CPU_insertion_sort(int specSize)
{
	IntDVec spectrumIndices(specSize);
	thrust::sequence(spectrumIndices.begin(), spectrumIndices.end());	
	
	ZipInteger begin( thrust::make_tuple( flag.begin(), spectrumIndices.begin() ) );
    ZipInteger end  ( thrust::make_tuple( flag.end(),   spectrumIndices.end() ) );
	ZipInteger output_end = thrust::remove_if(begin, end, FlagDelete());
	int offset = output_end - begin;

	flag.clear();
	spectrumIndices.erase(spectrumIndices.begin() + offset, spectrumIndices.end());	

	if( spectrumIndices.size() > 0 )
	{
		printf("Simplices for exact check = %d\n", spectrumIndices.size());
		IntHVec sIH = spectrumIndices;	
		BackRefHVec brH = br;
		edge_list = edge_listD;
		tri_list = tri_listD;
		spectrum = spectrumD;

		Predicates CPUps;
		CPUps.exactinit();
		CPUps.CPUtemp2a   = new REAL[136130];
		CPUps.CPUtemp2b   = new REAL[136130];
		CPUps.CPUtemp2    = new REAL[272260];
		CPUps.CPUrunningSum1 = new REAL[1633536];
		CPUps.CPUrunningSum2 = new REAL[1633536];
		CPUps.CPUP1		  = new REAL[1633536];
		CPUps.CPUP2		  = new REAL[1633536];

		unsigned int i, counter = 0;	
		int j;
		CPUData d(edge_list, tri_list, spectrum, brH, pDT, pInput, &CPUps);
				
		for( i = 0; i < sIH.size(); i++)		
		{
			int nextI = i; //sIH[i];
			REAL nextS = d.spectrum._array[nextI];
			backRef brI = d.br._array[nextI];
			int res;

			for( j = nextI-1; j >= 0; j--)
			{												
				res = exactCompare(d, d.br._array[j], d.br._array[nextI]);
				if(res < 0)
					break;
				else if(res == 0)
				{				
					if( d.br._array[nextI].my_index != -1)
					{
						counter++;						
					}
					break;
				}

				d.spectrum._array[j+1] = d.spectrum._array[j];
				d.br._array[j+1] = d.br._array[j];
			}
			
			if( j != nextI-1 )
			{
				d.spectrum._array[j+1] = nextS;
				d.br._array[j+1] = brI;
			}
			if( res == 0 )
				d.br._array[j+1].my_index = -1;
		} 

		br = brH;
		printf("\nNumber of Equal to : %d\n", counter);

		delete[] CPUps.CPUtemp2a;
		delete[] CPUps.CPUtemp2b;
		delete[] CPUps.CPUtemp2;
		delete[] CPUps.CPUrunningSum1;
		delete[] CPUps.CPUrunningSum2;
		delete[] CPUps.CPUP1;
		delete[] CPUps.CPUP2;
		
		sIH.clear();
		brH.clear();
	}

	spectrumIndices.clear();
}

void Alpha_Shapes_2D::compute_intervals() 
{
	int size = num_edges;
	int noOfBlocks = (size+127)/128; 
	int noOfThreads = (size>128)?128:size;

	point_listD.resize(pInput->nPoints);
	DeviceData d(pDVec, triDVec, ptTriMap, point_listD, edge_listD, tri_listD, master_listD, spectrumD, br, tri_edge_countD, tri_edge_indicesD, ps, num_edges);

	//PrintMemInfo();

	edgeIntervals <<< noOfBlocks, noOfThreads >>> (d);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "\nError computing intervals for edges!\n" );

	size = pInput->nPoints;
	noOfBlocks = (size+127)/128; 
	noOfThreads = (size>128)?128:size;

	computeVertexIntervals <<< noOfBlocks, noOfThreads >>> (d);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "\nError computing intervals for vertices!\n" );

	br.clear();
	tri_edge_countD.clear();
}

void Alpha_Shapes_2D::compute_master_list()
{
	int size = 2*pInput->nPoints + 3*num_edges + pDT->nTris;	
	int noOfBlocks = (size+511)/512; 
	int noOfThreads = (size>512)?512:size;
	int offset = 0;

	master_listD.resize(size);
	IntDVec sRanks(size);	

	DeviceData d(pDVec, triDVec, ptTriMap, point_listD, edge_listD, tri_listD, master_listD, spectrumD, br, tri_edge_countD, tri_edge_indicesD, ps, num_edges);

	computeMasterList <<< noOfBlocks, noOfThreads >>> (d, sRanks);
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "\nError computing ro for triangle!\n" );	    	

	thrust::sort_by_key(sRanks.begin(), sRanks.end(), master_listD.begin());
	cudaThreadSynchronize();
	if ( cudaSuccess != cudaGetLastError() )
			printf( "Error computeMasterList!\n" );	

	ZipIterator begin( thrust::make_tuple( master_listD.begin(), sRanks.begin() ) );
    ZipIterator end  ( thrust::make_tuple( master_listD.end(),   sRanks.end() ) );
	ZipIterator output_end = thrust::remove_if(begin, end, MasterListDelete());
	offset = output_end - begin;

	master_listD.erase(master_listD.begin() + offset, master_listD.end());
	sRanks.erase(sRanks.begin() + offset, sRanks.end());

	IntDVec sPointers(master_listD.size());
	thrust::sequence(sPointers.begin(), sPointers.end());

	thrust::pair<IntDIter,IntDIter> ends = thrust::unique_by_key(sRanks.begin(), sRanks.end(), sPointers.begin());
	sPointers.erase(ends.second, sPointers.end());
	
	master_list = master_listD;
	point_list = point_listD;
	edge_list = edge_listD;
	tri_list = tri_listD;
	spectrum = spectrumD;
	spec_pointers = sPointers;

	spec_pointers.insert(spec_pointers.begin(),-1);

	sRanks.clear();
	master_listD.clear();
	point_listD.clear();
	edge_listD.clear();
	tri_listD.clear();
	spectrumD.clear();
	sPointers.clear();

	cout << "Spec pointer size = " << spec_pointers.size() << " Spectrum size = " << spectrum.size() << endl;

	//for(int tid = 0; tid < master_list.size(); tid++)
	//{
	//	Alpha master_entry = master_list[tid];	

	//	/*if( master_entry.st == VERTEX )
	//	{		
	//		if(master_entry.at == nu_min)
	//			printf("VERTEX: %d, nu_min = %d\n", master_entry.simplex_index, point_list[master_entry.simplex_index].nu_min_rank );

	//		if(master_entry.at == nu_max)
	//			printf("VERTEX: %d, nu_max = %d\n", master_entry.simplex_index, point_list[master_entry.simplex_index].nu_max_rank );
	//	}*/

	//	/*else */if( master_entry.st == EDGE)
	//	{			
	//		if(master_entry.at == ro)
	//			printf("EDGE: %d, ro = %d\n", master_entry.simplex_index, edge_list[master_entry.simplex_index].ro_rank );

	//		if(master_entry.at == nu_min)
	//			printf("EDGE: %d, nu_min = %d\n", master_entry.simplex_index, edge_list[master_entry.simplex_index].nu_min_rank );

	//		if(master_entry.at == nu_max)
	//			printf("EDGE: %d, nu_max = %d\n", master_entry.simplex_index, edge_list[master_entry.simplex_index].nu_max_rank );		
	//	}	

	//	/*else if( master_entry.st == TRIANGLE)
	//	{				
	//		if(master_entry.at == ro)
	//			printf("TRIANGLE: %d, ro = %d\n", master_entry.simplex_index, tri_list[master_entry.simplex_index].ro_rank );	
	//	}*/
	//}
}

MasterList_iterator Alpha_Shapes_2D::Master_begin()
{
	return master_list.begin();
}

MasterList_iterator Alpha_Shapes_2D::Master_end()
{
	return master_list.end();
}

void Alpha_Shapes_2D::setAlpha(REAL val)
{
	alpha = val;
}

REAL Alpha_Shapes_2D::getAlpha()
{
	return alpha;
}

void Alpha_Shapes_2D::dumpValues(string fname)
{	
	string outputFileName = "C:\\TestData\\TestsOut\\" + fname + "out.txt";			
	ofstream fout(outputFileName.c_str());
	
	/*fout << "VERTICES\n";
	for(unsigned int i=0; i<point_list.size(); i++)
		fout << pInput->points[i].x << " " << pInput->points[i].y << " " << spectrum[point_list[i].nu_min_rank] << " " << spectrum[point_list[i].nu_max_rank] << endl;

	fout << "EDGES\n";
	for(EdgeIntervalHVec::iterator edge_it = edge_list.begin(); edge_it != edge_list.end(); edge_it++)
	{
		int p1Index = pDT->triangles[ (*edge_it).tIndex ].vtx[ ((*edge_it).vOpp+1)%3 ];
		int p2Index = pDT->triangles[ (*edge_it).tIndex ].vtx[ ((*edge_it).vOpp+2)%3 ];

		fout << pInput->points[p1Index].x << " " << pInput->points[p1Index].y << " ";
		fout << pInput->points[p2Index].x << " " << pInput->points[p2Index].y << " ";
		fout << spectrum[ (*edge_it).ro_rank ] << " " << spectrum[ (*edge_it).nu_min_rank ];
		fout << " " << spectrum[ (*edge_it).nu_max_rank] << endl;
	}

	fout << "TRIANGLES" << endl;
	for(unsigned int i=0; i<tri_list.size(); i++)
	{
		fout << pInput->points[ pDT->triangles[i].vtx[0] ].x << " " << pInput->points[ pDT->triangles[i].vtx[0] ].y << " ";
		fout << pInput->points[ pDT->triangles[i].vtx[1] ].x << " " << pInput->points[ pDT->triangles[i].vtx[1] ].y << " ";
		fout << pInput->points[ pDT->triangles[i].vtx[2] ].x << " " << pInput->points[ pDT->triangles[i].vtx[2] ].y << " ";
		fout << spectrum[ tri_list[i].ro_rank ] << endl;
	}*/

	//fout << "SPECTRUM" << endl;
	
	int size = spectrum.size();
	fout << size << endl;

	/*for(unsigned int i=0; i<messages.size(); i++)
		fout << messages[i];*/
	/*for(unsigned int i=0; i<spectrum.size(); i++)
	{
		fout << spectrum[i] << endl;
	}*/
	
	messages.clear();
	fout.close();
}
