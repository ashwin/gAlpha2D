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

#include "headers.h"
#include "../gpudt.h"

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\sort.h>
#include <thrust\copy.h>
#include <thrust\scan.h>
#include <thrust\transform.h>
#include <thrust\remove.h>
#include <thrust/unique.h>

#include "error.h"

typedef thrust::host_vector< gpudtVertex >				PointHVec;
typedef thrust::device_vector< gpudtVertex >			PointDVec;

typedef thrust::host_vector< point_interval >			PointIntervalHVec;
typedef thrust::device_vector< point_interval >			PointIntervalDVec;

typedef thrust::host_vector< triangle_interval >		TriangleIntervalHVec;
typedef thrust::device_vector< triangle_interval >		TriangleIntervalDVec;

typedef thrust::host_vector< edge_interval >			EdgeIntervalHVec;
typedef thrust::device_vector< edge_interval >			EdgeIntervalDVec;

typedef thrust::host_vector< REAL >						DoubleHVec;
typedef thrust::device_vector< REAL >					DoubleDVec;

typedef thrust::host_vector< Alpha >					AlphaHVec;
typedef thrust::device_vector< Alpha >					AlphaDVec;

typedef thrust::host_vector< gpudtTriangle >			TriangleHVec;
typedef thrust::device_vector< gpudtTriangle >			TriangleDVec;

typedef thrust::host_vector<int>						IntHVec;
typedef thrust::device_vector<int>						IntDVec;

typedef thrust::host_vector<short>						ShortHVec;
typedef thrust::device_vector<short>					ShortDVec;

typedef thrust::host_vector<bool>						BoolHVec;
typedef thrust::device_vector<bool>						BoolDVec;

typedef thrust::host_vector<backRef>					BackRefHVec;
typedef thrust::device_vector<backRef>					BackRefDVec;

// Template structure to pass to kernel
template < typename T >
struct KernelArray
{
    T*  _array;
    int _size;

	__device__ __host__ KernelArray() {
	}

	__host__ KernelArray(thrust::device_vector< T >& dVec) {        
		this->_array = thrust::raw_pointer_cast( &dVec[0] );
		this->_size  = ( int ) dVec.size();
	}

	__host__ KernelArray(thrust::host_vector< T >& hVec) {        
		this->_array = thrust::raw_pointer_cast( &hVec[0] );
		this->_size  = ( int ) hVec.size();
	}
};

struct PrecisionData
{
	KernelArray< REAL > P1;
	KernelArray< REAL > P2;
	KernelArray< REAL > temp2a;
	KernelArray< REAL > temp2b;
	KernelArray< REAL > runningSum1;
	KernelArray< REAL > runningSum2;	

	PrecisionData(KernelArray< REAL >, KernelArray< REAL >, KernelArray< REAL >,
				  KernelArray< REAL >, KernelArray< REAL >, KernelArray< REAL >);
	//REAL P1[ 200 ], P2[ 200 ];								// 1633536
	//REAL temp2a[200], temp2b[200];							// 136130
	//REAL runningSum1[200], runningSum2[200];				// 1633536
};

#include "predicates.h"

struct DeviceData
{
	// Delaunay Triangulation
	KernelArray< gpudtVertex >				pDVec;
	KernelArray< gpudtTriangle >			triDVec;
	KernelArray< int >						ptTriMap;
	
	// Alpha Shape
	KernelArray< point_interval >			point_list;	
	KernelArray< edge_interval >			edge_list;
	KernelArray< triangle_interval >		triangle_list;
	
	KernelArray< Alpha >					master_list;
	KernelArray< REAL >						spectrum;
	KernelArray< backRef >					br;

	// Helper Structure
	KernelArray< int >						tri_edge_count;
	KernelArray< int >						tri_edge_indices;
	Predicates*								ps;
	int										num_edges;

	DeviceData();
	DeviceData(KernelArray< gpudtVertex >, KernelArray< gpudtTriangle >, KernelArray< int >, KernelArray< point_interval >,
			   KernelArray< edge_interval >, KernelArray< triangle_interval >, KernelArray< Alpha >, KernelArray< REAL >, 
			   KernelArray< backRef >, KernelArray< int >, KernelArray< int >, Predicates*, int);
	void set(KernelArray< gpudtVertex >, KernelArray< gpudtTriangle >, KernelArray< int >, KernelArray< point_interval >,
			   KernelArray< edge_interval >, KernelArray< triangle_interval >, KernelArray< Alpha >, KernelArray< REAL >, 
			   KernelArray< backRef >, KernelArray< int >, KernelArray< int >, Predicates*, int);
};

struct CPUData
{
	KernelArray< edge_interval >			edge_list;
	KernelArray< triangle_interval >		triangle_list;
	KernelArray< REAL >						spectrum;
	KernelArray< backRef >					br;
	Predicates*								ps;

	PGPUDTOUTPUT							pDT;
	PGPUDTPARAMS							pInput;

	CPUData(KernelArray< edge_interval >, KernelArray< triangle_interval >, KernelArray< REAL >, KernelArray< backRef >, PGPUDTOUTPUT, PGPUDTPARAMS, Predicates*);
};

class LessThanZero
{
public:
	__device__ __host__ bool operator()(const double &a) const
	{
		return (a < 0);
	}
};

class EqualToZero
{
public:
	__device__ __host__ bool operator()(const int &a) const
	{
		return (a == 0);
	}
};


class BackRefRemove
{
public:
	__device__ __host__ bool operator()(const backRef &a) const
	{
		return (a.face_index < 0);
	}
};

__global__ void computePtTriMap(KernelArray<gpudtVertex> pts, KernelArray<gpudtTriangle> tris, KernelArray<int> ptrimap);
__global__ void initPredicate(Predicates* ps);
__global__ void computeTriEdgeCount(DeviceData d);
__global__ void computeEdgeTriMap(DeviceData d, KernelArray<int> edge_tri_map);

__global__ void computeEdgeRo(DeviceData d, KernelArray<int> edge_tri_map);
__global__ void computeTriangleRo(DeviceData d);
__global__ void processAttached(DeviceData d);
__global__ void processSpectrum(DeviceData d);
__global__ void processEdges(DeviceData d);
__global__ void edgeIntervals(DeviceData d);

__device__ PairIndices get_edge(int& p1AbsIndex, int& p2AbsIndex, int cur, int tOpp, DeviceData d);
__global__ void computeVertexIntervals(DeviceData d);
__global__ void computeMasterList(DeviceData d, KernelArray< int > sRanks);
__global__ void markSpectrum(DeviceData* d, KernelArray< bool > gpu_marker/*, KernelArray< int > markerTid*/);
__device__ __host__ int categorize(DeviceData *d, const backRef& a, const backRef& b, int tid);
//__global__ void pre_mark (DeviceData *d, KernelArray< short > markerOrder);
__global__ void pre_sort(DeviceData* d, int phase, KernelArray< short > ordering, KernelArray< int > threadTid, KernelArray< bool > gpu_marker, int noOfThreads, int chunk);
//__global__ void orderSortReads(DeviceData* d, int phase, KernelArray< short > ordering, KernelArray< int > threadTid, KernelArray< bool > gpu_marker, int noOfThreads, int chunk);
__global__ void sorT(DeviceData* d, int phase, int *isSorted, KernelArray< int > threadTid, KernelArray< int > flag, KernelArray< bool > gpu_marker, KernelArray< bool > chunkSorted, int chunkNo, PrecisionData pd, int noOfThreads, int chunk, int round);
int exactCompare(CPUData d, const backRef& a, const backRef& b);

__global__ void UpdateBackRefs(DeviceData d, KernelArray< backRef > tempBr);

__global__ void fastCheck(DeviceData* d, int *isSortD);

__global__ void sorT2(DeviceData* d, int phase, int *isSorted, KernelArray< int > threadTid, KernelArray< int > flag, KernelArray< bool > gpu_marker, KernelArray< bool > chunkSorted, int chunkNo, PrecisionData pd, int noOfThreads, int chunk, int round);



