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

#include "kernel.h"
#include <cmath>

#define vertex(a,b) d.pDVec._array[d.triDVec._array[a].vtx[b]]
#define e_v(a) d.pDVec._array[a]
#define e_vp(a) d->pDVec._array[a]
#define e_vC(a) d.pInput->points[a]

DeviceData::DeviceData()
{
}

DeviceData::DeviceData(KernelArray< gpudtVertex > pv, KernelArray< gpudtTriangle > tv, KernelArray< int > pTri, KernelArray< point_interval > pList,
			   KernelArray< edge_interval > eList, KernelArray< triangle_interval > tl, KernelArray< Alpha > mD, KernelArray< REAL > sp, 
			   KernelArray< backRef > bR, KernelArray< int > tec, KernelArray< int > tei, Predicates* p, int ne)
{
	this->pDVec = pv;
	this->triDVec = tv;
	this->ptTriMap = pTri;
	this->point_list = pList;
	this->edge_list = eList;
	this->triangle_list = tl;
	this->master_list = mD;
	this->tri_edge_count = tec;
	this->tri_edge_indices = tei;
	this->spectrum = sp;
	this->br = bR;
	this->ps = p;
	this->num_edges = ne;
}


void DeviceData::set(KernelArray< gpudtVertex > pv, KernelArray< gpudtTriangle > tv, KernelArray< int > pTri, KernelArray< point_interval > pList,
			   KernelArray< edge_interval > eList, KernelArray< triangle_interval > tl, KernelArray< Alpha > mD, KernelArray< REAL > sp, 
			   KernelArray< backRef > bR, KernelArray< int > tec, KernelArray< int > tei, Predicates* p, int ne)
{
	this->pDVec = pv;
	this->triDVec = tv;
	this->ptTriMap = pTri;
	this->point_list = pList;
	this->edge_list = eList;
	this->triangle_list = tl;
	this->master_list = mD;
	this->tri_edge_count = tec;
	this->tri_edge_indices = tei;
	this->spectrum = sp;
	this->br = bR;
	this->ps = p;
	this->num_edges = ne;
}

PrecisionData::PrecisionData(KernelArray< REAL > p1, KernelArray< REAL > p2, KernelArray< REAL > t2a,
				  KernelArray< REAL > t2b, KernelArray< REAL > rsum1, KernelArray< REAL > rsum2)
{
	P1 = p1;
	P2 = p2;
	temp2a = t2a;
	temp2b = t2b;
	runningSum1 = rsum1;
	runningSum2 = rsum2;	
}

CPUData::CPUData(KernelArray< edge_interval > ei, KernelArray< triangle_interval > ti, KernelArray< REAL > s, KernelArray< backRef > brf,
					PGPUDTOUTPUT pD,PGPUDTPARAMS pInpu, Predicates *psp)
{
	this->edge_list = ei;
	this->triangle_list = ti;
	this->spectrum = s;
	this->br = brf;
	this->pDT = pD;
	this->pInput = pInpu;
	this->ps = psp;
}

__global__ void computePtTriMap(KernelArray<gpudtVertex> pts, KernelArray<gpudtTriangle> tris, KernelArray<int> ptrimap)
{	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= tris._size)
		return;

	while( tid < tris._size )
	{
		int ptIndex = 0;

		for(int i=0; i<3; i++)
		{
			ptIndex = tris._array[tid].vtx[i];
			// Check the performance without the check
			if(ptIndex >= 0 && ptIndex < ptrimap._size && ptrimap._array[ptIndex] == -1)
			{
				ptrimap._array[ptIndex] = tid;			
			}
		}
		tid += gridDim.x * blockDim.x;
	}
}

__global__ void initPredicate(Predicates* ps)
{	
	ps->exactinit();	
}

__global__ void computeEdgeTriMap(DeviceData d, KernelArray<int> edge_tri_map)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= d.triDVec._size)
		return;

	while( tid < d.triDVec._size )
	{
		int index = d.tri_edge_indices._array[tid];
		int count = d.tri_edge_count._array[tid];
		int s = 0;

		while(s < count)
		{
			edge_tri_map._array[index+s] = tid;
			s++;
		}

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void computeTriEdgeCount(DeviceData d)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= d.triDVec._size)
		return;

	while( tid < d.triDVec._size )
	{
		// Initialize the edges owned by current gpudtTriangle to 1
		d.tri_edge_count._array[ tid ] = 1;

		int p1Index = d.triDVec._array[tid].vtx[0];
		int p2Index = d.triDVec._array[tid].vtx[1];
		int p3Index = d.triDVec._array[tid].vtx[2];

		int opp1 = d.triDVec._array[tid].tri[0];
		int opp2 = d.triDVec._array[tid].tri[1];
		int opp3 = d.triDVec._array[tid].tri[2];

		int counter = 0;
		if( p1Index < p2Index )
			counter++;
		if( p2Index < p3Index )
			counter++;
		if( p3Index < p1Index )
			counter++;

		// check if the gpudtTriangle is a hull gpudtTriangle
		// ToDo: Minimize branch instructions
		if( opp1 == -1 && p2Index > p3Index )
		{
			if(counter < 3)
			{
				counter++;
			}
		}

		if( opp2 == -1 && p3Index > p1Index )
		{
			if(counter < 3)
			{
				counter++;
			}
		}

		if( opp3 == -1 && p1Index > p2Index )
		{
			if(counter < 3)
			{
				counter++;
			}
		}

		d.tri_edge_count._array[ tid ] = counter;	

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void computeTriangleRo(DeviceData d)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= d.triDVec._size)
		return;

	while( tid < d.triDVec._size )
	{
		if(!tid)
		{
			d.spectrum._array[0] = 0;
			d.br._array[0].face_index = 0;
			d.br._array[0].st = EDGE;
			d.br._array[0].my_index = 0;			
		}

		REAL alpha_face = 0;
		alpha_face = d.ps->birth_time( vertex(tid,0), vertex(tid,1), vertex(tid,2) );
		d.spectrum._array[ tid+1 ] = alpha_face;	

		triangle_interval ti;
		ti.ro_rank = tid+1;
		d.triangle_list._array[ tid ] = ti;

		// Set the backref to update while sorting
		d.br._array[ tid+1 ].face_index = tid;	
		d.br._array[ tid+1 ].st = TRIANGLE;
		d.br._array[ tid+1 ].my_index = tid+1;

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void computeEdgeRo(DeviceData d, KernelArray<int> edge_tri_map)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid >= d.num_edges )
		return;	

	while( tid < d.num_edges )
	{
		int tri_index = edge_tri_map._array[tid];
		int start = d.tri_edge_indices._array[ tri_index ];
		int edge_index = tid;

		int i=0, counter = 0;
		int p1AbsIndex = 0;
		int p2AbsIndex = 0;
		int tOpp = 0, j=0, k=0;		
		
		for( i=0; i < 3; i++)
		{
			j = (i+1)%3;
			k = (i+2)%3;
			p1AbsIndex = d.triDVec._array[ tri_index ].vtx[j];
			p2AbsIndex = d.triDVec._array[ tri_index ].vtx[k];
			tOpp = d.triDVec._array[ tri_index ].tri[i];

			if( p1AbsIndex < p2AbsIndex || tOpp == -1 )
			{
				if(counter == (edge_index-start))
				{					
					break;
				}
				counter++;			
			}
		}	

		if( p1AbsIndex < p2AbsIndex || tOpp == -1 )
		{
			int spectrumIndex = edge_index + d.triDVec._size + 1;
			d.spectrum._array[spectrumIndex] = d.ps->birth_time( e_v(p1AbsIndex), e_v(p2AbsIndex) );			

			// Set the back reference to update while sorting
			d.br._array[spectrumIndex].face_index = edge_index;
			d.br._array[spectrumIndex].st = EDGE;
			d.br._array[spectrumIndex].my_index = spectrumIndex;

			edge_interval ei;
			ei.ro_rank = spectrumIndex;
			ei.tIndex = tri_index;
			ei.vOpp = i;

			d.edge_list._array[edge_index] = ei;
			//printf("Edge = %d\n", edge_index);
		}

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void processAttached(DeviceData d)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid >= d.num_edges )
		return;

	while( tid < d.num_edges )
	{
		edge_interval ei = d.edge_list._array[tid];
		
		int tOpp = d.triDVec._array[ei.tIndex].tri[ei.vOpp];
		int i,j,k;
		int spectrumIndex = tid + d.triDVec._size + 1;

		i = (ei.vOpp + 1)%3;
		j = (ei.vOpp + 2)%3;	

		if(tOpp != -1)
		{
			for(int m=0; m<3; m++)
				if(d.triDVec._array[tOpp>>2].tri[m]>>2 == ei.tIndex)
					k = m;	
			//printf(" edge = %d, k = %d, tOpp = %d, tIndex = %d, vOpp = %d\n", tid, k, tOpp>>2, ei.tIndex, ei.vOpp);
		}	
		
		bool print = false;
		if( spectrumIndex == 289349 )
			print = true;
		if( d.ps->isAttached(vertex(ei.tIndex,i), vertex(ei.tIndex,j), vertex(ei.tIndex,ei.vOpp), print) )
		{
			d.edge_list._array[tid].ro_rank = 0;
			d.spectrum._array[spectrumIndex] = -1;
			d.br._array[spectrumIndex].face_index = -1;
			/*if(spectrumIndex == 289351)
				printf("Tri: %d, edge: %d\n", ei.tIndex, ei.vOpp);*/
		}

		else if( tOpp != -1 && d.ps->isAttached(vertex(ei.tIndex,i), vertex(ei.tIndex,j), vertex(tOpp>>2,k), print) )
		{
			d.edge_list._array[tid].ro_rank = 0;
			d.spectrum._array[spectrumIndex] = -1;
			d.br._array[spectrumIndex].face_index = -1;
			/*if(spectrumIndex == 289351)
				printf("Opp tri: %d, edge: %d\n", tOpp, k);*/
		}

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void processSpectrum(DeviceData d)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid >= d.spectrum._size )
		return;

	while( tid < d.spectrum._size )
	{
		if(d.br._array[tid].my_index != -1)
			d.br._array[tid].my_index = tid;
		tid = tid + gridDim.x * blockDim.x;
	}
}

__global__ void UpdateBackRefs(DeviceData d, KernelArray< backRef > tempBr)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid >= tempBr._size )
		return;

	while( tid < tempBr._size )
	{
		if( tid == 0 )
		{		
			return;
		}

		int index = tempBr._array[tid].my_index;
		do
		{
			backRef brs = d.br._array[index];			
			if(brs.st == EDGE)
				d.edge_list._array[ brs.face_index ].ro_rank = tid;
			else
				d.triangle_list._array[ brs.face_index ].ro_rank = tid;

			index++;
		}while( index < d.br._size && d.br._array[index].my_index == -1 );		

		tid = tid + gridDim.x * blockDim.x;
	}
}

__global__ void processEdges(DeviceData d)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid >= d.spectrum._size - d.triangle_list._size )
		return;

	while( tid < d.spectrum._size - d.triangle_list._size )
	{
		int brIndex = tid + d.triDVec._size + 1;
		if( brIndex < d.br._size )
		{
			backRef brs = d.br._array[brIndex];	
			
			if(brs.st == EDGE)
			{		
				d.edge_list._array[ brs.face_index ].ro_rank = brIndex;
				d.br._array[ brIndex ].my_index = brIndex;
			}
		}

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void edgeIntervals(DeviceData d)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid >= d.num_edges )
		return;

	while( tid < d.num_edges )
	{
		edge_interval ei = d.edge_list._array[tid];
		int tIndex = ei.tIndex;
		int tOpp = d.triDVec._array[tIndex].tri[ei.vOpp];
		int tRoRank = d.triangle_list._array[tIndex].ro_rank;

		if(tOpp == -1)
		{
			d.edge_list._array[tid].nu_max_rank = 0;
			d.edge_list._array[tid].nu_min_rank = tRoRank;
		}
		else
		{		
			int oppRoRank = d.triangle_list._array[tOpp>>2].ro_rank;
			if( tRoRank <= oppRoRank )
			{
				d.edge_list._array[tid].nu_min_rank = tRoRank;
				d.edge_list._array[tid].nu_max_rank = oppRoRank;
			}
			else
			{
				d.edge_list._array[tid].nu_min_rank = oppRoRank;
				d.edge_list._array[tid].nu_max_rank = tRoRank;
			}
		}

		tid += gridDim.x * blockDim.x;
	}
}

__device__ PairIndices get_edge(int& p1AbsIndex, int& p2AbsIndex, int cur, int tOpp, DeviceData d)
{
	PairIndices edge;

	int tBelong = 0;
	if( p1AbsIndex < p2AbsIndex || tOpp == -1)
	{
		tBelong = cur;
		//tOther = tOpp;
	}
	else
	{
		tBelong = tOpp>>2;
		int temp = p1AbsIndex;
		p1AbsIndex = p2AbsIndex;
		p2AbsIndex = temp;
		//tOther = cur;
	}

	edge.first = tBelong;

	for(int i=0; i<3; i++)
	{
		int j = (i+1)%3;
		int k = (i+2)%3;

		if( d.triDVec._array[tBelong].vtx[j] == p1AbsIndex && d.triDVec._array[tBelong].vtx[k] == p2AbsIndex )
		{
			edge.second = i;
			break;
		}
	}

	int counter = 0;
	for(int i = 0; i < 3; i++)
	{
		int locj = (i+1)%3;
		int lock = (i+2)%3;
		int locP1AbsIndex = d.triDVec._array[ tBelong ].vtx[locj];
		int locP2AbsIndex = d.triDVec._array[ tBelong ].vtx[lock];
		int locTOpp = d.triDVec._array[ tBelong ].tri[i];
		
		if( locP1AbsIndex < locP2AbsIndex || locTOpp == -1)
		{
			if(i == edge.second)
				break;
			counter++;
		}
	}

	edge.second = d.tri_edge_indices._array[ tBelong ] + counter;
	return edge;
}

__global__ void computeVertexIntervals(DeviceData d)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid >= d.pDVec._size )
		return;

	while( tid < d.pDVec._size )
	{
		int pGlobalIndex = tid, tIndex = d.ptTriMap._array[tid];
		int pLocalIndex = 0, pLocalIndexF = 0;
		int mini = -1, maxi = -1;
		int cur = tIndex, counter = 0;
		bool first = true;
		int mode = 1;

		do
		{			
			if(counter)
				cur >>= 2;
			
			if( cur >= d.triDVec._size || cur < 0 )
			{
				//printf("problem = %d, tid,ptIndex = %d\n", cur, tid);
				break;
			}

			for(int i=0; i<3; i++)
				if(d.triDVec._array[cur].vtx[i] == pGlobalIndex)
				{
					pLocalIndex = i;
					if(first)
						pLocalIndexF = i;
					break;
				}

			int p1AbsIndex = 0;
			int p2AbsIndex = 0;
			int tOpp = 0, j=0, k=0;		

			for(int i=0; i<3; i++)	
			{
				if(i == pLocalIndex)
					continue;
				
				j = (i+1)%3;
				k = (i+2)%3;
				p1AbsIndex = d.triDVec._array[ cur ].vtx[j];
				p2AbsIndex = d.triDVec._array[ cur ].vtx[k];
				tOpp = d.triDVec._array[ cur ].tri[i];
				
				PairIndices edge = get_edge(p1AbsIndex, p2AbsIndex, cur, tOpp, d);

				int edge_index = edge.second;			
				edge_interval ei = d.edge_list._array[edge_index];			

				if(cur == tIndex && first)
				{
					if(ei.ro_rank != 0)
					{
						mini = ei.ro_rank;
						maxi = ei.nu_max_rank;
					}
					else
					{
						mini = ei.nu_min_rank;
						maxi = ei.nu_max_rank;
					}				
					
					first = false;
					//printf("reaching : %d\n", tid);
				}			

				if( ei.ro_rank == 0 && ei.nu_min_rank < mini)
						mini = ei.nu_min_rank;
				else if( ei.ro_rank != 0 && ei.ro_rank < mini )
						mini = ei.ro_rank;					

				maxi = MAX(maxi, ei.nu_max_rank);
			}			

			int next = (pLocalIndex + mode)%3;
			cur = d.triDVec._array[cur].tri[next];
			counter++;

			if(cur == -1)
			{
				if(mode == 2)
					break;

				cur = tIndex;
				mode = 2;
				next = (pLocalIndexF + mode) % 3;			
				cur = d.triDVec._array[cur].tri[next];

				if(cur == -1)
					break;			
			}
		} while(cur != -1 && (cur>>2 != tIndex));

		d.point_list._array[pGlobalIndex].nu_min_rank = mini;
		d.point_list._array[pGlobalIndex].nu_max_rank = maxi;
		if(mini >= d.spectrum._size || maxi >= d.spectrum._size)
			printf("Greater: index = %d, tri = %d, mini = %d, maxi = %d, spec = %d\n", pGlobalIndex, tIndex, mini, maxi, d.spectrum._size);

		tid += gridDim.x * blockDim.x;
	}
}

__global__ void computeMasterList(DeviceData d, KernelArray< int > sRanks)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int nPoints = d.pDVec._size;
	int nTris = d.triDVec._size;
	int nEdges = d.num_edges;

	if( tid >= (2 * nPoints + 3 * nEdges + nTris ) )
		return;
	
	while( tid < ( 2* nPoints + 3 * nEdges + nTris ) )
	{
		Alpha master_entry;	

		if( tid < 2 * nPoints )
		{
			int point_value_index = tid;

			master_entry.st = VERTEX;
			master_entry.simplex_index = point_value_index / 2;
			master_entry.at = (alpha_type)((point_value_index % 2) + 1);

			if( master_entry.at == alpha_type::nu_min )
				sRanks._array[ tid ] = d.point_list._array[ master_entry.simplex_index ].nu_min_rank;

			else if( master_entry.at == alpha_type::nu_max )
				sRanks._array[ tid ] = d.point_list._array[ master_entry.simplex_index ].nu_max_rank;
		}

		else if( tid < 2 * nPoints + 3 * nEdges )
		{			
			int edge_value_index = tid - 2 * nPoints;
			int edge_index = edge_value_index / 3;				
			
			master_entry.st = EDGE;
			master_entry.simplex_index = edge_index;		
			master_entry.at = (alpha_type)(edge_value_index % 3);	

			if( master_entry.at == alpha_type::ro )
				sRanks._array[ tid ] = d.edge_list._array[ master_entry.simplex_index ].ro_rank;

			else if( master_entry.at == alpha_type::nu_min )
				sRanks._array[ tid ] = d.edge_list._array[ master_entry.simplex_index ].nu_min_rank;

			else if( master_entry.at == alpha_type::nu_max )
				sRanks._array[ tid ] = d.edge_list._array[ master_entry.simplex_index ].nu_max_rank;
		}	

		else if( tid < 2 * nPoints + 3 * nEdges + nTris )
		{				
			int tIndex = tid - 2 * nPoints - 3 * nEdges;		
							
			master_entry.st = TRIANGLE;
			master_entry.simplex_index = tIndex;
			master_entry.at = ro;

			if( master_entry.at == alpha_type::ro )
				sRanks._array[ tid ] = d.triangle_list._array[ master_entry.simplex_index ].ro_rank;
		}	

		d.master_list._array[ tid ] = master_entry;

		tid += gridDim.x * blockDim.x;
	}
}

__device__ __host__ int fastCompare(DeviceData *d, const backRef& a, const backRef& b, int print)
{
	int v1,v2,v4,v5;
	int p1=0,p2=0,p3=0,p4=0,p5=0,p6=0;		

	if(a.st == EDGE)
	{
		v1 = (d->edge_list._array[a.face_index].vOpp + 1) % 3;
		v2 = (d->edge_list._array[a.face_index].vOpp + 2) % 3;
		int t1 = d->edge_list._array[a.face_index].tIndex;		
		
		p1 = d->triDVec._array[ t1 ].vtx[ v1 ];
		p2 = d->triDVec._array[ t1 ].vtx[ v2 ];
	}
	else if(a.st == TRIANGLE)
	{		
		p1 = d->triDVec._array[ a.face_index ].vtx[0];
		p2 = d->triDVec._array[ a.face_index ].vtx[1];
		p3 = d->triDVec._array[ a.face_index ].vtx[2];
	}

	if(b.st == EDGE)
	{			
		v4 = (d->edge_list._array[b.face_index].vOpp + 1) % 3;
		v5 = (d->edge_list._array[b.face_index].vOpp + 2) % 3;
		int t2 = d->edge_list._array[b.face_index].tIndex;		
		
		p4 = d->triDVec._array[ t2 ].vtx[ v4 ];
		p5 = d->triDVec._array[ t2 ].vtx[ v5 ];
	}
	else if(b.st == TRIANGLE)
	{		
		p4 = d->triDVec._array[ b.face_index ].vtx[0];
		p5 = d->triDVec._array[ b.face_index ].vtx[1];
		p6 = d->triDVec._array[ b.face_index ].vtx[2];
	}

	int res = -1;

	if( a.st == EDGE && b.st == EDGE )
		res = d->ps->compareFast( e_vp(p1), e_vp(p2), e_vp(p4), e_vp(p5), print);

	else if( a.st == EDGE && b.st == TRIANGLE )
		res = d->ps->compareFast( e_vp(p1), e_vp(p2), e_vp(p4), e_vp(p5), e_vp(p6), 1, print);

	else if( a.st == TRIANGLE && b.st == EDGE)
		res = d->ps->compareFast( e_vp(p1), e_vp(p2), e_vp(p3), e_vp(p4), e_vp(p5), print);

	else
		res = d->ps->compareFast( e_vp(p1), e_vp(p2), e_vp(p3), e_vp(p4), e_vp(p5), e_vp(p6), print);
		
	return res;
}

//__global__ void pre_mark (DeviceData *d, KernelArray< short > markerOrder)
//{
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	int size = (d->spectrum._size);
//	if( tid >= size )
//		return;
//
//	int  m, n;
//	while( tid < size )
//	{
//		m = tid;
//		n = tid+1;
//
//		if(n < d->br._size && m != 0)
//		{
//			backRef b1 = d->br._array[m]; 
//			backRef b2 = d->br._array[n];						
//
//			int res = -1;
//
//			if( b1.st == EDGE && b2.st == EDGE)
//				res = 0;
//
//			else if( b1.st == EDGE && b2.st == TRIANGLE)
//				res = 1;
//
//			else if( b1.st == TRIANGLE && b2.st == EDGE)
//				res = 2;
//
//			else
//				res = 3;
//
//			markerOrder._array[tid] = res;
//		}
//		
//		tid += gridDim.x * blockDim.x;
//	}
//}

__global__ void markSpectrum(DeviceData* d, KernelArray< bool > gpu_marker/*, KernelArray< int > markerTid*/)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int size = (d->spectrum._size);
	if( tid >= size )
		return;

	int  m, n/*, processingTid*/;
	while( tid < size )
	{
		//processingTid = markerTid._array[tid];
		m = tid; //processingTid;
		n = tid+1; //processingTid+1;

		if(n < d->br._size && m != 0)
		{
			backRef b1 = d->br._array[m]; 
			backRef b2 = d->br._array[n];						

			int res = -1;
			int print = 0;
			res = fastCompare(d, b1, b2, print);			

			if( res != -1 )
			{
				gpu_marker._array[n] = true;				
			}
			if( d->spectrum._array[m] >= 0.0198248 && d->spectrum._array[m] < 0.019825)
			{
				gpu_marker._array[m-1] = true;
				gpu_marker._array[m] = true;
				gpu_marker._array[m+1] = true;
				//printf("m = %d, n = %d, spectrum[m] = %.20lf, spectrum[n] = %.20lf\n", m, n, res, d->spectrum._array[m], d->spectrum._array[n]);
			}
		}
		
		tid += gridDim.x * blockDim.x;
}
}

__global__ void fastCheck(DeviceData* d, int *isSortD/*, KernelArray< int > markerTid*/)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int size = (d->edge_list._size);
	if( tid >= size )
		return;

	int  m, n;
	while( tid < size )
	{
		m = d->edge_list._array[tid].ro_rank;				

		if(m < d->br._size && m != 0)
		{
			backRef b1 = d->br._array[m];
			n = b1.face_index;

			if( b1.st != EDGE )
			{
				printf("Problem: Not an edge");
				return;
			}

			backRef b2 = d->br._array[n];

			int res = fastCompare(d, b1, b2, 0);

			if( res != 0 )
			{
				*isSortD = 0;				
			}
		}
		
		tid += gridDim.x * blockDim.x;
	}
}

__device__ __host__ int categorize(DeviceData *d, const backRef& a, const backRef& b, int tid)
{		
	int res = -1;	

	if( a.st == EDGE && b.st == EDGE)
		res = 0;

	else if( a.st == EDGE && b.st == TRIANGLE)
		res = 1;

	else if( a.st == TRIANGLE && b.st == EDGE)
		res = 2;

	else
		res = 3;	

	return res;
}

__global__ void pre_sort(DeviceData* d, int phase, KernelArray< short > ordering, KernelArray< int > threadTid, KernelArray< bool > gpu_marker, int noOfThreads, int chunk)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid >= noOfThreads )
		return;

	int origTid = tid;
	while( tid < (noOfThreads) && tid < d->spectrum._size )
	{
		int m,n;
		if( phase == 1 )
		{
			m = chunk + 2*tid;
			n = chunk + 2*tid + 1;
		}
		else
		{
			m = chunk + 2*tid + 1;
			n = chunk + 2*tid + 2;
		}

		if(n < d->br._size && m >= 0 && (gpu_marker._array[m] || gpu_marker._array[n]))
		{
			backRef b1 = d->br._array[m]; 
			backRef b2 = d->br._array[n];						

			int res = -1;
							
			res = categorize(d, b1, b2, origTid);

			ordering._array[ tid ] = res;			
		}
		
		tid += (noOfThreads);
	}
}

//__global__ void orderSortReads(DeviceData* d, int phase, KernelArray< short > ordering, KernelArray< int > threadTid, KernelArray< bool > gpu_marker, int noOfThreads, int chunk)
//{
//	int tid = threadIdx.x + blockIdx.x * blockDim.x;
//	if( tid >= noOfThreads )
//		return;
//	
//	while( tid < (noOfThreads) && tid < d->spectrum._size )
//	{
//		int m, n, processingTid;		
//		processingTid = threadTid._array[tid];
//
//		if( phase == 1 )
//		{
//			m = chunk + 2*processingTid;
//			n = chunk + 2*processingTid + 1;
//		}
//		else
//		{
//			m = chunk + 2*processingTid + 1;
//			n = chunk + 2*processingTid + 2;
//		}
//
//		if(n < d->br._size && m >= 0 && (gpu_marker._array[m] || gpu_marker._array[n]))
//		{
//			backRef b1 = d->br._array[m]; 
//			backRef b2 = d->br._array[n];	
//			
//			int index = b1.face_index;			
//
//			if( b1.st == EDGE && b2.st == TRIANGLE)
//				index = index + noOfThreads;
//
//			else if( b1.st == TRIANGLE && b2.st == EDGE)
//				index = index + 2*noOfThreads;
//
//			else if(b1.st == TRIANGLE && b2.st == TRIANGLE)
//				index = index + 3*noOfThreads;
//
//			ordering._array[tid] = index;
//		}
//		tid += noOfThreads;
//	}
//}

__device__ __host__ int secondPassCompare(DeviceData *d, const backRef& a, const backRef& b, PrecisionData pd, int tid)
{
	int v1,v2,v4,v5;
	int p1=0,p2=0,p3=0,p4=0,p5=0,p6=0;		

	if(a.st == EDGE)
	{		
		v1 = (d->edge_list._array[a.face_index].vOpp + 1) % 3;
		v2 = (d->edge_list._array[a.face_index].vOpp + 2) % 3;
		int t1 = d->edge_list._array[a.face_index].tIndex;		
		
		p1 = d->triDVec._array[ t1 ].vtx[ v1 ];
		p2 = d->triDVec._array[ t1 ].vtx[ v2 ];
	}
	else if(a.st == TRIANGLE)
	{				
		p1 = d->triDVec._array[ a.face_index ].vtx[0];
		p2 = d->triDVec._array[ a.face_index ].vtx[1];
		p3 = d->triDVec._array[ a.face_index ].vtx[2];
	}

	if(b.st == EDGE)
	{			
		v4 = (d->edge_list._array[b.face_index].vOpp + 1) % 3;
		v5 = (d->edge_list._array[b.face_index].vOpp + 2) % 3;
		int t2 = d->edge_list._array[b.face_index].tIndex;		
		
		p4 = d->triDVec._array[ t2 ].vtx[ v4 ];
		p5 = d->triDVec._array[ t2 ].vtx[ v5 ];
	}
	else if(b.st == TRIANGLE)
	{			
		p4 = d->triDVec._array[ b.face_index ].vtx[0];
		p5 = d->triDVec._array[ b.face_index ].vtx[1];
		p6 = d->triDVec._array[ b.face_index ].vtx[2];
	}

	int res = -1;	

	if( a.st == EDGE && b.st == EDGE)
		res = d->ps->compareSlow( e_vp(p1), e_vp(p2), e_vp(p4), e_vp(p5), pd, tid );

	else if( a.st == EDGE && b.st == TRIANGLE)
		res = d->ps->compareSlow( e_vp(p1), e_vp(p2), e_vp(p4), e_vp(p5), e_vp(p6), 1, pd, tid );

	else if( a.st == TRIANGLE && b.st == EDGE)
		res = d->ps->compareSlow( e_vp(p1), e_vp(p2), e_vp(p3), e_vp(p4), e_vp(p5), pd, tid );

	else
		res = d->ps->compareSlow( e_vp(p1), e_vp(p2), e_vp(p3), e_vp(p4), e_vp(p5), e_vp(p6), pd, tid );	

	return res;
}

__global__ void sorT(DeviceData* d, int phase, int *isSorted, KernelArray< int > threadTid, KernelArray< int > flag, KernelArray< bool > gpu_marker, KernelArray< bool > chunkSorted, int chunkNo, PrecisionData pd, int noOfThreads, int chunk, int round)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if( tid >= noOfThreads )
		return;

	int origTid = tid;	
	while( tid < noOfThreads && tid < d->spectrum._size )
	{
		int m, n, processingTid;
		processingTid = threadTid._array[tid];

		if( phase == 1 )
		{
			m = chunk + 2*processingTid;
			n = chunk + 2*processingTid + 1;
		}
		else
		{
			m = chunk + 2*processingTid + 1;
			n = chunk + 2*processingTid + 2;
		}		

		if(n < d->br._size && m >= 0 && (gpu_marker._array[m] || gpu_marker._array[n]))
		{
			backRef b1 = d->br._array[m]; 
			backRef b2 = d->br._array[n];						

			int res = -1;
							
			res = secondPassCompare(d, b1, b2, pd, origTid);			

			if( res > 0 )
			{
				REAL tmp = d->spectrum._array[m];
				d->spectrum._array[m] = d->spectrum._array[n];
				d->spectrum._array[n] = tmp;				
				
				d->br._array[m] = b2;
				d->br._array[n] = b1;	
				
				if(*isSorted == 1)
				{
					*isSorted = 0;
				}

				chunkSorted._array[ chunkNo ] = true;				
				if( (m == chunk || m == chunk + 1) && chunkNo > 0 && !chunkSorted._array[ chunkNo - 1 ])
				{
					chunkSorted._array[ chunkNo - 1 ] = true;
					gpu_marker._array[m-1] = true;
				}

				if( (n == chunk+2*noOfThreads-1 || n == chunk+2*noOfThreads || n == chunk+2*noOfThreads+2) && chunkNo < chunkSorted._size-1 && !chunkSorted._array[ chunkNo + 1 ])
				{
					chunkSorted._array[ chunkNo + 1 ] = true;
					gpu_marker._array[n+1] = true;
				}
								
				gpu_marker._array[m] = true;
				gpu_marker._array[n] = true;								
			}						
			else if( res == -1 )
			{
				if( d->br._array[ m ].my_index != -1)
					flag._array[ m ] = 0;
				if( d->br._array[ n ].my_index != -1)
					flag._array[ n ] = 0;	

				if( round != 1 )
					gpu_marker._array[m] = false;
				gpu_marker._array[n] = false;
			}
			else if( res == 0 )
			{								
				d->br._array[ n ].my_index = -1;								
			}
			else if( res < -1 )
			{
				flag._array[ m ] = 1;
				flag._array[ n ] = 1;
			}			
		}		

		tid += (noOfThreads);
	}
}

int exactCompare(CPUData d, const backRef& a, const backRef& b)
{
	int v1,v2,v4,v5;
	int p1=0,p2=0,p3=0,p4=0,p5=0,p6=0;	
			
	if(a.st == EDGE)
	{
		v1 = (d.edge_list._array[a.face_index].vOpp + 1) % 3;
		v2 = (d.edge_list._array[a.face_index].vOpp + 2) % 3;
		int t1 = d.edge_list._array[a.face_index].tIndex;		
		
		p1 = d.pDT->triangles[ t1 ].vtx[ v1 ];
		p2 = d.pDT->triangles[ t1 ].vtx[ v2 ];
	}
	else if(a.st == TRIANGLE)
	{
		p1 = d.pDT->triangles[ a.face_index ].vtx[0];
		p2 = d.pDT->triangles[ a.face_index ].vtx[1];
		p3 = d.pDT->triangles[ a.face_index ].vtx[2];
	}

	if(b.st == EDGE)
	{			
		v4 = (d.edge_list._array[b.face_index].vOpp + 1) % 3;
		v5 = (d.edge_list._array[b.face_index].vOpp + 2) % 3;
		int t2 = d.edge_list._array[b.face_index].tIndex;		
		
		p4 = d.pDT->triangles[ t2 ].vtx[ v4 ];
		p5 = d.pDT->triangles[ t2 ].vtx[ v5 ];
	}
	else if(b.st == TRIANGLE)
	{		
		p4 = d.pDT->triangles[ b.face_index ].vtx[0];
		p5 = d.pDT->triangles[ b.face_index ].vtx[1];
		p6 = d.pDT->triangles[ b.face_index ].vtx[2];
	}

	int res = -1;	
	
	if( a.st == EDGE && b.st == EDGE )
		res = d.ps->CPUcompareSlow( e_vC(p1), e_vC(p2), e_vC(p4), e_vC(p5) );

	else if( a.st == EDGE && b.st == TRIANGLE )
		res = d.ps->CPUcompareSlow( e_vC(p1), e_vC(p2), e_vC(p4), e_vC(p5), e_vC(p6), 1 );

	else if( a.st == TRIANGLE && b.st == EDGE)
		res = d.ps->CPUcompareSlow( e_vC(p1), e_vC(p2), e_vC(p3), e_vC(p4), e_vC(p5) );

	else
		res = d.ps->CPUcompareSlow( e_vC(p1), e_vC(p2), e_vC(p3), e_vC(p4), e_vC(p5), e_vC(p6) );
		
	return res;
}
