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

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <map>

using namespace std;

enum simplex_type { EDGE=0, TRIANGLE=1, VERTEX=2 };
enum alpha_type { ro=0, nu_min=1, nu_max=2 };
enum simplex_category { ATTACHED=0, UNATTACHED };

struct point_interval
{
	int nu_min_rank, nu_max_rank;
};

struct edge_interval
{
	int ro_rank, nu_min_rank, nu_max_rank;
	simplex_category sc;
	int tIndex, vOpp;

	void init(int,int,int);
};

struct triangle_interval
{
	int ro_rank;
};

struct PairIndices
{
	int first, second;
};

struct backRef
{
	int my_index;
	int face_index;
	simplex_type st;
};

struct Alpha
{		
	int simplex_index;
	simplex_type st;
	alpha_type at;

	void init(int simlex_index, simplex_type s);	
};
