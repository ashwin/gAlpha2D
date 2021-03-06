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


// #ifdef _WIN32
// #  define WINDOWS_LEAN_AND_MEAN
// #  include <windows.h>
// #endif

// #ifndef _WIN32
// #define fopen_s(p,f,t) {(*p)=fopen(f,t);if((*p)==NULL){printf("Error opening file %s, file: %s, line: %d\n",f,__FILE__,__LINE__);exit(EXIT_FAILURE);}}
// #endif

// #ifndef _WIN32
// #ifndef max
// #define max(p1,p2) (p1>p2)?p1:p2
// #endif
// #endif

#include <unistd.h>


#ifndef _CLOCK_T_DEFINED
typedef long clock_t;
#define _CLOCK_T_DEFINED
#endif


#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "include/GL/glut.h"
#include <time.h>
#include <iostream>
#include<fstream>
#include <string>

#include "gpudt/gpudt.h"
#include "gpudt/predicates.h"

// Added for Alpha Shapes
#include "gpudt/AlphaShape/Alpha_Shapes_2D.h"
// #include "gpudt/AlphaShape/Visualizer.h"

using namespace std;

//**********************************************************************************************
//* Testing
//**********************************************************************************************
// Global Vars
extern bool alpha_display;
int winWidth    = 512;
int winHeight   = 512;

int fboSize     = 2048;  

char vertex_file[]      = "data.vertex"; 
char constraint_file[]  = "constraint.txt"; 

// Mouse Control
int     oldMouseX, oldMouseY;
int     currentMouseX, currentMouseY;
float   clickedX        = 0.0f;
float   clickedY        = 0.0f; 
bool    isLeftMouseActive, isRightMouseActive;

// Transformation
float zDepth          = 1.0f;
float xTranslate      = 0.0f;
float yTranslate      = 0.0f;

// Triangle selection
int clickedTri = -1; 

// GPU-DT Vars
PGPUDTPARAMS    pInput;
PGPUDTOUTPUT    pOutput;

// Alpha Shapes Variables
bool alpha_display = false;
Alpha_Shapes_2D *as_global;

string filename = "";

void ReadDataFromFileBinary(PGPUDTPARAMS &pParams)
{  
    pParams = new GPUDTPARAMS; 	
	printf("Reading data from file: %s ... \n", filename.c_str());

	ifstream fp1(filename.c_str());
	fp1 >> pParams->nPoints;
	pParams->points  = new gpudtVertex[pParams->nPoints];

	for(int i=0; i<pParams->nPoints; i++)
	{
		double x,y;
		int index;
		fp1 >> index >> x >> y;
		pParams->points[index].x = x;
		pParams->points[index].y = y;
	}
	pParams->nConstraints = 0;	

    // Find min and max coordinates
    REAL minx = 1E20,  miny = 1E20;
    REAL maxx = -1E20, maxy = -1E20;

    for (int i = 0; i < pParams->nPoints; i++)
    {		
        if(pParams->points[i].x > maxx)
            maxx = pParams->points[i].x;

        if(pParams->points[i].x < minx)
            minx = pParams->points[i].x;

        if(pParams->points[i].y > maxy)
            maxy = pParams->points[i].y;

        if(pParams->points[i].y < miny)
            miny = pParams->points[i].y;
    }

    pParams->maxX = maxx;
    pParams->maxY = maxy;
    pParams->minX = minx;
    pParams->minY = miny;

    printf("Input is ready\n");
}

void checkResult(PGPUDTPARAMS input, PGPUDTOUTPUT output)
{
    gpudtVertex    *points = (gpudtVertex *) input->points; 
    int            *oneTri = new int[input->nPoints]; 

    // Find for each vertex one triangle containing it
    memset(oneTri, -1, input->nPoints * sizeof(int)); 

    printf("Checking vertex indices..."); 

    gpudtOrientedTriangle otri, onexttri; 

    for (int i = 0; i < output->nTris; i++) 
    {
        const gpudtTriangle& t = output->triangles[i]; 

        if ( ( t.vtx[0] < 0 || t.vtx[0] >= input->nPoints ) || 
            ( t.vtx[1] < 0 || t.vtx[1] >= input->nPoints ) || 
            ( t.vtx[2] < 0 || t.vtx[2] >= input->nPoints ) )
        {
            printf("\n*** ERROR *** Invalid vertex index in triangle %i\n", i); 
            printf("%i %i %i\n", t.vtx[0], t.vtx[1], t.vtx[2]); 
            continue; 
        }

        otri.tri = i; 

        otri.orient = 0; oneTri[ t.vtx[0] ] = Encode( otri ); 
        otri.orient = 1; oneTri[ t.vtx[1] ] = Encode( otri ); 
        otri.orient = 2; oneTri[ t.vtx[2] ] = Encode( otri ); 
    }

    for (int i = 0; i < input->nPoints; i++) 
    {
        int firstTri = oneTri[i]; 

        if ( firstTri == -1 )
            printf("\n*** ERROR *** Vertex %i is not in the triangulation\n", i); 
    }

    printf("\nChecking orientations..."); 

    for (int i = 0; i < output->nTris; i++) 
    {
        gpudtTriangle t = output->triangles[i]; 
        REAL check = counterclockwise(&points[t.vtx[0]], &points[t.vtx[1]], &points[t.vtx[2]]); 

        if (check <= 0.0) 
        {
            printf("\n*** ERROR *** Wrong orientation at triangle %i (%i, %i, %i)\n", 
                i, t.vtx[0], t.vtx[1], t.vtx[2]); 
            printf("%.20f\n", check); 
        }
    }

    printf("\nChecking constraints..."); 

    bool *edgeIsConstraint = new bool[output->nTris * 3]; 
    int ptr; 

    memset(edgeIsConstraint, false, output->nTris * 3 * sizeof(bool)); 

    for (int i = 0; i < input->nConstraints; i++) 
    {
        int a = input->constraints[ i * 2 ]; 
        int b = input->constraints[ i * 2 + 1 ]; 

        // Walk around a
        bool    hasConstraint   = false; 
        int     nextTri         = oneTri[a]; 
        int     orgVert, destVert; 

        decode(nextTri, otri); 

        int firstTri = otri.tri; 

        do 
        {
            lnext(otri, onexttri); 
            symself(output->triangles, onexttri); 

            if (onexttri.tri < 0) 
                break; 

            lnext(onexttri, otri); 
        } while (otri.tri != firstTri); 

        if (onexttri.tri < 0)
            firstTri = -1; 

        do 
        {
            org(output->triangles, otri, orgVert); 
            dest(output->triangles, otri, destVert); 

            if ( orgVert == b ) 
            {
                lprev(otri, onexttri); 
                edgeIsConstraint[onexttri.tri * 3 + onexttri.orient] = true; 
                hasConstraint = true; 
            }

            if ( destVert == b ) 
            {
                lnext(otri, onexttri); 
                edgeIsConstraint[onexttri.tri * 3 + onexttri.orient] = true; 
                hasConstraint = true; 
            }

            lprevself(otri); 
            symself(output->triangles, otri); 

            if (otri.tri < 0) 
                break; 

            lprevself(otri); 

        } while (otri.tri != firstTri); 

        if (!hasConstraint) 
        {
            printf("\n*** ERROR *** Constraint %i(%i, %i) not found in the triangulation\n", i, a, b); 
        }
    }

    printf("\nChecking incircle property..."); 

    int edgeCount = 0; 

    for (otri.tri = 0; otri.tri < output->nTris; otri.tri++)       
        for (otri.orient = 0; otri.orient < 3; otri.orient++)
        {
            sym(output->triangles, otri, onexttri);	

            if (onexttri.tri < 0)
            {
                edgeCount += 2; 
                continue; 
            } else
                edgeCount++; 

            if (!edgeIsConstraint[otri.tri * 3 + otri.orient])
            {
                int triOrg, triDest, triApex, oppoApex; 

                // current triangle origin, destination and apex
                org(output->triangles, otri, triOrg);
                dest(output->triangles, otri, triDest);
                apex(output->triangles, otri, triApex);				

                // opposite triangle apex
                apex(output->triangles, onexttri, oppoApex);

                if (incircle(&input->points[triOrg], 
                    &input->points[triDest], 
                    &input->points[triApex], 
                    &input->points[oppoApex]) > 0) 
                {
                    printf("\n*** ERROR *** Incircle test fail for triangle %i and %i\n", 
                        otri.tri, onexttri.tri);    
                }
            } // Not a constraint

        } // Loop through 3 edges

        edgeCount /= 2; 

        int euler = input->nPoints - edgeCount + output->nTris + 1; 
		printf("Edgecount = %d\n", edgeCount);

        if (euler != 2)
        {
            printf("\n*** ERROR *** Euler characteristic test fail\n"); 
            printf("V = %i, E = %i, F = %i\n", input->nPoints, edgeCount, output->nTris + 1); 
        }

        printf("\nDONE\n"); 

        delete [] oneTri; 
        delete [] edgeIsConstraint; 
}

//DWORD WINAPI displayAlphaShapes(void* param)
//{	
//	as_global->setAlpha(0);
//	int index = 0;
//	for(MasterList_iterator master_it = as_global->Master_begin(); master_it != as_global->Master_end(); master_it++)
//	{
//		index = master_it - as_global->Master_begin();
//		as_global->setAlpha(index);
//		if(pInput->nPoints < 1000)
//			Sleep(10);
//		else
//			Sleep(0);
//	}
//	index++;
//	as_global->setAlpha(index);
//
//	return 0;
//}

int main(int argc,char **argv)
{   
    printf("GAlpha - A 2D Alpha Shapes library using the GPU\n"); 
    printf("\n\n");

	string fname;
	cout << "Enter input file name: ";
	cin >> fname;
	filename = fname;

	// Read input from a file
	ReadDataFromFileBinary(pInput);

	// GPU-DT setting
	pInput->fboSize = fboSize;
	pOutput         = NULL; 

	// Run GPU-DT	    
	clock_t tv[2];
  

	printf("Running GAlpha...\n");    

	if (pOutput) 
		gpudtReleaseDTOutput(pOutput);

	tv[0] = clock();

	pOutput = gpudtComputeDT(pInput);

	Alpha_Shapes_2D as(pInput, pOutput);
	as.computeAlphaShapes();
	as_global = &as;

	tv[1] = clock();
	
	double sum = (tv[1]-tv[0])/(REAL)CLOCKS_PER_SEC;
	printf("GAlpha time: %.4fs\n", (tv[1]-tv[0])/(REAL)CLOCKS_PER_SEC); 	
	
	checkResult(pInput, pOutput); 
	//as.dumpValues((foldername + "\\" + fname));		

    // Visualization
	// alpha_display = true;
	// Visualizer visualizer(&as);
	// visualizer.initDisplay();
	// visualizer.setScreen();
    printf("\n"); 

    printf("Visualization\n"); 
    printf("  - Left  mouse + Motion : Move\n"); 
    printf("  - Right mouse + Motion : Change alpha index\n"); 
    printf("  - Scroll mous button   : Zoom in and out\n"); 
    
	/*DWORD  threadId;
    HANDLE hThread;
    
    hThread = CreateThread( NULL, 0, displayAlphaShapes, NULL, 
                            0, &threadId );  */  

	// visualizer.start_display();	
	
    /*WaitForSingleObject( hThread, INFINITE );
    CloseHandle( hThread );	  */      

	delete [] pInput->points;
	delete [] pInput->constraints;

	if (pOutput)
		gpudtReleaseDTOutput(pOutput); 	

	/*string outputFileName = "C:\\TestData\\TestsOut\\";
	outputFileName = outputFileName + foldername + "\\AverageTimeGAlpha.txt";	
	
	ofstream fout(outputFileName.c_str());		
	fout << foldername << " " << sum << endl;	
	fout.close();*/

    return 0;
}
