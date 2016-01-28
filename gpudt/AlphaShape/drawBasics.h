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

#include <cmath>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <windows.h>
#include <conio.h>
#include <stdio.h>
#include <sstream>
#include "..\..\include\GL\glut.h"
#include "Alpha_Shapes_2D.h"

extern int MAXX;
extern int MAXY;

using namespace std;

extern GLfloat PI;

struct Color {
	double r,g,b;

	void set(double red, double green, double blue);	
};

struct CoordSystem {
	double origin_x, origin_y;
	double scale, w, h;
	
	double init_x, init_y, cur_x, cur_y;
	int old_x, old_y;

	double slider_x, slider_y;
	double slider_prev_x, slider_prev_y;
	int right_button;

	double increment;
};

void fillTriangle(gpudtVertex &a, gpudtVertex &b, gpudtVertex &c, Color col);
void drawLine(Color c, gpudtVertex p1, gpudtVertex p2);
void drawCircle(Color c, double radius);
void drawTriangle(gpudtVertex &a, gpudtVertex &b, gpudtVertex &c);
double round(double x, int r);

void glPrint(float x, float y, GLfloat red, GLfloat green, GLfloat blue, const char *string, ...);
void InitText();

void drawSlider(CoordSystem cs);