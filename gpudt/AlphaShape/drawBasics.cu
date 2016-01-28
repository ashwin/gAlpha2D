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

#include "drawBasics.h"

void drawLine(Color c, gpudtVertex p1, gpudtVertex p2)
{
	glBegin(GL_LINES);
		glColor3f(c.r/255, c.g/255, c.b/255);
		glVertex2f(p1.x, p1.y);
		glVertex2f(p2.x, p2.y);
	glEnd();
}

void drawCircle(Color c, double radius)
{
	glBegin(GL_POLYGON);
		for(int i=0; i<30; i++) {
			glColor3f(c.r/255,c.g/255,c.b/255);
			glVertex2f(radius*sin(2*PI*i/30), radius*cos(2*PI*i/30));
		}
	glEnd();
}

void drawTriangle(gpudtVertex &a, gpudtVertex &b, gpudtVertex &c)
{
	glBegin(GL_LINE_LOOP);		
		glVertex2f(a.x, a.y);
		glVertex2f(b.x, b.y);
		glVertex2f(c.x, c.y);
	glEnd();
}

void fillTriangle(gpudtVertex &a, gpudtVertex &b, gpudtVertex &c, Color col)
{	
	glColor3f(col.r/255,col.g/255,col.b/255);
	glBegin(GL_TRIANGLE_STRIP);		
		glVertex2f(a.x, a.y);
		glVertex2f(b.x, b.y);
		glVertex2f(c.x, c.y);
	glEnd();
}

double round(double x, int r) {
	std::ostringstream strs;
	strs.precision(r);
	strs << x;
	std::istringstream i_strs(strs.str());
	i_strs >> x;
	return x;
}

GLuint FontListID;

void InitText()
{
	HDC hDC;
	HFONT hFont;
	HFONT hOldFont;

	hDC=wglGetCurrentDC();

	FontListID = glGenLists ( 255 );

	hFont = CreateFont(	-14,						// Height Of Font ( NEW )
		0,								
		0,								
		0,								
		FW_NORMAL,						
		FALSE,							
		FALSE,							
		FALSE,							
		ANSI_CHARSET,					
		OUT_TT_PRECIS,					
		CLIP_DEFAULT_PRECIS,			
		PROOF_QUALITY,			
		FF_DONTCARE|DEFAULT_PITCH,		
		"Arial");

	hOldFont = (HFONT) SelectObject(hDC, hFont);
	wglUseFontBitmaps(hDC, 0, 255,  FontListID);	// Builds 96 Characters Starting At Character 32
	DeleteObject(hFont);							// Delete The Font
}

void glPrint(float x, float y, GLfloat red, GLfloat green, GLfloat blue, const char *string, ...)
{
	char	strText[256];
	va_list argumentPtr;
	GLfloat color[4];

	va_start(argumentPtr, string);
	vsprintf_s(strText, sizeof(strText), string, argumentPtr);
	va_end(argumentPtr);

    glPushMatrix();
    glLoadIdentity(); 
    glDisable(GL_TEXTURE_2D);
	
	glGetFloatv(GL_CURRENT_COLOR, color);
	glColor3f(red,green,blue);

		glRasterPos2f(x, y);

	glPushAttrib(GL_LIST_BIT);
	glListBase(FontListID);
	glCallLists(strlen(strText), GL_UNSIGNED_BYTE, strText);
	glPopAttrib();

	glEnable(GL_TEXTURE_2D);
    glPopMatrix();
}

void drawRectangle(double x1, double y1, double x2, double y2)
{
	glBegin(GL_QUADS);
	glVertex2f(x1, y1);
	glVertex2f(x2, y1);
	glVertex2f(x2, y2);
	glVertex2f(x1, y2);
	glEnd();
}

void drawSlider(CoordSystem cs)
{
	double x1, y1, x2, y2;

	// Draw Slider base
	x1 = cs.origin_x + 50/cs.scale;
	y1 = cs.origin_y - 30/cs.scale;
	x2 = cs.origin_x + 650/cs.scale;
	y2 = cs.origin_y - 34/cs.scale;

	drawRectangle(x1,y1,x2,y2);

	// Draw Slider
	x1 = cs.slider_x;
	y1 = cs.slider_y;
	x2 = cs.slider_x + 5/cs.scale;
	y2 = cs.slider_y - 12/cs.scale;

	drawRectangle(x1,y1,x2,y2);
}
