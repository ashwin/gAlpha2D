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

#include "Visualizer.h"
#include <map>

extern bool alpha_display;

GLfloat PI = 3.14159;
CoordSystem cs;

void setParams(int x, int y)
{
	if(x == cs.old_x && y == cs.old_y)
		return;

	cs.old_x = x;
	cs.old_y = y;

	cs.init_x = cs.cur_x;	
	cs.init_y = cs.cur_y;
	
	cs.cur_x = (x / cs.scale) + cs.origin_x;
	cs.cur_y = ( -y / cs.scale ) + (cs.origin_y);
}

void Visualizer::setScreen()
{
	
	double worldFrameWidth = (double)(Visualizer::as->pInput->maxX - Visualizer::as->pInput->minX);
	cs.scale = 700.0 / (3*worldFrameWidth);
	cs.origin_x = Visualizer::as->pInput->minX - (worldFrameWidth);
	cs.origin_y = Visualizer::as->pInput->minY + (1.5*worldFrameWidth);
}

void panning()
{
	double dispmt_x = (cs.cur_x - cs.init_x);
	double dispmt_y = (cs.cur_y - cs.init_y);


	cs.origin_x -= dispmt_x;
	cs.origin_y -= dispmt_y;

	cs.cur_x -= dispmt_x;
	cs.cur_y -= dispmt_y;
	cs.init_x -= dispmt_x;
	cs.init_y -= dispmt_y;

	cs.slider_x -= dispmt_x;
	cs.slider_y -= dispmt_y;
	cs.slider_prev_x = cs.slider_x;
}

Alpha_Shapes_2D* Visualizer::as = NULL;
double Visualizer::alpha = 0;
vector<Color> Visualizer::colors;

Visualizer::Visualizer(Alpha_Shapes_2D* a_s)
{	
	Visualizer::as = a_s;

	srand ( time(NULL) );
	for(int i=0; i<a_s->pDT->nTris; i++)
	{
		double r = (rand()*255/RAND_MAX), g = ((rand()+25)*255/RAND_MAX), b = (rand()*255/RAND_MAX);
		Color c;
		c.set(r,g,b);
		Visualizer::colors.push_back(c);
	}
	
	cs.init_x = 0;
	cs.init_y = 0;
	cs.cur_x = 0;
	cs.cur_y = 0;

	cs.origin_x = -20;
	cs.origin_y = 20;
	cs.w = 700.0;
	cs.h = 700.0;
	cs.scale = 700.0/40.0;

	cs.old_x = 0;
	cs.old_y = 0;	
	
	cs.right_button = 0;
}

void reshape (int w, int h)
{	
	h = (h == 0) ? 1 : h;
	w = (w == 0) ? 1 : w;	

	cs.w = w;
	cs.h = h;
	glViewport (0, 0, (GLsizei) w, (GLsizei) h);
}

void init(void)
{
	glClearColor (0.0, 0.0, 0.0, 1.0);
	glShadeModel (GL_SMOOTH);
}

void mouse (int x, int y)
{			
	setParams(x,y);
	if( cs.right_button == 0 )
		panning();
	else
	{		
		double cur_slider_x = cs.origin_x + x/cs.scale;	
		if( cur_slider_x > cs.origin_x + 640/cs.scale || cur_slider_x < cs.origin_x + 60/cs.scale)
			return;

		cs.slider_prev_x = cs.slider_x;
		cs.slider_x = cur_slider_x;
		int index = (cs.slider_x - (cs.origin_x + 59/cs.scale)) / cs.increment;
		if( index >= Visualizer::as->master_list.size() )
			index = Visualizer::as->master_list.size()-1;
		Visualizer::as->alpha_index = index;
	}
}

void passive_mouse (int x, int y) {
	setParams(x,y);	
}

void keyboard (unsigned char key, int x, int y)
{
	/*if(key == 's' || key == 'S')
		Visualizer::as->dumpValues();*/
}

void Visualizer::initDisplay()
{
	glutInit(&__argc,__argv);
	glutInitDisplayMode (GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize (700, 700);
	glutInitWindowPosition (50, 0);
	glutCreateWindow ("GAlpha");
	init ();	
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);

	// Mouse Event Handlers
	glutMotionFunc(mouse);
	glutPassiveMotionFunc(passive_mouse);
	glutMouseFunc(mouse_press);	
	glutKeyboardFunc(keyboard);

	InitText();
}

void simulate_right_click(double curx, double cury)
{	
}

void mouse_press(int button, int state, int x, int y)
{	
	setParams(x,y);

	if(button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		cs.right_button = 1;
		simulate_right_click(cs.cur_x, cs.cur_y);			
	}
	else if(button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {
		cs.right_button = 0;					
	}
	else if(button == 3 || button == 4) {
		double k = cs.scale, add = 0.1;
		//double diff = (cs.slider_x - cs.origin_x)*cs.scale;
		//double old_scale = cs.scale;

		if ( button == 3 )		
			cs.scale += (add * cs.scale);		
		else if( button == 4 && cs.scale > (add*cs.scale))
			cs.scale -= (add * cs.scale);

		double dx = (cs.cur_x - cs.origin_x);
		double dx_ = ( dx * k / cs.scale);
		cs.origin_x = cs.cur_x - dx_;

		double dy = (cs.cur_y - cs.origin_y);
		double dy_ = ( dy * k / cs.scale);
		cs.origin_y = cs.cur_y - dy_;

		cs.increment = 580/cs.scale;
		cs.increment /= Visualizer::as->master_list.size();

		//double prev = cs.slider_x;
		cs.slider_x = (cs.origin_x + 60/cs.scale) + cs.increment * Visualizer::as->alpha_index;
		cs.slider_y = cs.origin_y - 26/cs.scale;
		cs.slider_prev_x = cs.slider_x;		
	}
}

void Color::set(double red, double green, double blue) {
	r = red;
	g = green;
	b = blue;
}

void drawAxis()
{
	gpudtVertex p1, p2;
	Color c;
	c.set(255,0,0);

	p1.x = cs.origin_x;
	p1.y = 0;
	p2.x = cs.origin_x + (cs.w/cs.scale);
	p2.y = 0;
	drawLine(c, p1, p2);

	p1.x = 0;
	p1.y = cs.origin_y;
	p2.x = 0;
	p2.y = cs.origin_y - (cs.h/cs.scale);
	drawLine(c, p1, p2);
}

void display(void)
{
	double dx = cs.w / cs.scale;
	double dy = cs.h / cs.scale;

	glMatrixMode (GL_PROJECTION);
	glLoadIdentity();
	glOrtho(cs.origin_x, cs.origin_x + dx, cs.origin_y - dy, cs.origin_y, -10, 10);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	Color c = {0.0,0.0,0.0};	
	glClearColor(c.r,c.g,c.b,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);			

	glPushMatrix();		

		drawAxis();
		//draw your stuff here
		c.set(255,255,255);
		
		//while(!Visualizer::dt->lock()) { Sleep(0); };

		if(!alpha_display)
		{
			int size = Visualizer::as->pDT->nTris;
			gpudtTriangle tri;		

			for(int i=0; i<size; i++)
			{
				tri = Visualizer::as->pDT->triangles[i];
				glColor3f(1,1,1);
				drawTriangle(Visualizer::as->pInput->points[tri.vtx[0]], Visualizer::as->pInput->points[tri.vtx[1]], Visualizer::as->pInput->points[tri.vtx[2]]);
			}
			Visualizer::as->setAlpha(0);
		}
		else
		{
			int size = Visualizer::as->master_list.size();
			map<int,int> can_draw;
			
			for(int i=Visualizer::as->alpha_index; i>=0; i--)
			{				
					simplex_type st = Visualizer::as->master_list[i].st;
					if(st == EDGE && Visualizer::as->master_list[i].at != nu_max)
					{
						//cout << "Reaching draw edge\n";
						int edge_index = Visualizer::as->master_list[i].simplex_index;
						edge_interval edge = Visualizer::as->edge_list[edge_index];
						int p1Index = Visualizer::as->pDT->triangles[edge.tIndex].vtx[((edge.vOpp+1)%3)];
						int p2Index = Visualizer::as->pDT->triangles[edge.tIndex].vtx[((edge.vOpp+2)%3)];
						gpudtVertex p1 = Visualizer::as->pInput->points[p1Index];
						gpudtVertex p2 = Visualizer::as->pInput->points[p2Index];

						if( can_draw.count(edge_index) == 0 )
						{
							Color c;
							c.set(230,210,30);						
							drawLine(c, p1,p2);
						}
						/*else
						{
							Color c;
							c.set(255,255,255);						
							drawLine(c, p1,p2);
						}*/
					}
					else if(st == EDGE && Visualizer::as->master_list[i].at == nu_max)
					{
						int edge_index = Visualizer::as->master_list[i].simplex_index;
						//edge_interval edge = Visualizer::as->edge_list[edge_index];
						//int p1Index = Visualizer::as->pDT->triangles[edge.tIndex].vtx[((edge.vOpp+1)%3)];
						//int p2Index = Visualizer::as->pDT->triangles[edge.tIndex].vtx[((edge.vOpp+2)%3)];						
		
						can_draw[edge_index] = 1;
					}
					if(st == TRIANGLE)
					{
						int face_index = Visualizer::as->master_list[i].simplex_index;
						/*if(!Visualizer::dt->checkTri(face_index))
							continue;*/

						gpudtTriangle& t = Visualizer::as->pDT->triangles[face_index];
						gpudtVertex p1 = Visualizer::as->pInput->points[t.vtx[0]];
						gpudtVertex p2 = Visualizer::as->pInput->points[t.vtx[1]];
						gpudtVertex p3 = Visualizer::as->pInput->points[t.vtx[2]];
												
						fillTriangle(p1,p2,p3,Visualizer::colors[face_index]);
					}
			}
			can_draw.clear();
		}
		
		glColor3f(1,1,1);
		glPointSize(1.5f);
		glBegin(GL_POINTS);
		for(int i = 0; i < Visualizer::as->pInput->nPoints; i++)
			glVertex2f(Visualizer::as->pInput->points[i].x, Visualizer::as->pInput->points[i].y);
		glEnd();
		
		//Visualizer::dt->unlock();	
	
	glPopMatrix();	

	ostringstream l_x, l_y, al;
	l_x << cs.cur_x;
	l_y << cs.cur_y;
	string coords = "(";
	coords = coords + l_x.str() + ",";
	glPrint(cs.cur_x + 15/cs.scale, cs.cur_y, 1.0f, 1.0f, 1.0f, coords.c_str());
	coords =  l_y.str() + ")";
	glPrint(cs.cur_x + 15/cs.scale, cs.cur_y-15/(cs.scale), 1.0f, 1.0f, 1.0f, coords.c_str());
	
	al << Visualizer::as->alpha_index;
	coords = "Alpha = " + al.str();
	glPrint(cs.origin_x + 15/cs.scale, cs.origin_y - 15/cs.scale, 1.0f, 1.0f, 1.0f, coords.c_str());	

	drawSlider(cs);

	glFlush ();
	Sleep(10);	
	
	glutPostRedisplay();	
}

void Visualizer::start_display()
{
	cs.slider_x = cs.origin_x + 55/cs.scale;
	cs.slider_y = cs.origin_y - 26/cs.scale;
	cs.slider_prev_x = cs.slider_x;
	cs.increment = 580/cs.scale;
	cs.increment = cs.increment / (double)Visualizer::as->master_list.size();
	Visualizer::as->alpha_index = 0;

	printf("Length = %lf, dx = %lf, master size = %d\n", 590/cs.scale, cs.increment, Visualizer::as->master_list.size());

	glutMainLoop();
}