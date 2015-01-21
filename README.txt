GAlpha: A 2D Alpha Shapes using Graphics Hardware (version 1.0)
===============================================================

Authors:
Srinivasan Kidambi Sridharan
Ashwin Nanjappa
Cao Thanh Tung

Copyright (c) 2012 School of Computing, National University of Singapore. 
All rights reserved.

If you use GAlpha and you like it or have comments on its usefulness etc., we 
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

-------------------------------------------------------------------------

GAlpha is a C++ Library that utilizes graphics hardware to compute 2D Alpha Shapes and 
Delaunay triangulation. The result is a triangle mesh, each contain the index of its 3 
vertices and the three neighbor triangles and set of intervals augmented with it to 
represent the whole family of Alpha Shapes.

1. Requirement
==============
- CUDA Toolkit version 4.0 and above. 
- A GPU capable of running CUDA.

By default, GAlpha performs all floating point computation in Double precision. 

2. Tested
=========
GAlpha has been tested on NVIDIA Geforce GTX580, GTX 460.

3. File format
==============
In order to use GAlpha to compute the Alpha Shapes for a set of points, user should provide input data files into GAlpha. 
The input file should contain the number of points followed by coordinates of all the 2D points in the set.

-  .vertex files 

    - First iterm: <# of vertices> 

    - Remaining iterms: <x> <y> (i.e., the coordinates for the points)


Note: If you only want to compute the Delaunay triangulation for a set of points. Before using GPU-DT, you can use the 
"Generator" provided by us to generate the input files, or you can directly provide the input files by yourself to GAlpha. 

4. In this folder
=================
The following files are included in this distribution:
	
	readme.txt		    The file you're reading now
	gpudt.cpp		    The main CPU code of GPU-DT
	gpudt.h		        Header file, include some configurations, 
				        instructions, and some useful macro. 
	*.cu			    CUDA source codes
	

The distribution include a sample Visual Studio 2008 project using GAlpha with CUDA Toolkit 4.0 (32-bit) to compute Alpha Shapes
for a 2D point set. The triangulation is then drawn using OpenGL, and the user can zoom in and pan around the triangle mesh. 

Note: When compiling the CUDA code using Double precision, you have to enable compute capability 2.0 using the switch -sm_20. 

5. Acknowledgements
===================
We acknowledge that the code in predicates.h is extracted from the file predicates.c obtained from the webpage 
http://www.cs.cmu.edu/~quake/robust.html. The code in  cudaCCW.cu is also extracted from the same file, with some minor adjustment
to make it work in CUDA. 

----------------------------------------------------------------------------------
Graphics, Geometry & Games Lab
School of Computing, National University of Singapore
Computing 1
13 Computing Drive
Singapore 117417
Repulic of Singapore
January 2011
----------------------------------------------------------------------------------
Please send bugs and comments to: tants@comp.nus.edu.sg
