## Some Changes for Linux

* Rename macro function `encode()` as `Encode()` in file `gpudt.h` and `main.cu`.
* The Encoding of the following file is changed to UTF-8: `main.cu`, `cudaConstraint.cu`, `cudaMissing.cu`, `cudaVoronoi.cu`.
* Visualisation functions has been suspended.
* Mild changes on `#Include` in `main.cu`.
* Mild changes on `#Include` in `kernel.h`.

## Compile

`CMakeList.txt` is added. 

```
mkdir build
cd build
cmake ..
make
```

