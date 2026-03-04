# 💻 CHEM 284 - Parallelizing Virtual Screening with OpenMP

## 🧪 Goal

The goal of this lab is to:

1. Learn about **OpenMP and molecular docking with a grid**.
2. Learn how to use **OpenMP compiler directives**. 
3. Practice using **exercises including molecular docking**.
4. Profile serial and parallel versions of the code.

---
## 🗂️ Provided

- A `docker` file to set up the dev environment.
- A `CMakeLists.txt` file to help build the executables.
- A `src` directory with the relevant files.

---
## 💻 Setup
```bash
./docker_build.sh # You may need to chmod +x
./docker_run.sh # You may need to chmod +x
```
To build:
```
# From the main directory
mkdir build
cd build
cmake ..
make
```

Note: If you are running on an apple silicone based Macbook (M chips), you will run into an error while building due to line 32 of the CMakeLists.txt. Commenting it out should let the build succeed. This also means that you can't run the SIMD optimized code but you can still examine it to see the differences.

To run the executables
```
# Make sure you are in the build directory
./vector_norm
./serial_docking
./parallel_docking_custom_reduction
./parallel_docking_manual_reduction
./simd_parallel_docking
```

## ✅ Tasks
### Parallelize Vector Norm:
Inside `src/vector_norm.cpp` there is a serial implementation of a vector norm. Your task is to create a parallel version of this code in `norm_parallel()`. Its not as simple as just using a `#pragma omp parallel for` directive on the loop because there is a race condition! There are many ways to go about avoiding the race condition: using `atomic`, `critical` or a `reduction`. Try all 3 and record how long each one takes:

| OpenMP Directive | Time taken |
|------------------|------------|
| critcal          |            |
| atomic           |            |
| reduction        |            |

Make sure your parallel solution has the same answer as the serial!
Results using reduction below on 16 threads:
```
root@f66324f58549:/repo/build# ./vector_norm
Serial norm    = 7071.07
Parallel norm  = 7071.07

Serial time    = 0.297196 s
Parallel time  = 0.0353333 s
Speedup        = 8.41121x
```

### Parallelize Molecular Docking:
Now as we've seen before with MPI we will be parallelizing a "virtual screen". Except we will be using different poses of the same ligand. In this example we will be using a precomputed grid, so there is no need for a receptor or calculating distances or even the LJ potential! The code provided to you uses [trilinear interpolation](https://en.wikipedia.org/wiki/Trilinear_interpolation) to determine the value of each atom in the grid based on the sampled points. This dramatically speeds up how fast we can calculate the LJ potential for each pose (we do lose some accuracy though). The serial code is provided to you, your task is to parallelize the pose energy evaluation for `1,000,000` poses. There are 2 main methods to do this:

1) Manual: you will need to keep track of the local minimum energy and local best index for each thread, so you will need to create `#pragma omp parallel` region so that each thread has its own private variables before running your parallelized loop over the poses. Then once all your threads have finished and each have their best local minimum energy and best local index, you will manually reduce to find the global minimum energy and best local index.
2) Custom reduction: you will need to create a custom reduction using `#pragma omp declare reduction`. This reduction will reduce using the given `struct BestResult`. Then you can use `#pragma omp parallel for reduction(reduction_name:variable_name)` to handle the reduction for you.

Do both methods and compare the speeds:

| Method           | Time taken |
|------------------|------------|
| manual reduction |            |
| custom reduction |            |

Make sure your parallel solution has the same answer as the serial!

```
root@f66324f58549:/repo/build# ./serial_docking
Interpolated value = -6.38924
Best pose = 306172
Time taken = 1.1498

root@f66324f58549:/repo/build# ./parallel_docking_manual_reduction
Interpolated value = -6.38924
Best pose = 306172
Time taken = 0.146975

root@f66324f58549:/repo/build# ./parallel_docking_custom_reduction
Interpolated value = -6.38924
Best pose = 306172
Time taken = 0.166902
```

### OpenMP SIMD
Looking at the parallelized code you are probably wondering if we can parallelize the inner most loop over all the atoms to go even faster? And the answer to that is yes we can! But it requires refactorting the data so that it is vectorizable. In `simd_parallel_docking.cpp` this has been done for you, and you can compare how fast this code is to `serial_docking.cpp`, `parallel_docking_manual_reduction.cpp`, and `parallel_docking_custom_reduction.cpp`. What did we need to change to the code to allow for us vectorize the instructions? Look at `include/simd.h` and `src/simd_parallel_docking.cpp`.

```
root@f66324f58549:/repo/build# ./simd_parallel_docking
Interpolated value = -6.38924
Best pose = 306172
Time taken = 0.0329537
```

### Extra time
If we increased the size of our grid by sampling more points our accuracy would improve but how would that effect the speed of our parallelized code? Especially the SIMD code?
