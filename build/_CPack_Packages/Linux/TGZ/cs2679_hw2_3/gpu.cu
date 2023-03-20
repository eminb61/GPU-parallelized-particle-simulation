#include "common.h"
#include <cuda.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <vector>
#include <stdio.h>

#define BINSIZE (cutoff * 2.1)
#define NUM_THREADS 256
#define MIN(x,y) (((x)<(y))?(x):(y))

// Put any static global variables here that you will use throughout the simulation.
int blks;
int dim;
int *bins_array, *bin_count;
int *bin_count_mem;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;

    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* parts,int *bins_array, int *bin_count, int dim){

    int directions[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    int bin_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(bin_id >= dim * dim) return;

    int bin_i = bin_id / dim;
    int bin_j = bin_id % dim;

    for(int d = 0; d < 9; d++){
        int neigh_i = bin_i + directions[d][0];
        int neigh_j = bin_j + directions[d][1];
        if(neigh_i < 0 or neigh_i >= dim or neigh_j < 0 or neigh_j >= dim)
            continue;
        int neigh_id = neigh_i * dim + neigh_j;

        for(int i = bin_count[bin_id]; i < bin_count[bin_id+1]; ++i){
            particle_t &p1 = parts[bins_array[i]];
            for(int j = bin_count[neigh_id]; j < bin_count[neigh_id+1]; ++j){
                particle_t &p2 = parts[bins_array[j]];
                apply_force_gpu(p1, p2);
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_parts)
        return;

    particle_t* p = &particles[idx];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    dim = floor(size / BINSIZE) + 1;

    // initialize bins_array and bin_count
    cudaMalloc((void **)&bins_array, num_parts * sizeof(int));
    cudaMalloc((void **)&bin_count, (dim * dim + 1) * sizeof(int));
    cudaMalloc((void **)&bin_count_mem, (dim * dim + 1) * sizeof(int));

}

__global__ void count_part_per_bin_gpu(particle_t *parts, int *bin_count, int num_parts, int dim){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= num_parts) return;
    int bin_id = floor(parts[tid].x / BINSIZE) * dim + floor(parts[tid].y / BINSIZE);

    // increase the number of particles of the bin
    atomicAdd(&bin_count[bin_id], 1);
}

__global__ void add_particle_to_bin_gpu(particle_t *parts, int *bins_array, int *bin_count, int num_parts, int dim){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= num_parts) return;
    int bin_id = floor(parts[tid].x / BINSIZE) * dim + floor(parts[tid].y / BINSIZE);

    // get target bin index
    int tar_bin_id = atomicAdd(&bin_count[bin_id], 1);

    // add particle to target bin
    bins_array[tar_bin_id] = tid;

    // reset ax, ay of the particle
    parts[tid].ax = parts[tid].ay = 0;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    // Reset the bin_count to 0
    thrust::device_ptr<int> counts1(bin_count);
    thrust::fill(counts1, counts1 + dim * dim + 1, 0);

    // Count number of particles per bin
    count_part_per_bin_gpu<<<blks, NUM_THREADS>>>(parts, bin_count + 1, num_parts, dim);

    // Prefix sum the bin counts
    thrust::device_ptr<int> counts2(bin_count);
    thrust::inclusive_scan(counts2, counts2 + dim * dim + 1, counts2);

    // Add particles to separate array starting from bin idx
    cudaMemcpy(bin_count_mem, bin_count, (dim*dim + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    add_particle_to_bin_gpu<<<blks, NUM_THREADS>>>(parts, bins_array, bin_count_mem, num_parts, dim);

    // Each thread will be responsible for one bin
    compute_forces_gpu<<<(dim * dim + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(parts, bins_array, bin_count, dim);

    // Each thread move the particle
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}