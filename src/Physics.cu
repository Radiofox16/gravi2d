//
// Created by ilya on 28.02.2021.
//

#include "Physics.hpp"
#include <algorithm>
#include <limits>
#include <iostream>

constexpr int THREAD_WARP_SIZE = 32;

__device__ bool intersect(const Body &l, const Body &r) {
    float X = l.x - r.x, Y = l.y - r.y, R = r.radius + l.radius;
    return (X * X + Y * Y - R * R) < 0.f;
}

__device__ void merge_two_bodies(Body &a, const Body &b) {
    if (a.mass == 0.f || b.mass == 0.f)
        return;

    a.x = a.radius > b.radius ? a.x : b.x;
    a.y = a.radius > b.radius ? a.y : b.y;

    a.radius = sqrt(a.radius * a.radius + b.radius * b.radius);

    auto mass_sum = b.mass + a.mass;
    a.speed_x = a.speed_x * (a.mass / mass_sum) + (b.speed_x) * (b.mass / mass_sum);
    a.speed_y = a.speed_y * (a.mass / mass_sum) + (b.speed_y) * (b.mass / mass_sum);

    a.mass = mass_sum;
}

__device__ void merge(const Body *body_vec_in, Body *body_vec_out, int bodies_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ Body cached_bodies[THREAD_WARP_SIZE];
    Body current_body = body_vec_in[idx];

    cached_bodies[threadIdx.x] = idx < bodies_count ? current_body : Body{0, 0, 0, 0, 0, 0, 0, idx, -1000};

    __syncthreads();

    for (int i = 0; i < THREAD_WARP_SIZE; i++) {
        if (threadIdx.x == i || current_body.mass == 0.f) continue;
        if (cached_bodies[i].mass == 0.f) continue;

        if (intersect(current_body, cached_bodies[i])) {
            if (threadIdx.x > i) {
                current_body = {0, 0, 0, 0, 0, 0, 0, idx, blockIdx.x * blockDim.x + i};
            } else {
                merge_two_bodies(current_body, cached_bodies[i]);
                current_body.dbg_1 = idx;
                current_body.dbg_2 = blockIdx.x * blockDim.x + i;
            }
        }
    }

    __syncthreads();

    for (int i = 0; i < blockIdx.x; i++) {
        cached_bodies[threadIdx.x] = body_vec_in[i * blockDim.x + threadIdx.x];
        __syncthreads();

        if (current_body.mass == 0.f) continue;

        for (int j = 0; j < THREAD_WARP_SIZE; j++) {
            if (cached_bodies[j].mass == 0.f)
                continue;

            if (intersect(current_body, cached_bodies[j])) {
                current_body.mass = 0.f;
            }
        }
    }

    for (int i = blockIdx.x + 1; i < gridDim.x; i++) {
        if (i * blockDim.x + threadIdx.x > bodies_count) {
            cached_bodies[threadIdx.x] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        } else {
            cached_bodies[threadIdx.x] = body_vec_in[i * blockDim.x + threadIdx.x];
        }

        __syncthreads();

        if (current_body.mass == 0.f) continue;

        for (int j = 0; j < THREAD_WARP_SIZE; j++) {
            if (cached_bodies[j].mass == 0.f)
                continue;

            if (intersect(current_body, cached_bodies[j])) {
                merge_two_bodies(current_body, cached_bodies[j]);
            }
        }
    }

    body_vec_out[idx] = current_body;
}

__device__ void update_positions(const Body *body_vec_in, Body *body_vec_out, int bodies_count) {
    constexpr auto step = 1.f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > bodies_count)
        return;

    float Fx = 0., Fy = 0.;

    Body current_body = body_vec_out[idx];
    for (int i = 0; i < bodies_count; i++) {
        auto tmp = body_vec_in[i];
    
        if (current_body.mass == 0.f)
            continue;
        
        if (idx == i || tmp.mass == 0.f)
            continue;

        auto X = tmp.x - current_body.x, Y = tmp.y - current_body.y;
        auto D2 = (X * X + Y * Y);
        auto F = 6.674184 * 10e-9 * static_cast<double >(current_body.mass * tmp.mass) / D2;
        auto D = sqrt(D2);

        if (D < (tmp.radius + current_body.radius))
            continue;

        Fx += F * X / D;
        Fy += F * Y / D;
    }

    current_body.x += current_body.speed_x * step;
    current_body.y += current_body.speed_y * step;
    current_body.speed_x += Fx * step / current_body.mass;
    current_body.speed_y += Fy * step / current_body.mass;

    body_vec_out[idx] = current_body;
}

__global__ void update_gpu_bodies(const Body *body_vec_in, Body *body_vec_out, int bodies_count) {
    merge(body_vec_in, body_vec_out, bodies_count);
    update_positions(body_vec_in, body_vec_out, bodies_count);
}

Physics::Physics() : gpu_bodies_vec_a_(nullptr) {}

Physics::~Physics() {
    if (gpu_bodies_vec_a_){
        cudaFree(gpu_bodies_vec_a_);
        cudaFree(gpu_bodies_vec_b_);
    }
}

void Physics::load(std::vector<Body> &bodies) {
    if (gpu_bodies_vec_a_){
        cudaFree(gpu_bodies_vec_a_);
        cudaFree(gpu_bodies_vec_b_);
    }

    float min_X, min_Y, max_X, max_Y;
    max_X = max_Y = std::numeric_limits<float>::lowest();
    min_X = min_Y = std::numeric_limits<float>::max();

    for (const Body &b : bodies) {
        if (min_X > b.x)
            min_X = b.x;
        else if (max_X < b.x)
            max_X = b.x;

        if (min_Y > b.y)
            min_Y = b.y;
        else if (max_Y < b.y)
            max_Y = b.y;
    }

    std::sort(std::begin(bodies), std::end(bodies), [min_X, min_Y, max_X, max_Y](Body &a, Body &b) {
        auto diff = (b.y - a.y) * 2 + b.x - a.x;
        if (b.y < a.y) diff -= 10;
        return diff > 0;
    });

    cudaMalloc(&gpu_bodies_vec_a_, sizeof(Body) * bodies.size());
    cudaMalloc(&gpu_bodies_vec_b_, sizeof(Body) * bodies.size());
    // cuda Malloc Host
    cudaMemcpy(gpu_bodies_vec_a_, bodies.data(), sizeof(Body) * bodies.size(), cudaMemcpyHostToDevice);
}

void Physics::update(std::vector<Body> &bodies) {
    if (!bodies.empty()) {
        std::vector<Body> tmp;
        tmp.reserve(((bodies.size() / THREAD_WARP_SIZE) + 1) * THREAD_WARP_SIZE);

        for (const auto &b: bodies) {
            if (b.mass != 0.f)
                tmp.push_back(b);
        }

//        auto sz_diff = bodies.size() - tmp.size();
//        for (int i = 0; i < sz_diff; i++) {
//            tmp.push_back({});
//        }

        bodies = tmp;
    }

    dim3 threads = dim3(THREAD_WARP_SIZE, 1);
    dim3 blocks = bodies.size() / threads.x != 0 ? dim3((bodies.size() / threads.x) + 1, 1) : dim3(1, 1);

    cudaMemcpy(gpu_bodies_vec_a_, bodies.data(), sizeof(Body) * bodies.size(), cudaMemcpyHostToDevice);
    update_gpu_bodies<<<blocks, threads>>>(gpu_bodies_vec_a_, gpu_bodies_vec_b_, bodies.size());
    cudaMemcpy(bodies.data(), gpu_bodies_vec_b_, sizeof(Body) * bodies.size(), cudaMemcpyDeviceToHost);

//    float rad_sz = 0.f;
//    for (auto &a: bodies) {
//        rad_sz += a.radius;
//        if (a.dbg_1 == 0 && a.dbg_2 == 0) continue;
//        std::cout << "dbg_1: " << a.dbg_1;
//        std::cout << " | dbg_2: " << a.dbg_2 << '\n';
//        a.dbg_1 = 0;
//        a.dbg_2 = 0;
//
//    }
//    std::cout << "------------------ sz: " << bodies.size() << "------------------\n";
}
