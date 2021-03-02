//
// Created by ilya on 28.02.2021.
//

#include "Physics.hpp"
#include <float.h>

__device__ bool intersect(const Body &l, const Body &r) {
    float X = l.x - r.x, Y = l.y - r.y, R = r.radius + l.radius;
    return (X * X + Y * Y - R * R) < 0.f;
}

__device__ void merge(Body *body_vec, int bodies_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Body tmp = body_vec[idx];

    if (idx > bodies_count)
        return;

    if (tmp.mass == 0.f)
        return;

    // ERROR IN MULTICOLLISION !!!!
    for (int i = idx + 1; i < bodies_count; i++) {
        if (body_vec[i].mass == 0.f)
            continue;

        if (intersect(tmp, body_vec[i])) {
            tmp.x = tmp.radius > body_vec[i].radius ? tmp.x : body_vec[i].x;
            tmp.y = tmp.radius > body_vec[i].radius ? tmp.y : body_vec[i].y;

            tmp.radius = sqrt(tmp.radius * tmp.radius + body_vec[i].radius * body_vec[i].radius);

            auto mass_sum = body_vec[i].mass + tmp.mass;
            tmp.speed_x = tmp.speed_x * (tmp.mass / mass_sum) + (body_vec[i].speed_x) * (body_vec[i].mass / mass_sum);
            tmp.speed_y = tmp.speed_y * (tmp.mass / mass_sum) + (body_vec[i].speed_y) * (body_vec[i].mass / mass_sum);

            tmp.mass = mass_sum;

            body_vec[i].mass = 0.f;
            body_vec[i].radius = 0.f;
        }
    }

    __syncthreads();

    if (body_vec[idx].mass != 0.f)
        body_vec[idx] = tmp;

    __syncthreads();
}

__device__ void update_positions(Body *body_vec, size_t bodies_count) {
    constexpr auto step = 0.2f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > bodies_count)
        return;

    float Fx = 0., Fy = 0.;

    Body &tmp = body_vec[idx];
    for (int i = 0; i < bodies_count; i++) {
        if (idx == i || body_vec[i].mass == 0.f)
            continue;

        auto X = body_vec[i].x - tmp.x, Y = body_vec[i].y - tmp.y;
        auto D2 = (X * X + Y * Y);
        auto F = 6.674184 * 10e-9 * (tmp.mass * body_vec[i].mass) / D2;
        auto D = sqrt(D2);

        Fx += F * X / D;
        Fy += F * Y / D;
    }

    body_vec[idx].x += tmp.speed_x * step;
    body_vec[idx].y += tmp.speed_y * step;
    body_vec[idx].speed_x += Fx * step / tmp.mass;
    body_vec[idx].speed_y += Fy * step / tmp.mass;
}

__global__ void update_gpu_bodies(Body *body_vec, int bodies_count) {
    merge(body_vec, bodies_count);
    update_positions(body_vec, bodies_count);
}

Physics::Physics() : gpu_bodies_vec_(nullptr) {}

Physics::~Physics() {
    if (gpu_bodies_vec_)
        cudaFree(gpu_bodies_vec_);
};

void Physics::load(const std::vector<Body> &bodies) {
    if (gpu_bodies_vec_)
        cudaFree(gpu_bodies_vec_);

    cudaMalloc(&gpu_bodies_vec_, sizeof(Body) * bodies.size());
    // cuda Malloc Host
    cudaMemcpy(gpu_bodies_vec_, bodies.data(), sizeof(Body) * bodies.size(), cudaMemcpyHostToDevice);
}

void Physics::update(std::vector<Body> &bodies) {
    dim3 threads = dim3(64, 1);
    dim3 blocks = bodies.size() / threads.x != 0 ? dim3((bodies.size() / threads.x) + 1, 1) : dim3(1, 1);

//    cudaMemcpy(gpu_bodies_vec_, bodies.data(), sizeof(Body) * bodies.size(), cudaMemcpyHostToDevice);
    update_gpu_bodies<<<blocks, threads>>>(gpu_bodies_vec_, bodies.size());
    cudaMemcpy(bodies.data(), gpu_bodies_vec_, sizeof(Body) * bodies.size(), cudaMemcpyDeviceToHost);
}
