//
// Created by ilya on 28.02.2021.
//

#include "Physics.hpp"
#include <float.h>

__device__ bool intersect(const Body &l, const Body &r) {
    float X = l.x - r.x, Y = l.y - r.y, R = r.radius + l.radius;
    return (X * X + Y * Y - R * R) < 0.f;
}

__device__ void merge(Body *body_vec, Body *result, int bodies_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Body current_body = body_vec[idx];
    bool delete_this = false;

    if (idx > bodies_count)
        return;

    if (current_body.mass == 0.f)
        return;

    for (int i = 0; i < idx; i++) {
        if (intersect(current_body, body_vec[i])) {
            delete_this = true;
            break;
        }
    }

    if (!delete_this) {
        for (int i = idx + 1; i < bodies_count; i++) {
            auto tmp = body_vec[i];

            if (tmp.mass == 0.f)
                continue;

            if (intersect(current_body, tmp)) {
                current_body.x = current_body.radius > tmp.radius ? current_body.x : tmp.x;
                current_body.y = current_body.radius > tmp.radius ? current_body.y : tmp.y;

                current_body.radius = sqrt(current_body.radius * current_body.radius + tmp.radius * tmp.radius);

                auto mass_sum = tmp.mass + current_body.mass;
                current_body.speed_x =
                        current_body.speed_x * (current_body.mass / mass_sum) + (tmp.speed_x) * (tmp.mass / mass_sum);
                current_body.speed_y =
                        current_body.speed_y * (current_body.mass / mass_sum) + (tmp.speed_y) * (tmp.mass / mass_sum);

                current_body.mass = mass_sum;
            }
        }
    } else {
        current_body.mass = 0.f;
        current_body.radius = 0.f;
    }

    result[idx] = current_body;
}

__device__ void update_positions(Body *body_vec, Body *result, size_t bodies_count) {
    constexpr auto step = 0.2f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > bodies_count)
        return;

    float Fx = 0., Fy = 0.;

    Body &current_body = body_vec[idx];
    for (int i = 0; i < bodies_count; i++) {
        auto tmp = body_vec[i];

        if (idx == i || tmp.mass == 0.f)
            continue;

        auto X = tmp.x - current_body.x, Y = tmp.y - current_body.y;
        auto D2 = (X * X + Y * Y);
        auto F = 6.674184 * 10e-9 * (current_body.mass * tmp.mass) / D2;
        auto D = sqrt(D2);

        Fx += F * X / D;
        Fy += F * Y / D;
    }

    result[idx].x += current_body.speed_x * step;
    result[idx].y += current_body.speed_y * step;
    result[idx].speed_x += Fx * step / current_body.mass;
    result[idx].speed_y += Fy * step / current_body.mass;
}

__global__ void update_gpu_bodies(Body *body_vec, Body *result, int bodies_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    merge(body_vec, result, bodies_count);

    __syncthreads();

    body_vec[idx] = result[idx];
    update_positions(body_vec, result, bodies_count);
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

    // TMP
    Body *gpu_result;
    cudaMalloc(&gpu_result, sizeof(Body) * bodies.size());

    cudaMemcpy(gpu_result, gpu_bodies_vec_, sizeof(Body) * bodies.size(), cudaMemcpyDeviceToDevice);
    update_gpu_bodies<<<blocks, threads>>>(gpu_bodies_vec_, gpu_result, bodies.size());
    cudaMemcpy(bodies.data(), gpu_result, sizeof(Body) * bodies.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_bodies_vec_, gpu_result, sizeof(Body) * bodies.size(), cudaMemcpyDeviceToDevice);

    // TMP
    cudaFree(gpu_result);
}
