//
// Created by ilya on 28.02.2021.
//

#include "Physics.hpp"
#include <algorithm>
#include <limits>

constexpr int THREAD_WARP_SIZE = 32;

__device__ bool intersect(const Body &l, const Body &r) {
    float X = l.x - r.x, Y = l.y - r.y, R = r.radius + l.radius;
    return (X * X + Y * Y - R * R) < 0.f;
}

__device__ void merge_two_bodies(Body &a, const Body &b) {
    a.x = a.radius > b.radius ? a.x : b.x;
    a.y = a.radius > b.radius ? a.y : b.y;

    a.radius = sqrt(a.radius * a.radius + b.radius * b.radius);

    auto mass_sum = b.mass + a.mass;
    a.speed_x =
            a.speed_x * (a.mass / mass_sum) + (b.speed_x) * (b.mass / mass_sum);
    a.speed_y =
            a.speed_y * (a.mass / mass_sum) + (b.speed_y) * (b.mass / mass_sum);

    a.mass = mass_sum;
}

__device__ void merge(Body *body_vec, int bodies_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > bodies_count)
        return;

    __shared__ Body cached_bodies[THREAD_WARP_SIZE];
    Body current_body = body_vec[idx];
    cached_bodies[threadIdx.x] = body_vec[idx];
    __syncwarp();

    for (int i = 0; i < THREAD_WARP_SIZE; i++) {
        if (intersect(current_body, cached_bodies[i])) {
            if (current_body.id < cached_bodies[i].id)
                current_body.mass = 0.f;
            else
                merge_two_bodies(current_body, cached_bodies[i]);
        }
    }

    for (int i = blockIdx.x + 1; i < gridDim.x; i++) {
        cached_bodies[threadIdx.x] = body_vec[i * blockDim.x + threadIdx.x];
        __syncwarp();

        for (int j = 0; j < THREAD_WARP_SIZE; j++) {
            auto tmp = cached_bodies[j];
            if (tmp.mass == 0.f)
                continue;

            if (intersect(current_body, tmp)) {
                if (current_body.id < cached_bodies[i].id)
                    current_body.mass = 0.f;
                else
                    merge_two_bodies(current_body, cached_bodies[i]);
            }
        }
    }

    body_vec[idx] = current_body;
}

__device__ void update_positions(Body *body_vec, size_t bodies_count) {
    constexpr auto step = 0.2f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > bodies_count)
        return;

    float Fx = 0., Fy = 0.;

    Body current_body = body_vec[idx];
    for (int i = 0; i < bodies_count; i++) {
        auto tmp = body_vec[i];

        if (idx == i || tmp.mass == 0.f)
            continue;

        auto X = tmp.x - current_body.x, Y = tmp.y - current_body.y;
        auto D2 = (X * X + Y * Y);
        auto F = 6.674184 * 10e-9 * (current_body.mass * tmp.mass) / D2;
        auto D = sqrt(D2);

        if (D < (tmp.radius + current_body.radius))
            continue;

        Fx += F * X / D;
        Fy += F * Y / D;
    }

    body_vec[idx].x += current_body.speed_x * step;
    body_vec[idx].y += current_body.speed_y * step;
    body_vec[idx].speed_x += Fx * step / current_body.mass;
    body_vec[idx].speed_y += Fy * step / current_body.mass;
}

__global__ void update_gpu_bodies(Body *body_vec, int bodies_count) {
    merge(body_vec, bodies_count);
    update_positions(body_vec, bodies_count);
}

Physics::Physics() : gpu_bodies_vec_(nullptr) {}

Physics::~Physics() {
    if (gpu_bodies_vec_)
        cudaFree(gpu_bodies_vec_);
}

void Physics::load(std::vector<Body> &bodies) {
    if (gpu_bodies_vec_)
        cudaFree(gpu_bodies_vec_);

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

    cudaMalloc(&gpu_bodies_vec_, sizeof(Body) * bodies.size());
    // cuda Malloc Host
    cudaMemcpy(gpu_bodies_vec_, bodies.data(), sizeof(Body) * bodies.size(), cudaMemcpyHostToDevice);
}

void Physics::update(std::vector<Body> &bodies) {
    if (!bodies.empty()) {
        std::vector<Body> tmp;
        tmp.reserve(((bodies.size() / THREAD_WARP_SIZE) + 1) * THREAD_WARP_SIZE);

        for (const auto &b: bodies) {
            if (b.mass != 0.f)
                tmp.push_back(b);
        }

        for (int i = 0; i < THREAD_WARP_SIZE - (bodies.size() % THREAD_WARP_SIZE); i++) {
            tmp.push_back({});
        }

        bodies = tmp;
    }

    dim3 threads = dim3(THREAD_WARP_SIZE, 1);
    dim3 blocks = bodies.size() / threads.x != 0 ? dim3((bodies.size() / threads.x) + 1, 1) : dim3(1, 1);

    cudaMemcpy(gpu_bodies_vec_, bodies.data(), sizeof(Body) * bodies.size(), cudaMemcpyHostToDevice);
    update_gpu_bodies<<<blocks, threads>>>(gpu_bodies_vec_, bodies.size());
    cudaMemcpy(bodies.data(), gpu_bodies_vec_, sizeof(Body) * bodies.size(), cudaMemcpyDeviceToHost);
}
