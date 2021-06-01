//
// Created by ilya on 28.02.2021.
//

#include "Physics.hpp"
#include <algorithm>
#include <limits>
#include <iostream>

constexpr int THREAD_WARP_SIZE = 64;
constexpr float GRAVITY_CONST = 6.674184 * 10e-5;

__device__ inline bool intersect(const Body &l, const Body &r) {
    float X = l.x - r.x, Y = l.y - r.y, R = r.radius + l.radius;
    return (X * X + Y * Y - R * R) < 0.f;
}

__global__ void detect_collisions(const Body *body_vec_in, Physics::Collision *collisions_out, int bodies_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const Body current_body = body_vec_in[idx];

    __shared__ Body cached_bodies[THREAD_WARP_SIZE];
    auto collision = collisions_out[idx];

    for (int block_i = blockIdx.x; block_i < gridDim.x; block_i++) {
        __syncthreads();

        if (block_i * blockDim.x + threadIdx.x >= bodies_count) {
            cached_bodies[threadIdx.x] = Body{};//, 0, 0
        } else {
            cached_bodies[threadIdx.x] = body_vec_in[block_i * blockDim.x + threadIdx.x];
        }

        __syncthreads();
        int collisions_count = 0;

        for (int j = 0; j < THREAD_WARP_SIZE; j++) {
            const auto current_idx = block_i * blockDim.x + j;
            if (idx == current_idx || cached_bodies[j].mass == 0.f)
                continue;

            if (intersect(current_body, cached_bodies[j])) {
                collision.idxs[collisions_count] = current_idx;

                collisions_count++;
                if (collisions_count >= Physics::Collision::collisions_count)
                    break;
            }
        }
    }

    collisions_out[idx] = collision;
}

__global__ void d_update_positions(const Body *body_vec_in, Body *body_vec_out, int bodies_count) {
    constexpr auto step = 1.f;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ Body cached_bodies[THREAD_WARP_SIZE];
    Body current_body = body_vec_in[idx];
    float Fx = 0., Fy = 0.;

    for (int block_i = 0; block_i < gridDim.x; block_i++) {
        __syncthreads();

        if (block_i * blockDim.x + threadIdx.x >= bodies_count) {
            cached_bodies[threadIdx.x] = Body{};//, 0, 0
        } else {
            cached_bodies[threadIdx.x] = body_vec_in[block_i * blockDim.x + threadIdx.x];
        }

        __syncthreads();

        for (int j = 0; j < THREAD_WARP_SIZE; j++) {
            if (idx == block_i * blockDim.x + j || cached_bodies[j].mass == 0.f)
                continue;

            auto X = cached_bodies[j].x - current_body.x, Y = cached_bodies[j].y - current_body.y;
            auto D2 = (X * X + Y * Y);
            auto F = GRAVITY_CONST * static_cast<double>(current_body.mass * cached_bodies[j].mass) / D2;
            auto D = sqrt(D2);

            // Because spheres do not collide
            if (D <= current_body.radius + cached_bodies[j].radius)
                continue;

            Fx += F * X / D;
            Fy += F * Y / D;
        }
    }

    current_body.speed_x += Fx * step / current_body.mass;
    current_body.speed_y += Fy * step / current_body.mass;

    current_body.x += current_body.speed_x * step;
    current_body.y += current_body.speed_y * step;

    // current_body.dbg_1 = gridDim.x;

    __syncthreads();
    body_vec_out[idx] = current_body;
}

Physics::Physics() : d_bodies_in_{nullptr, nullptr}, d_bodies_out_{nullptr, nullptr},
                     d_collisions_{nullptr, nullptr} {}

Physics::~Physics() = default;

void Physics::allocate_device_memory(int bodies_count) {
    void *d_ptr = nullptr;
    cudaMalloc(&d_ptr, sizeof(Body) * bodies_count);
    d_bodies_in_ = decltype(d_bodies_in_){reinterpret_cast<Body *> (d_ptr), [](Body *b) { cudaFree(b); }};

    cudaMalloc(&d_ptr, sizeof(Body) * bodies_count);
    d_bodies_out_ = decltype(d_bodies_out_){reinterpret_cast<Body *> (d_ptr), [](Body *b) { cudaFree(b); }};

    cudaMalloc(&d_ptr, sizeof(Collision) * bodies_count);
    d_collisions_ = decltype(d_collisions_){reinterpret_cast<Collision *> (d_ptr),
                                            [](Collision *b) { cudaFree(b); }};
}


void Physics::load(std::vector<Body> &bodies) {
    allocate_device_memory(bodies.size());

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

    const auto minmax_x = std::minmax_element(std::cbegin(bodies), std::cend(bodies),
                                              [](const Body &a, const Body &b) { return a.x < b.x; });
    const auto y_multiplier = minmax_x.second->x - minmax_x.first->x;
    std::sort(std::begin(bodies), std::end(bodies), [y_multiplier](Body &a, Body &b) {
        auto diff = (b.y - a.y) * y_multiplier + b.x - a.x;
        return diff > 0;
    });
}


void Physics::update_bodies_positions(std::vector<Body> &bodies) {
    const dim3 threads = dim3(THREAD_WARP_SIZE, 1);

    auto blk_count = bodies.size() / threads.x;
    dim3 blocks = dim3(1, 1);
    if (blk_count)
        blocks = dim3(blk_count + ((bodies.size() % threads.x) == 0 ? 0 : 1), 1);

    cudaMemcpy(d_bodies_in_.get(), bodies.data(), sizeof(Body) * bodies.size(), cudaMemcpyHostToDevice);
    d_update_positions <<< blocks, threads >>>(d_bodies_in_.get(), d_bodies_out_.get(), bodies.size());
    cudaMemcpy(bodies.data(), d_bodies_out_.get(), sizeof(Body) * bodies.size(), cudaMemcpyDeviceToHost);
}

void Physics::merge_bodies(std::vector<Body> &bodies) {
    const dim3 threads = dim3(THREAD_WARP_SIZE, 1);

    auto blk_count = bodies.size() / threads.x;
    dim3 blocks = dim3(1, 1);
    if (blk_count)
        blocks = dim3(blk_count + ((bodies.size() % threads.x) == 0 ? 0 : 1), 1);

    std::vector<Collision> collisions;
    collisions.resize(bodies.size());
    std::fill(std::begin(collisions), std::end(collisions), Collision{-1, -1, -1, -1});

    cudaMemcpy(d_collisions_.get(), collisions.data(), sizeof(Collision) * collisions.size(), cudaMemcpyHostToDevice);
    detect_collisions <<< blocks, threads >>>(d_bodies_in_.get(), d_collisions_.get(), bodies.size());
    cudaMemcpy(collisions.data(), d_collisions_.get(), sizeof(Collision) * collisions.size(), cudaMemcpyDeviceToHost);

    auto tmp = bodies;
    bodies.clear();

    for (int i = 0; i < collisions.size(); i++) {
        const Collision clz = collisions[i];
        Body base = tmp[i];
        if (base.mass == 0.f) continue;

        for (int j = 0; j < Collision::collisions_count; j++) {
            if (clz.idxs[j] <= -1 || clz.idxs[j] <= i) {
                break;
            }

            Body patch = tmp[clz.idxs[j]];
            tmp[clz.idxs[j]] = Body{};
            if (patch.mass == 0.f || base.mass == 0.f)
                continue;

            float radius_ratio = patch.radius / (base.radius + patch.radius);
            base.x += (patch.x - base.x) * radius_ratio;
            base.y += (patch.y - base.y) * radius_ratio;

            base.radius = sqrt(base.radius * base.radius + patch.radius * patch.radius);

            auto mass_sum = patch.mass + base.mass;
            base.speed_x = base.speed_x * (base.mass / mass_sum) + (patch.speed_x) * (patch.mass / mass_sum);
            base.speed_y = base.speed_y * (base.mass / mass_sum) + (patch.speed_y) * (patch.mass / mass_sum);

            base.mass = mass_sum;
        }

        bodies.emplace_back(base);
    }
}

void Physics::update(std::vector<Body> &bodies) {
    if (bodies.empty())
        return;

    update_bodies_positions(bodies);
    cudaMemcpy(d_bodies_in_.get(), d_bodies_out_.get(), sizeof(Body) * bodies.size(), cudaMemcpyDeviceToDevice);
    merge_bodies(bodies);

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
