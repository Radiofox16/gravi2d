//
// Created by ilya on 28.02.2021.
//

#ifndef GRAVI2D_PHYSICS_HPP
#define GRAVI2D_PHYSICS_HPP

#include "Body.hpp"
#include <vector>
#include <memory>
#include <functional>

class Physics {
public:
    struct Collision
    {
        int32_t collision_with_idx[4];
    };

private:
    std::unique_ptr<Body, void(*)(Body *)>  d_bodies_in_;
    std::unique_ptr<Body, void(*)(Body *)>  d_bodies_out_;

    std::unique_ptr<Collision, void(*)(Collision *)>  d_bodies_collisions_;

    void allocate_device_memory(int bodies_count);

public:
    Physics();
    ~Physics();

    void load(std::vector<Body>& bodies);

    void update(std::vector<Body>& bodies);
};

#endif //GRAVI2D_PHYSICS_HPP
