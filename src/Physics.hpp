//
// Created by ilya on 28.02.2021.
//

#ifndef GRAVI2D_PHYSICS_HPP
#define GRAVI2D_PHYSICS_HPP

#include "Body.hpp"
#include <vector>

class Physics {
    Body *gpu_bodies_vec_;

public:
    Physics();

    ~Physics();

    void load(const std::vector<Body> &bodies);

    void update(std::vector<Body> &bodies);
};

#endif //GRAVI2D_PHYSICS_HPP
