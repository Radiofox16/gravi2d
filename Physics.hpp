//
// Created by ilya on 28.02.2021.
//

#ifndef GRAVI2D_PHYSICS_HPP
#define GRAVI2D_PHYSICS_HPP

#include "Body.hpp"
#include <vector>

class Physics {

public:
    Physics();

    ~Physics();

    void update(std::vector<Body> &bodies);
};

#endif //GRAVI2D_PHYSICS_HPP
