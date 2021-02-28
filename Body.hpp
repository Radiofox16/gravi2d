//
// Created by ilya on 28.02.2021.
//

#ifndef GRAVI2D_BODY_HPP
#define GRAVI2D_BODY_HPP

#include <cstdint>

struct Body {
    // Position
    float x, y;

    // Radius
    float radius;

    // Mass of object
    float mass;

    // Speed
    float speed_x, speed_y;

    // Unique id
    std::int_fast32_t id;
};


#endif //GRAVI2D_BODY_HPP
