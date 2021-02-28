//
// Created by ilya on 28.02.2021.
//

#ifndef GRAVI2D_CREATE_RANDOM_UNIVERSE_HPP
#define GRAVI2D_CREATE_RANDOM_UNIVERSE_HPP

#include "Body.hpp"
#include <vector>
#include <cstdint>
#include <random>
#include "random.hpp"
#include <stdexcept>

std::vector<Body>
create_random_universe(float spawn_square_side, std::uint_fast32_t bodies_count, float max_abs_speed,
                       float max_radius,
                       float max_mass) {
    if (max_radius <= 0.)
        throw std::runtime_error("Impossible max_radius");

    if (max_mass <= 0.)
        throw std::runtime_error("Impossible max_mass");

    if (max_abs_speed < 0.)
        throw std::runtime_error("Impossible max_abs_speed");

    if (spawn_square_side <= 0.)
        throw std::runtime_error("Impossible spawn_square_side");

    std::vector<Body> result;

    for (int bn = 0; bn < bodies_count; ++bn) {
        Body b;

        b.id = bn;

        b.mass = 0.001 + erand48(SEED) * max_mass;
        b.radius = 0.001 + erand48(SEED) * max_radius;
        b.speed_x = max_abs_speed * (-1.f + 2.f * erand48(SEED));
        b.speed_y = (erand48(SEED) > 0.5f ? 1 : -1) * std::sqrt(max_abs_speed * max_abs_speed - b.speed_x * b.speed_x);
        b.x = spawn_square_side * (-1.f + 2.f * erand48(SEED)) / 2.f;
        b.y = spawn_square_side * (-1.f + 2.f * erand48(SEED)) / 2.f;

        result.push_back(b);
    }

    return result;
}

#endif //GRAVI2D_CREATE_RANDOM_UNIVERSE_HPP
