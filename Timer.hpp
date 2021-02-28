#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;

public:
    inline void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    inline std::chrono::milliseconds value() {
        auto tmp = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(tmp - start_);
    }
};