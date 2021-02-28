//
// Created by ilya on 28.02.2021.
//

#ifndef GRAVI2D_WINDOW_HPP
#define GRAVI2D_WINDOW_HPP

#include <cstdint>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Body.hpp"

const static std::array BODIES_COLORS{
        cv::Scalar{157, 180, 255},
        cv::Scalar{162, 185, 255},
        cv::Scalar{167, 188, 255},
        cv::Scalar{170, 191, 255},
        cv::Scalar{186, 204, 255},
        cv::Scalar{192, 209, 255},
        cv::Scalar{202, 216, 255},
        cv::Scalar{228, 232, 255},
        cv::Scalar{237, 238, 255},
        cv::Scalar{251, 248, 255},
        cv::Scalar{255, 249, 249},
        cv::Scalar{255, 245, 236},
        cv::Scalar{255, 244, 232},
        cv::Scalar{255, 241, 223},
        cv::Scalar{255, 235, 209},
        cv::Scalar{255, 215, 174},
        cv::Scalar{255, 198, 144},
        cv::Scalar{255, 190, 127},
        cv::Scalar{255, 187, 123},
        cv::Scalar{255, 187, 123},
        cv::Scalar{000, 255, 236},
        cv::Scalar{073, 214, 255},
        cv::Scalar{255, 225, 209}
};

template<std::uint_fast16_t t_width, std::uint_fast16_t t_height>
class Window {

    cv::Mat scene_;

public:
    Window() : scene_(t_width, t_height, CV_8UC3, cv::Scalar(0, 0, 0)) {
    }

    constexpr auto width() const { return t_width; };

    constexpr auto height() const { return t_height; };

    void draw(const std::string_view window_name, const std::vector<Body> &bodies, float horizontal_scene_size) {
        scene_ = cv::Mat::zeros(scene_.size(), scene_.type());
        auto coord_bias = horizontal_scene_size / 2.f;
        auto pix_per_coord = t_width / horizontal_scene_size;

        for (const auto &body : bodies) {
            auto x = static_cast<int>((body.x + coord_bias) * pix_per_coord);
            auto y = static_cast<int>((body.y + coord_bias) * pix_per_coord);

            cv::circle(scene_, {x, y}, static_cast<int>(body.radius * pix_per_coord),
                       BODIES_COLORS[body.id % BODIES_COLORS.size()], cv::FILLED);
        }

        cv::imshow(window_name.data(), scene_);
    }
};


#endif //GRAVI2D_WINDOW_HPP
