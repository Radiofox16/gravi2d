#include "Window.hpp"
#include "create_random_universe.hpp"
#include "Timer.hpp"
#include "Physics.hpp"

constexpr auto WINDOW_NAME = "gravi2d";
constexpr auto SCENE_WIDTH = 1280;
constexpr auto SCENE_HEIGHT = 1280;

constexpr auto SPAWN_AREA_SIZE = 10000.f;
constexpr auto SPAWN_BODIES_COUNT = 1024 * 16;
constexpr auto SPAWN_MAX_MASS = 1000.f;
constexpr auto SPAWN_MAX_RADIUS = 10.f;
constexpr auto SPAWN_MAX_ABS_SPEED = 5.f;

int main() {
    Window<SCENE_WIDTH, SCENE_HEIGHT> window;
    auto bodies = create_random_universe(SPAWN_AREA_SIZE, SPAWN_BODIES_COUNT, SPAWN_MAX_ABS_SPEED,
                                         SPAWN_MAX_RADIUS, SPAWN_MAX_MASS);

    char key = 0;

    Physics physics;
    Timer t;

    physics.load(bodies);

    auto horizontal_draw_size = SPAWN_AREA_SIZE * 2;
    window.draw(WINDOW_NAME, bodies, horizontal_draw_size);
    key = cv::waitKey(10); // TEMPORARY

    while (key != 27 && key != 'q') {
        t.reset();
        physics.update(bodies);
        auto update_tm = t.value();
        std::cout << "Bodies count: " << bodies.size() << '\n';
        std::cout << "Update time: " << update_tm.count() << " ms\n";

        window.draw(WINDOW_NAME, bodies, horizontal_draw_size);
        key = cv::waitKey(1); // TEMPORARY

        if (key == '-')
            horizontal_draw_size *= 1.2f;

        if (key == '+' || key == '=')
            horizontal_draw_size /= 1.2f;
    }

    cv::destroyAllWindows();
    return 0;
}
