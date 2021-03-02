#include "Window.hpp"
#include "create_random_universe.hpp"
#include "Timer.hpp"
#include "Physics.hpp"

constexpr auto WINDOW_NAME = "gravi2d";
constexpr auto SCENE_WIDTH = 1080;
constexpr auto SCENE_HEIGHT = 1080;

constexpr auto SPAWN_AREA_SIZE = 4000.f;
constexpr auto SPAWN_BODIES_COUNT = 8192;
constexpr auto SPAWN_MAX_MASS = 10000000.f;
constexpr auto SPAWN_MAX_RADIUS = 10.f;
constexpr auto SPAWN_MAX_ABS_SPEED = 10.f;

int main() {
    Window<SCENE_WIDTH, SCENE_HEIGHT> window;
    auto bodies = create_random_universe(SPAWN_AREA_SIZE, SPAWN_BODIES_COUNT, SPAWN_MAX_ABS_SPEED,
                                         SPAWN_MAX_RADIUS, SPAWN_MAX_MASS);

    char key = 0;
    Physics physics;
    Timer t;

    physics.load(bodies);

    int i = 0;

    while (key != 27 && key != 'q') {
        t.reset();
        physics.update(bodies);
        auto update_tm = t.value();

        window.draw(WINDOW_NAME, bodies, SPAWN_AREA_SIZE);
        key = cv::waitKey(1); // TEMPORARY
        i++;
    }

    cv::destroyWindow(WINDOW_NAME);

    return 0;
}
