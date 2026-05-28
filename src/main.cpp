#include <spdlog/spdlog.h>
#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>
#include "camera.h"
#include "window.h"
#include "input.h"
#include "renderer.h"
#include "performance.h"
#include "time.h"
#include "image.h"

struct ControllerSettings {
    float max_movement_speed = 3.0f;
    float movement_damping = 15.0f;
    float mouse_sensitivity = 0.002f;
    float scroll_sensitivity = 0.07f;
    float zoom_damping = 8.0f;
    float base_fov_radians = glm::radians(80.0f);

    Key fly_forward = Key::W;
    Key fly_back = Key::S;
    Key fly_left = Key::A;
    Key fly_right = Key::D;
    Key fly_up = Key::Space;
    Key fly_down = Key::LeftShift;
};

struct Controller {
    ControllerSettings settings;

    glm::vec3 velocity = glm::vec3(0.0f);
    float current_log_zoom = 0.0f;
    float target_log_zoom = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;

    void update(
        Transform& transform,
        Projection& projection,
        const Input& input,
        float delta_time
    ) {
        this->look_around(transform, input);
        this->fly_around(transform, input, delta_time);
        this->zoom(input, delta_time);
        this->update_position(transform, delta_time);
        this->update_fov(projection);
    }

private:
    void look_around(Transform& transform, const Input& input) {
        glm::vec2 cursor_delta = input.cursor_delta();
        this->yaw += cursor_delta.x * this->settings.mouse_sensitivity;
        this->pitch -= cursor_delta.y * this->settings.mouse_sensitivity;
        this->pitch = std::clamp(this->pitch, -1.56f, 1.56f);
        transform.rotation = 
            glm::angleAxis(this->yaw, glm::vec3(0.0f, 1.0f, 0.0f))
                * glm::angleAxis(this->pitch, glm::vec3(1.0f, 0.0f, 0.0f));
    }

    void fly_around(const Transform& transform, const Input& input, float delta_time) {
        glm::vec3 right = transform.right();
        glm::vec3 forward = transform.forward();
        forward.y = 0.0f;
        glm::vec3 direction(0.0f);

        if (input.key_pressed(this->settings.fly_forward)) {
            direction += forward;
        }
        if (input.key_pressed(this->settings.fly_back)) {
            direction -= forward;
        }
        if (input.key_pressed(this->settings.fly_left)) {
            direction -= right;
        }
        if (input.key_pressed(this->settings.fly_right)) {
            direction += right;
        }

        if (direction != glm::vec3(0.0f)) {
            direction = glm::normalize(direction);
        }

        if (input.key_pressed(this->settings.fly_up)) {
            direction.y -= 1.0f;
        }
        if (input.key_pressed(this->settings.fly_down)) {
            direction.y += 1.0f;
        }

        glm::vec3 target_velocity = direction * this->settings.max_movement_speed;
        float target_factor = 1.0f - std::exp(-this->settings.movement_damping * delta_time);

        this->velocity = glm::mix(this->velocity, target_velocity, target_factor);
    }

    void zoom(const Input& input, float delta_time) {
        glm::vec2 scroll_delta = input.scroll_delta();
        if (scroll_delta.y != 0.0f) {
            this->target_log_zoom += this->settings.scroll_sensitivity * scroll_delta.y;
        }

        float t = 1.0f - std::exp(-this->settings.zoom_damping * delta_time);
        this->current_log_zoom = glm::mix(this->current_log_zoom, this->target_log_zoom, t);
    }

    void update_position(Transform& transform, float delta_time) {
        transform.translation += this->velocity * delta_time;
    }

    void update_fov(Projection& projection) {
        if (auto *p = std::get_if<PerspectiveProjection>(&projection); p != nullptr) {
            float base_tan = std::tan(this->settings.base_fov_radians * 0.5f);
            p->fov_radians = 2.0f * std::atan(base_tan * std::exp(-this->current_log_zoom));
        }
    }
};

class App {
public:
    App() = default;
    ~App() = default;

    void run() {
        this->window = Window(1280, 720, "Hello Vulkan");
        this->renderer = Renderer(this->window.target());
        this->input = this->window.input();
        this->main_loop();
    }

private:
    void main_loop() {
        this->start();

        PerformanceCounter perf_counter;
        Instant last_frame = Instant::now();
        while (true) {
            auto delta_time = last_frame.elapsed_seconds<float>();
            last_frame = Instant::now();

            for (auto event: this->window.poll_events()) {
                switch (event) {
                    case WindowEvent::Resized:
                        this->renderer.resize();
                        break;
                    case WindowEvent::CloseRequested:
                        return;
                }
            }

            this->update(delta_time);
            this->renderer.set_camera(this->camera);
            this->renderer.draw_frame();

            if (auto perf_stats = perf_counter.record(1.0); perf_stats != std::nullopt) {
                spdlog::info(
                    "FPS: {:.0f} | avg: {:.2f}ms | median: {:.2f}ms | 1% low: {:.2f}ms | 0.1% low: {:.2f}ms",
                    perf_stats->frames_per_second,
                    perf_stats->frame_time_avg_ms,
                    perf_stats->frame_time_median_ms,
                    perf_stats->frame_time_high_1pct_ms,
                    perf_stats->frame_time_high_0_1pct_ms
                );
            }
        }
    }

    void start() {
        this->window.set_cursor_locked(true);
    }

    void update(float delta_time) {
        this->poll_app_actions();
        this->control_camera(delta_time);
    }

    void poll_app_actions() {
        if (this->input.key_just_pressed(Key::Escape)) {
            if (paused) {
                this->paused = false;
                this->window.set_cursor_locked(true);
            } else {
                this->paused = true;
                this->window.set_cursor_locked(false);
            }
        }
        if (this->input.mouse_button_just_pressed(MouseButton::Left) && paused) {
            this->paused = false;
            this->window.set_cursor_locked(true);
        }
    }
    
    void control_camera(float delta_time) {
        if (this->paused) {
            return;
        }

        this->controller.update(
            this->camera.transform, 
            this->camera.projection,
            this->input,
            delta_time
        );
    }

    Window window;
    Input input;
    Renderer renderer;

    Camera camera;
    Controller controller;
    
    bool paused = false;
};

int main() {
    spdlog::set_level(spdlog::level::trace);
    App app;
    app.run();
    return 0;
}