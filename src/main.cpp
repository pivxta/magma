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

struct CameraController {
    float max_movement_speed = 3.0f;
    float movement_damping = 15.0f;
    float mouse_sensitivity = 0.002f;
    float scroll_sensitivity = 0.07f;
    float zoom_damping = 8.0f;
    float base_fov_radians = glm::radians(80.0f);
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
                        this->resize();
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

    void resize() {
        WindowSize size = this->window.physical_size();
        this->renderer.resize();
        this->camera.aspect_ratio = 
            static_cast<float>(size.width) / static_cast<float>(size.height);
    }

    void start() {
        this->window.set_cursor_locked(true);
        this->camera.aspect_ratio = 
            static_cast<float>(this->window.physical_size().width)
                / static_cast<float>(this->window.physical_size().height);
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

        glm::vec2 cursor_delta = this->input.cursor_delta();
        this->camera.yaw_radians += cursor_delta.x * this->controller.mouse_sensitivity;
        this->camera.pitch_radians -= cursor_delta.y * this->controller.mouse_sensitivity;
        this->camera.pitch_radians = std::clamp(this->camera.pitch_radians, -1.56f, 1.56f);

        auto right = this->camera.right();
        auto forward = this->camera.forward();
        forward.y = 0.0f;
        glm::vec3 direction(0.0f);

        if (this->input.key_pressed(Key::W)) {
            direction += forward;
        }
        if (this->input.key_pressed(Key::S)) {
            direction -= forward;
        }
        if (this->input.key_pressed(Key::A)) {
            direction -= right;
        }
        if (this->input.key_pressed(Key::D)) {
            direction += right;
        }

        if (direction != glm::vec3(0.0f)) {
            direction = glm::normalize(direction);
        }

        if (this->input.key_pressed(Key::Space)) {
            direction.y -= 1.0f;
        }
        if (this->input.key_pressed(Key::LeftShift)) {
            direction.y += 1.0f;
        }

        glm::vec3 target_velocity = direction * this->controller.max_movement_speed;
        float target_factor = 1.0f - std::exp(-this->controller.movement_damping * delta_time);

        this->camera_velocity = glm::mix(this->camera_velocity, target_velocity, target_factor);
        this->camera.translation += this->camera_velocity * delta_time;

        glm::vec2 scroll_delta = this->input.scroll_delta();
        if (scroll_delta.y != 0.0f) {
            this->target_log_zoom += this->controller.scroll_sensitivity * scroll_delta.y;
        }
        if (this->input.key_just_pressed(Key::Q)) {
            this->target_log_zoom += this->controller.scroll_sensitivity * 2.0f;
        }
        if (this->input.key_just_pressed(Key::E)) {
            this->target_log_zoom -= this->controller.scroll_sensitivity * 2.0f;
        }

        float t = 1.0f - std::exp(-this->controller.zoom_damping * delta_time);
        this->current_log_zoom = glm::mix(this->current_log_zoom, this->target_log_zoom, t);
        float base_tan = std::tan(this->controller.base_fov_radians * 0.5f);
        this->camera.fov_radians = 2.0f * std::atan(base_tan * std::exp(-this->current_log_zoom));
    }

    Window window;
    Input input;
    Renderer renderer;

    Camera camera;
    CameraController controller;
    glm::vec3 camera_velocity = glm::vec3(0.0f);
    float current_log_zoom = 0.0f;
    float target_log_zoom = 0.0f;
    bool paused = false;
};

int main() {
    spdlog::set_level(spdlog::level::trace);
    App app;
    app.run();
    return 0;
}