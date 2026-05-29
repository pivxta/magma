#pragma once
#include <memory>
#include <cstdint>
#include <glm/mat4x4.hpp>

class Target;
struct Camera;
struct Image;

class Renderer {
public:
    Renderer();
    Renderer(const std::shared_ptr<Target>& target);
    ~Renderer();

    Renderer(Renderer&& other) noexcept;
    Renderer& operator=(Renderer&& other) noexcept;

    void resize();
    void set_camera(const Camera& world_to_clip);

    void draw_frame();

private:
    struct Inner;
    std::unique_ptr<Inner> inner;
};