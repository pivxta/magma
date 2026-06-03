#pragma once
#include <memory>
#include <cstdint>
#include <glm/mat4x4.hpp>
#include "resource.h"

class Target;
struct Camera;
struct Material;
struct Image;
struct Mesh;

class Renderer {
public:
    Renderer();
    Renderer(const std::shared_ptr<Target>& target);
    ~Renderer();

    Renderer(Renderer&& other) noexcept;
    Renderer& operator=(Renderer&& other) noexcept;

    void resize();
    void set_camera(const Camera& world_to_clip);

    TextureId add_texture(const Image& image);
    MaterialId add_material(const Material& material);
    void add_mesh(const Mesh& mesh);

    void draw_frame();

private:
    struct Inner;
    std::unique_ptr<Inner> inner;
};