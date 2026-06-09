#pragma once
#include <memory>
#include "resource.h"

class Target;
struct Camera;
struct Material;
struct Image;
struct Mesh;
struct Scene;

class Renderer {
public:
    Renderer();
    Renderer(const std::shared_ptr<Target>& target);
    ~Renderer();

    Renderer(Renderer&& other) noexcept;
    Renderer& operator=(Renderer&& other) noexcept;

    void resize();

    TextureId add_texture(const Image& image); 
    void set_texture(TextureId id, const Image& image); 
    void free_texture(TextureId id); 

    MaterialId add_material(const Material& material);
    const Material* get_material(MaterialId id);
    void set_material(MaterialId id, const Material& material);
    void free_material(MaterialId id);

    MeshId add_mesh(const Mesh& mesh);
    void set_mesh(MeshId id, const Mesh& mesh);
    void free_mesh(MeshId id);

    void draw(const Scene& scene);

private:
    struct Inner;
    std::unique_ptr<Inner> inner;
};