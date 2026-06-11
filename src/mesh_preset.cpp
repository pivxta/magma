#include "mesh_preset.h"
#include <glm/gtc/constants.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

// Picks a tangent perpendicular to `n`. The bitangent is taken as cross(n, t),
// so tangent.w is always +1 with this basis. Chosen so a default up-facing
// plane (normal -Y) yields tangent +X / bitangent +Z, matching the renderer's
// CCW-front / +Y-down conventions.
static glm::vec3 tangent_for_normal(glm::vec3 n) {
    auto ref = std::abs(n.z) < 0.999f ? 
        glm::vec3(0.0f, 0.0f, 1.0f):
        glm::vec3(1.0f, 0.0f, 0.0f);
    return glm::normalize(glm::cross(ref, n));
}

namespace {
struct MeshBuilder {
    std::vector<uint32_t> indices;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> tangents;
    std::vector<glm::vec2> tex_coords;

    uint32_t vertex(glm::vec3 position, glm::vec3 normal, glm::vec4 tangent, glm::vec2 uv) {
        auto index = static_cast<uint32_t>(this->positions.size());
        this->positions.push_back(position);
        this->normals.push_back(normal);
        this->tangents.push_back(tangent);
        this->tex_coords.push_back(uv);
        return index;
    }

    void triangle(uint32_t a, uint32_t b, uint32_t c) {
        this->indices.push_back(a);
        this->indices.push_back(b);
        this->indices.push_back(c);
    }

    // Corners laid out as v0=(-u,-v) v1=(+u,-v) v2=(-u,+v) v3=(+u,+v); the
    // (0,1,2)+(1,3,2) split winds CCW around (u x v).
    void quad(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
        this->triangle(v0, v1, v2);
        this->triangle(v1, v3, v2);
    }

    Mesh build() {
        return Mesh()
            .set_positions(std::move(this->positions))
            .set_normals(std::move(this->normals))
            .set_tangents(std::move(this->tangents))
            .set_tex_coords(std::move(this->tex_coords))
            .set_indices(std::move(this->indices));
    }
};

void add_quad(
    MeshBuilder& builder,
    glm::vec3 center,
    glm::vec3 normal,
    glm::vec3 u,
    float half_u,
    float half_v
) {
    glm::vec3 v = glm::cross(normal, u);
    glm::vec4 tangent(u, 1.0f);
    uint32_t v0 = builder.vertex(center - half_u * u - half_v * v, normal, tangent, {0.0f, 0.0f});
    uint32_t v1 = builder.vertex(center + half_u * u - half_v * v, normal, tangent, {1.0f, 0.0f});
    uint32_t v2 = builder.vertex(center - half_u * u + half_v * v, normal, tangent, {0.0f, 1.0f});
    uint32_t v3 = builder.vertex(center + half_u * u + half_v * v, normal, tangent, {1.0f, 1.0f});
    builder.quad(v0, v1, v2, v3);
}

Mesh flatten(const Mesh& mesh) {
    MeshBuilder builder;
    for (size_t i = 0; i + 3 <= mesh.indices.size(); i += 3) {
        std::array tri = {mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]};

        glm::vec3 p0 = mesh.positions[tri[0]];
        glm::vec3 face = glm::cross(mesh.positions[tri[1]] - p0, mesh.positions[tri[2]] - p0);
        float face_length = glm::length(face);
        if (face_length < 1e-12f) {
            continue;
        }
        glm::vec3 normal = face / face_length;

        std::array<uint32_t, 3> out;
        for (int k = 0; k < 3; k++) {
            uint32_t index = tri[k];
            glm::vec4 src = mesh.tangents[index];
            glm::vec3 tangent = glm::vec3(src) - normal * glm::dot(normal, glm::vec3(src));
            tangent = glm::dot(tangent, tangent) > 1e-12f ? glm::normalize(tangent)
                                                          : tangent_for_normal(normal);
            out[k] = builder.vertex(
                mesh.positions[index],
                normal,
                glm::vec4(tangent, src.w),
                mesh.tex_coords[index]
            );
        }
        builder.triangle(out[0], out[1], out[2]);
    }
    return builder.build();
}
}

Mesh Quad::to_mesh() {
    glm::vec3 center(0.0f);
    glm::vec3 normal = glm::normalize(this->normal);
    MeshBuilder builder;
    add_quad(
        builder,
        center,
        normal, 
        tangent_for_normal(normal),
        0.5f * this->size,
        0.5f * this->size
    );
    return builder.build();
}

Mesh Cuboid::to_mesh() {
    glm::vec3 h = this->size * 0.5f;
    MeshBuilder builder;
    add_quad(builder, {+h.x, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, h.z, h.y);
    add_quad(builder, {-h.x, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, h.z, h.y);
    add_quad(builder, {0.0f, +h.y, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, h.x, h.z);
    add_quad(builder, {0.0f, -h.y, 0.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, h.x, h.z);
    add_quad(builder, {0.0f, 0.0f, +h.z}, {0.0f, 0.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}, h.x, h.y);
    add_quad(builder, {0.0f, 0.0f, -h.z}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, h.x, h.y);
    return builder.build();
}

Mesh Sphere::to_mesh() {
    const auto pi = glm::pi<float>();
    MeshBuilder builder;

    for (uint32_t stack = 0; stack <= this->stacks; stack++) {
        float phi = pi * static_cast<float>(stack) / static_cast<float>(this->stacks);
        float cos_phi = std::cos(phi);
        float sin_phi = std::sin(phi);

        for (uint32_t slice = 0; slice <= this->slices; slice++) {
            float theta = 2.0f * pi * static_cast<float>(slice) / static_cast<float>(this->slices);
            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);

            glm::vec3 normal(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
            glm::vec2 uv(
                static_cast<float>(slice) / static_cast<float>(this->slices),
                static_cast<float>(stack) / static_cast<float>(this->stacks)
            );
            glm::vec4 tangent(-sin_theta, 0.0f, cos_theta, 1.0f);
            builder.vertex(normal * this->radius, normal, tangent, uv);
        }
    }

    uint32_t stride = this->slices + 1;
    for (uint32_t stack = 0; stack < this->stacks; stack++) {
        for (uint32_t slice = 0; slice < this->slices; slice++) {
            uint32_t a = stack * stride + slice;
            uint32_t b = a + 1;
            uint32_t c = a + stride;
            uint32_t d = c + 1;
            builder.triangle(a, b, c);
            builder.triangle(c, b, d);
        }
    }

    Mesh mesh = builder.build();
    return this->shading == ShadingMode::Flat ? flatten(mesh) : mesh;
}

Mesh Cylinder::to_mesh() {
    const auto pi = glm::pi<float>();
    const auto half_height = this->height * 0.5f;
    MeshBuilder builder;

    for (uint32_t slice = 0; slice <= this->slices; slice++) {
        float theta = 2.0f * pi * static_cast<float>(slice) / static_cast<float>(this->slices);
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);
        float u = static_cast<float>(slice) / static_cast<float>(this->slices);

        glm::vec3 normal(cos_theta, 0.0f, sin_theta);
        glm::vec4 tangent(-sin_theta, 0.0f, cos_theta, 1.0f);
        glm::vec3 rim(this->radius * cos_theta, 0.0f, this->radius * sin_theta);
        builder.vertex(rim + glm::vec3(0.0f, +half_height, 0.0f), normal, tangent, {u, 0.0f});
        builder.vertex(rim + glm::vec3(0.0f, -half_height, 0.0f), normal, tangent, {u, 1.0f});
    }
    for (uint32_t slice = 0; slice < this->slices; slice++) {
        uint32_t a = (slice + 0) * 2 + 0; // ring0[slice]
        uint32_t b = (slice + 1) * 2 + 0; // ring0[slice + 1]
        uint32_t c = (slice + 0) * 2 + 1; // ring1[slice]
        uint32_t d = (slice + 1) * 2 + 1; // ring1[slice + 1]
        builder.triangle(a, b, c);
        builder.triangle(c, b, d);
    }

    // Caps: top faces +Y, bottom faces -Y. Triangle fans around a center vertex.
    struct Cap { float y; glm::vec3 normal; bool flip; };
    std::array caps = {
        Cap{.y = half_height, .normal = {0.0f, 1.0f, 0.0f}, .flip = true},
        Cap{.y = -half_height, .normal = {0.0f, -1.0f, 0.0f}, .flip = false}
    };
    for (const Cap& cap: caps) {
        glm::vec4 tangent(1.0f, 0.0f, 0.0f, 1.0f);
        float v_sign = cap.flip ? -1.0f : 1.0f;
        uint32_t center = builder.vertex({0.0f, cap.y, 0.0f}, cap.normal, tangent, {0.5f, 0.5f});

        auto first = static_cast<uint32_t>(builder.positions.size());
        for (uint32_t slice = 0; slice < this->slices; slice++) {
            float theta = 2.0f * pi * static_cast<float>(slice) / static_cast<float>(this->slices);
            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);
            builder.vertex(
                {this->radius * cos_theta, cap.y, this->radius * sin_theta},
                cap.normal,
                tangent,
                {0.5f + 0.5f * cos_theta, 0.5f + v_sign * 0.5f * sin_theta}
            );
        }
        for (uint32_t slice = 0; slice < this->slices; slice++) {
            uint32_t a = first + slice;
            uint32_t b = first + (slice + 1) % this->slices;
            if (cap.flip) {
                builder.triangle(center, b, a); // CCW about +Y
            } else {
                builder.triangle(center, a, b); // CCW about -Y
            }
        }
    }

    Mesh mesh = builder.build();
    return this->shading == ShadingMode::Flat ? flatten(mesh) : mesh;
}
