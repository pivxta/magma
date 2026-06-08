#pragma once
#include <vector>
#include <span>
#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

struct Mesh {
    std::vector<uint32_t> indices;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> tangents;
    std::vector<glm::vec2> tex_coords;

    Mesh() = default;

    Mesh& set_indices(std::vector<uint32_t>&& indices) {
        this->indices = indices;
        return *this;
    }

    Mesh& set_indices(std::span<const uint32_t> indices) {
        this->indices = std::vector(indices.begin(), indices.end());
        return *this;
    }

    Mesh& set_positions(std::vector<glm::vec3>&& positions) {
        this->positions = positions;
        return *this;
    }

    Mesh& set_positions(std::span<const glm::vec3> positions) {
        this->positions = std::vector(positions.begin(), positions.end());
        return *this;
    }
    
    Mesh& set_normals(std::vector<glm::vec3>&& normals) {
        this->normals = normals;
        return *this;
    }

    Mesh& set_normals(std::span<const glm::vec3> normals) {
        this->normals = std::vector(normals.begin(), normals.end());
        return *this;
    }

    Mesh& set_tangents(std::vector<glm::vec4>&& tangents) {
        this->tangents = tangents;
        return *this;
    }

    Mesh& set_tangents(std::span<const glm::vec4> tangents) {
        this->tangents = std::vector(tangents.begin(), tangents.end());
        return *this;
    }

    Mesh& set_tex_coords(std::vector<glm::vec2>&& tex_coords) {
        this->tex_coords = tex_coords;
        return *this;
    }

    Mesh& set_tex_coords(std::span<const glm::vec2> tex_coords) {
        this->tex_coords = std::vector(tex_coords.begin(), tex_coords.end());
        return *this;
    }
};