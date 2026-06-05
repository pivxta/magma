#pragma once
#include <cstdint>

struct TextureId {
    uint32_t index;
    uint32_t generation;

    bool operator==(const TextureId&) const = default;
};

struct MaterialId {
    uint32_t index;
    uint32_t generation;

    bool operator==(const MaterialId&) const = default;
};

struct MeshId {
    uint32_t index;
    uint32_t generation;
    
    bool operator==(const MeshId&) const = default;
};