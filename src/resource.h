#pragma once
#include <cstdint>
#include <functional>

struct TextureId {
    uint32_t index;
    uint32_t generation;

    bool operator==(const TextureId& other) const = default;
};

struct MaterialId {
    uint32_t index;
    uint32_t generation;

    bool operator==(const MaterialId& other) const = default;
};

struct MeshId {
    uint32_t index;
    uint32_t generation;
};

template<>
struct std::hash<TextureId> {
    size_t operator()(const TextureId& id) const noexcept {
        return (static_cast<size_t>(id.index) << 16)
            ^ static_cast<size_t>(id.generation);
    }
};

template<>
struct std::hash<MaterialId> {
    size_t operator()(const MaterialId& id) const noexcept {
        return (static_cast<size_t>(id.index) << 16)
            ^ static_cast<size_t>(id.generation);
    }
};