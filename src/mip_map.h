#pragma once
#include "texture.h"

class MipmapGenerator {
public:
    MipmapGenerator() = default;

    void generate(Texture texture);
    void flush(vk::CommandBuffer command_buffer);

private:
    struct PendingGeneration {
        Texture texture;
    };

    std::vector<PendingGeneration> pending_gens;
};