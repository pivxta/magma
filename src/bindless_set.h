#pragma once
#include "device.h"
#include "texture.h"
#include "texture_filtering.h"
#include "slot_map.h"

class BindlessSet {
public:
    BindlessSet() = default;

    BindlessSet(
        DeviceHandle device, 
        uint32_t frames_in_flight,
        uint32_t texture_descriptor_count,
        const GlobalSamplerInfo& sampler_info = {}
    );

    BindlessSet(const BindlessSet&) = delete;
    BindlessSet& operator=(const BindlessSet&) = delete;
    BindlessSet(BindlessSet&&) noexcept = default;
    BindlessSet& operator=(BindlessSet&&) noexcept = default;

    ~BindlessSet();

    void update_pending();

    std::optional<SlotKey<Texture>> add_texture(
        const std::function<Texture(const DeviceHandle&)>& create
    );
    const Texture* get_texture(SlotKey<Texture> key) const;
    void free_texture(SlotKey<Texture> key);
    void configure_samplers(const GlobalSamplerInfo& info);
    uint32_t get_sampler(Sampler sampler) const;

    uint32_t texture_capacity() const {
        return this->textures.capacity().value();
    }

    void begin_frame(uint64_t frame_counter) {
        this->frame_counter = frame_counter;
    }

    vk::DescriptorSetLayout descriptor_set_layout() const {
        return this->desc_set_layout;
    }
    
    vk::DescriptorSet descriptor_set() const {
        return this->desc_set;
    }

private:
    void destroy_texture(SlotKey<Texture> texture);
    void bind_samplers(vk::DescriptorSet set, std::span<Sampler> types);
    void create_samplers();
    void destroy_samplers();

    struct PendingDestroy {
        uint64_t request_frame;
        SlotKey<Texture> texture;
    };

    DeviceHandle device;

    vk::DescriptorPool desc_pool;
    vk::DescriptorSetLayout desc_set_layout;
    vk::DescriptorSet desc_set;

    SlotMap<Texture> textures;
    std::deque<PendingDestroy> destroy_queue;
    std::array<vk::Sampler, static_cast<size_t>(Sampler::Count)> samplers;
    GlobalSamplerInfo sampler_info;
    bool should_reconfigure_samplers = false;
    uint32_t frames_in_flight;
    uint64_t frame_counter = 0;
};