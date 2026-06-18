#include "bindless_set.h"
#include "vk_error.h"


static vk::DescriptorPool create_descriptor_pool(vk::Device device, uint32_t max_textures) {
    std::array pool_sizes = {
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eSampler)
            .setDescriptorCount(BindlessSet::SAMPLER_COUNT),
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eSampledImage)
            .setDescriptorCount(max_textures),
    };
    auto [result, descriptor_pool] = device.createDescriptorPool(
        vk::DescriptorPoolCreateInfo()
            .setMaxSets(1)
            .setPoolSizes(pool_sizes)
            .setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind)
    );
    vk_expect(result, "Failed to create descriptor pool");
    return descriptor_pool;
}

static vk::DescriptorSetLayout create_set_layout(vk::Device device, uint32_t max_textures) {
    std::array bindings = {
        vk::DescriptorSetLayoutBinding()
            .setBinding(BindlessSet::TEXTURE_BINDING)
            .setDescriptorType(vk::DescriptorType::eSampledImage)
            .setDescriptorCount(max_textures)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment),
        vk::DescriptorSetLayoutBinding()
            .setBinding(BindlessSet::SAMPLER_BINDING)
            .setDescriptorType(vk::DescriptorType::eSampler)
            .setDescriptorCount(BindlessSet::SAMPLER_COUNT)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment),
    };

    std::array<vk::DescriptorBindingFlags, 2> binding_flags = {
        vk::DescriptorBindingFlagBits::eUpdateAfterBind
            | vk::DescriptorBindingFlagBits::ePartiallyBound,
        vk::DescriptorBindingFlagBits::eUpdateAfterBind
            | vk::DescriptorBindingFlagBits::ePartiallyBound,
    };

    auto [result1, layout] = device.createDescriptorSetLayout(
        vk::StructureChain {
            vk::DescriptorSetLayoutCreateInfo()
                .setBindings(bindings)
                .setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool),
            vk::DescriptorSetLayoutBindingFlagsCreateInfo()
                .setBindingFlags(binding_flags)
        }.get()
    );
    vk_expect(result1, "Failed to create descriptor set layout");
    return layout;
}

static vk::DescriptorSet create_set(
    vk::Device device,
    vk::DescriptorPool pool,
    vk::DescriptorSetLayout layout
) {
    auto [result2, sets] = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(pool)
            .setSetLayouts(layout)
    );
    vk_expect(result2, "Failed to create descriptor set layout");
    return sets[0];
}

BindlessSet::BindlessSet(
    DeviceHandle device, 
    uint32_t max_textures,
    const TextureSamplerInfo& sampler_info
) {
    this->device = std::move(device);
    this->textures = std::make_shared<SlotMap<Texture>>(max_textures);
    this->desc_pool = create_descriptor_pool(this->device->logical, max_textures);
    this->desc_set_layout = create_set_layout(this->device->logical, max_textures);
    this->desc_set = create_set(this->device->logical, this->desc_pool, this->desc_set_layout);
    this->sampler_info = sampler_info;
    this->create_samplers();
}

BindlessSet::~BindlessSet() {
    if (this->device != nullptr) {
        this->device->wait_idle();
        this->device->logical.destroyDescriptorSetLayout(this->desc_set_layout);
        this->device->logical.destroyDescriptorPool(this->desc_pool);
        this->destroy_samplers();
        this->textures->clear([&](Texture& texture) {
            texture.destroy(this->device);
        });
    }
}

void BindlessSet::update_dirty_samplers() {
    if (this->should_reconfigure_samplers) {
        this->device->wait_idle();
        this->destroy_samplers();
        this->create_samplers();
    }
}

std::optional<BindlessKey> BindlessSet::add_texture(
    const std::function<Texture(const DeviceHandle&)>& create
) {
    if (auto key = this->textures->reserve(); key.has_value()) {
        Texture texture = create(this->device);
        auto info = vk::DescriptorImageInfo()
            .setImageView(texture.default_view())
            .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
        this->device->logical.updateDescriptorSets(
            vk::WriteDescriptorSet()
                .setDstSet(this->desc_set)
                .setDstArrayElement(key->index)
                .setDstBinding(TEXTURE_BINDING)
                .setDescriptorType(vk::DescriptorType::eSampledImage)
                .setImageInfo(info), 
            {}
        );
        *this->textures->get(key.value()) = texture;
        return key;
    }
    return std::nullopt;
}

void BindlessSet::free_texture(BindlessKey key) {
    auto device = this->device;
    auto textures = this->textures;
    this->device->defer([device, textures, key]() {
        textures->free(key, [&](Texture& texture) {
            texture.destroy(device);
            texture = Texture{};
        });
    });
}

const Texture* BindlessSet::get_texture(BindlessKey key) const {
    return this->textures->get(key);
}

static uint32_t get_sampler_index(Sampler sampler) { 
    return static_cast<uint32_t>(sampler);
}

static uint32_t get_sampler_index(ComparisonSampler sampler) {
    return static_cast<uint32_t>(sampler) + static_cast<uint32_t>(Sampler::Count);
}

void BindlessSet::configure_samplers(const TextureSamplerInfo& info) {
    this->sampler_info = info;
    this->should_reconfigure_samplers = true;
}

uint32_t BindlessSet::get_sampler(Sampler sampler) const {
    return get_sampler_index(sampler);
}

uint32_t BindlessSet::get_sampler(ComparisonSampler sampler) const {
    return get_sampler_index(sampler);
}

static vk::SamplerAddressMode get_address_mode(Sampler sampler) {
    switch (sampler) {
        case Sampler::LinearRepeat: return vk::SamplerAddressMode::eRepeat;
        case Sampler::LinearMirrored: return vk::SamplerAddressMode::eMirroredRepeat;
        case Sampler::LinearClamp: return vk::SamplerAddressMode::eClampToEdge;
        case Sampler::NearestRepeat: return vk::SamplerAddressMode::eRepeat;
        case Sampler::NearestMirrored: return vk::SamplerAddressMode::eMirroredRepeat;
        case Sampler::NearestClamp: return vk::SamplerAddressMode::eClampToEdge;
        default: return vk::SamplerAddressMode::eRepeat;
    }
}

static vk::Filter get_filter(ComparisonSampler sampler) {
    switch (sampler) {
        case ComparisonSampler::Nearest: return vk::Filter::eNearest;
        case ComparisonSampler::Linear:
        default:
            return vk::Filter::eLinear;
    }
}

static vk::Filter get_filter(Sampler sampler) {
    switch (sampler) {
        case Sampler::LinearRepeat:
        case Sampler::LinearMirrored:
        case Sampler::LinearClamp: 
            return vk::Filter::eLinear;
        case Sampler::NearestRepeat: 
        case Sampler::NearestMirrored:
        case Sampler::NearestClamp: 
        default:
            return vk::Filter::eNearest;
    }
}

static vk::Filter get_filter(Filter filter) {
    switch (filter) {
        case Filter::Nearest: return vk::Filter::eNearest;
        case Filter::Linear:
        default:
            return vk::Filter::eLinear;
    }
}

static vk::SamplerMipmapMode get_mip_map_filter(Filter filter) {
    switch (filter) {
        case Filter::Nearest: return vk::SamplerMipmapMode::eNearest;
        case Filter::Linear:
        default:
            return vk::SamplerMipmapMode::eLinear;
    }
}

static vk::Sampler create_sampler(
    const DeviceHandle& device, 
    const TextureSamplerInfo& info, 
    Sampler type
) {
    vk::SamplerAddressMode address_mode = get_address_mode(type);
    vk::Filter filter = get_filter(type);
    auto sampler_info = vk::SamplerCreateInfo()
        .setMagFilter(filter)
        .setAddressModeU(address_mode)
        .setAddressModeV(address_mode)
        .setAddressModeW(address_mode)
        .setMinFilter(get_filter(info.minify_filter))
        .setMipmapMode(get_mip_map_filter(info.mip_map_filter))
        .setMaxAnisotropy(info.max_anisotropy)
        .setAnisotropyEnable(info.minify_filter == Filter::Linear && info.max_anisotropy > 1)
        .setMinLod(0.0f)
        .setMaxLod(128.0f);

    auto [result, sampler] = device->logical.createSampler(sampler_info);
    vk_expect(result, "Failed to create sampler");
    return sampler;
}

static vk::Sampler create_sampler(const DeviceHandle& device, ComparisonSampler type) {
    vk::Filter filter = get_filter(type);
    auto sampler_info = vk::SamplerCreateInfo()
        .setMagFilter(filter)
        .setMinFilter(vk::Filter::eNearest)
        .setMipmapMode(vk::SamplerMipmapMode::eNearest)
        .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
        .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
        .setCompareEnable(true)
        .setCompareOp(vk::CompareOp::eGreaterOrEqual)
        .setMinLod(0.0f)
        .setMaxLod(128.0f);

    auto [result, sampler] = device->logical.createSampler(sampler_info);
    vk_expect(result, "Failed to create sampler");
    return sampler;
}

void BindlessSet::create_samplers() {
    std::array normal_types = {
        Sampler::NearestRepeat,
        Sampler::NearestMirrored,
        Sampler::NearestClamp,
        Sampler::LinearRepeat,
        Sampler::LinearMirrored,
        Sampler::LinearClamp
    };
    std::array cmp_types = {
        ComparisonSampler::Linear,
        ComparisonSampler::Nearest,
    };
    for (auto type: normal_types) {
        this->samplers[get_sampler_index(type)] = 
            create_sampler(this->device, this->sampler_info, type);
    }
    for (auto type: cmp_types) {
        this->samplers[get_sampler_index(type)] = create_sampler(this->device, type);
    }
    this->bind_samplers(this->desc_set);
}

void BindlessSet::bind_samplers(vk::DescriptorSet set) {
    std::vector<vk::DescriptorImageInfo> infos;
    infos.reserve(SAMPLER_COUNT);
    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(SAMPLER_COUNT);

    for (uint32_t index = 0; index < SAMPLER_COUNT; index++) {
        vk::Sampler sampler = this->samplers[index];
        if (sampler == vk::Sampler()) {
            continue;
        }

        infos.push_back(vk::DescriptorImageInfo().setSampler(sampler));
        writes.push_back(
            vk::WriteDescriptorSet()
                .setDstSet(set)
                .setDstBinding(SAMPLER_BINDING)
                .setDstArrayElement(index)
                .setDescriptorType(vk::DescriptorType::eSampler)
                .setImageInfo(infos.back())
        );
    }

    this->device->logical.updateDescriptorSets(writes, {});
}

void BindlessSet::destroy_samplers() {
    for (auto sampler: this->samplers) {
        if (sampler != vk::Sampler()) {
            this->device->logical.destroySampler(sampler);
        }
    }
}

void BindlessSet::destroy_texture(BindlessKey key) {
    this->textures->free(key, [&](Texture& texture) {
        texture.destroy(this->device);
        texture = Texture{};
    });
}