#define VMA_IMPLEMENTATION
#include "renderer.h"
#include "swapchain.h"
#include "vk_error.h"
#include "uploader.h"
#include "target.h"
#include "frame_arena_buffer.h"
#include "camera.h"
#include "image.h"
#include "mesh.h"
#include "texture.h"
#include "texture_manager.h"
#include "material_manager.h"
#include "mesh_manager.h"
#include "device.h"
#include "scene.h"
#include "time.h"
#include <vector>
#include <optional>
#include <fstream>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <vk_mem_alloc.hpp>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/mat4x4.hpp>
#include <glm/mat4x3.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vulkan/vulkan.hpp>

static constexpr uint32_t MAX_TEXTURES = 4096;
static constexpr uint32_t MAX_MATERIALS = 2048;
static constexpr auto FRAME_ARENA_CAPACITY_PER_FIF = static_cast<vk::DeviceSize>(16 * 1024 * 1024);
static constexpr auto UPLOADER_CAPACITY_PER_FIF = static_cast<vk::DeviceSize>(64 * 1024 * 1024);
static constexpr auto VERTEX_HEAP_CAPACITY = static_cast<vk::DeviceSize>(128 * 1024 * 1024);
static constexpr auto INDEX_HEAP_CAPACITY = static_cast<vk::DeviceSize>(64 * 1024 * 1024);

static const std::filesystem::path SHADER_DIRECTORY = "shaders/compiled";

// NOLINTBEGIN(performance-enum-size)
enum class TonemapType: uint32_t {
    None = 0,
    Reinhard = 1,
    Agx = 2,
};
// NOLINTEND(performance-enum-size)

struct AgxConstants {
    float saturation;
    alignas(16) glm::vec3 slope;
    alignas(16) glm::vec3 offset;
    alignas(16) glm::vec3 power;
};

struct TonemapConstants {
    TonemapType type;
    float exposure;

    AgxConstants agx;

    TonemapConstants(const Tonemap& tonemap, float film_exposure) {
        std::visit([&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, DisabledTonemap>) {
                this->type = TonemapType::None;
                this->exposure = film_exposure;
            } else if constexpr (std::is_same_v<T, ReinhardTonemap>) {
                this->type = TonemapType::Reinhard;
                this->exposure = film_exposure * value.exposure_factor; 
            } else if constexpr (std::is_same_v<T, AgxTonemap>) {
                this->type = TonemapType::Agx;
                this->exposure = film_exposure * value.exposure_factor; 
                this->agx.saturation = value.look.saturation;
                this->agx.power = value.look.power;
                this->agx.slope = value.look.slope;
                this->agx.offset = value.look.offset;
            }
        }, tonemap);
    }
};

struct DrawConstants {
    vk::DeviceAddress view_data;
    vk::DeviceAddress light_data;
    vk::DeviceAddress point_lights;
    vk::DeviceAddress directional_lights;
    vk::DeviceAddress materials;
    vk::DeviceAddress instances;
    vk::DeviceAddress draws;
};

struct InstanceData {
    glm::mat4x3 local_to_world;
    uint32_t material_index;

    InstanceData(const Transform& transform, uint32_t material_index) {
        this->material_index = material_index;
        this->local_to_world = transform.affine_matrix();
    }
};

struct AmbientLightData {
    glm::vec3 color;
    float illuminance;
};

struct PointLightData {
    alignas(16) glm::vec3 position;
    float radius;
    alignas(16) glm::vec3 color;
    float intensity;
};

struct DirectionalLightData {
    alignas(16) glm::vec3 direction;
    alignas(16) glm::vec3 color;
    float illuminance;
};

struct ViewData {
    alignas(16) glm::mat4 world_to_clip;
    alignas(16) glm::vec3 position;
};

struct LightData {
    alignas(16) glm::vec3 ambient_color;
    float ambient_illuminance;
    uint32_t point_count;
    uint32_t directional_count;
};

struct DrawData {
    vk::DeviceAddress vertices;
    uint32_t instance_offset;
};

struct ScratchData {
    MeshId mesh_id;
    uint32_t object_id;
};

struct SceneDrawData {
    ViewData view;
    std::vector<InstanceData> instances;
    std::vector<DirectionalLightData> dir_lights;
    std::vector<PointLightData> point_lights;
    AmbientLightData ambient_light;
    glm::vec3 clear_color;

    std::vector<ScratchData> scratch;
    std::vector<DrawData> draws;
    std::vector<vk::DrawIndexedIndirectCommand> draw_commands;
};

struct FrameData {
    glm::vec3 clear_color;
    FrameSubBuffer<ViewData> view_data;
    FrameSubBuffer<LightData> light_data;
    FrameSubBuffer<DirectionalLightData> dir_lights;
    FrameSubBuffer<PointLightData> point_lights;
    FrameSubBuffer<InstanceData> instances;
    FrameSubBuffer<DrawData> draws;
    FrameSubBuffer<vk::DrawIndexedIndirectCommand> draw_commands;
};

static inline std::vector<uint32_t> read_spirv_file(const std::filesystem::path& path) {
    auto file_size = std::filesystem::file_size(path);
    if (file_size == 0) {
        return {};
    }
    try {
        std::vector<uint32_t> buffer((file_size + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        std::ifstream stream(path, std::ios::binary);
        stream.read(
            reinterpret_cast<char*>(buffer.data()), 
            static_cast<std::streamsize>(file_size)
        );
        return buffer;
    } catch (const std::exception& e) {
        spdlog::error("Failed to open file '{}': {}", path.generic_string(), e.what());
        return {};
    }
};

static vk::DescriptorPool create_tonemap_descriptor_pool(vk::Device device, uint32_t set_count) {
    std::array pool_sizes = {
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eSampledImage)
            .setDescriptorCount(set_count),
        vk::DescriptorPoolSize()
            .setType(vk::DescriptorType::eSampler)
            .setDescriptorCount(set_count),
    };
    auto [result, pool] = device.createDescriptorPool(
        vk::DescriptorPoolCreateInfo()
            .setMaxSets(set_count)
            .setPoolSizes(pool_sizes)
    );
    vk_expect(result, "Failed to create tonemap descriptor pool");
    return pool;
}

static vk::DescriptorSetLayout create_tonemap_set_layout(vk::Device device) {
    std::array bindings = {
        vk::DescriptorSetLayoutBinding()
            .setBinding(0)
            .setDescriptorType(vk::DescriptorType::eSampledImage)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment),
        vk::DescriptorSetLayoutBinding()
            .setBinding(1)
            .setDescriptorType(vk::DescriptorType::eSampler)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eFragment),
    };
    auto [result, layout] = device.createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo().setBindings(bindings)
    );
    vk_expect(result, "Failed to create tonemap descriptor set layout");
    return layout;
}

static std::vector<vk::DescriptorSet> allocate_tonemap_sets(
    vk::Device device,
    vk::DescriptorPool pool,
    vk::DescriptorSetLayout layout,
    uint32_t count
) {
    std::vector<vk::DescriptorSetLayout> layouts(count, layout);
    auto [result, sets] = device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(pool)
            .setSetLayouts(layouts)
    );
    vk_expect(result, "Failed to allocate tonemap descriptor sets");
    return sets;
}

static vk::Sampler create_tonemap_sampler(vk::Device device) {
    auto [result, sampler] = device.createSampler(
        vk::SamplerCreateInfo()
            .setMagFilter(vk::Filter::eLinear)
            .setMinFilter(vk::Filter::eLinear)
            .setMipmapMode(vk::SamplerMipmapMode::eNearest)
            .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
            .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
            .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
            .setMinLod(0.0f)
            .setMaxLod(0.0f)
    );
    vk_expect(result, "Failed to create tonemap sampler");
    return sampler;
}

struct Renderer::Inner {
    static constexpr uint32_t FRAMES_IN_FLIGHT = 2;

    ~Inner() {
        this->destroy();
    }

    void initialize(const std::shared_ptr<Target>& target) {
        this->instance = create_instance(*target);
        this->swapchain = Swapchain(this->instance, target);

        auto device = create_device(this->instance, this->swapchain);
        if (!device.has_value()) {
            spdlog::critical("Failed to create a suitable device");
            std::exit(1);
        }
        this->device = device.value();

        this->swapchain_info = SwapchainConfigureInfo()
            .set_format(vk::Format::eB8G8R8A8Srgb)
            .set_colorspace(vk::ColorSpaceKHR::eSrgbNonlinear)
            .set_present_mode(vk::PresentModeKHR::eMailbox);
        this->configure_render_targets();
        this->create_sync_primitives();
        this->create_command_pool();
        this->uploader = Uploader(
            this->device,
            FRAMES_IN_FLIGHT,
            UPLOADER_CAPACITY_PER_FIF
        );
        this->texture_manager = TextureManager(
            this->device,
            this->uploader,
            FRAMES_IN_FLIGHT,
            MAX_TEXTURES
        );
        this->material_manager = MaterialManager(
            this->device,
            FRAMES_IN_FLIGHT,
            MAX_MATERIALS
        );
        this->mesh_manager = MeshManager(
            this->device,
            FRAMES_IN_FLIGHT,
            VERTEX_HEAP_CAPACITY,
            INDEX_HEAP_CAPACITY
        );
        this->frame_arena = FrameArenaBuffer(
            this->device,
            vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eIndirectBuffer,
            FRAMES_IN_FLIGHT,
            FRAME_ARENA_CAPACITY_PER_FIF
        );
        this->allocate_command_buffers();
        this->create_pipeline();
        this->setup_tonemap();
        this->create_tonemap_pipeline();
    }

    void destroy() {
        this->device->wait_idle();
        this->device->logical.destroyPipeline(this->tonemap_pipeline);
        this->device->logical.destroyPipelineLayout(this->tonemap_pipeline_layout);
        this->device->logical.destroyPipeline(this->pipeline);
        this->device->logical.destroyPipelineLayout(this->pipeline_layout);
        this->device->logical.freeCommandBuffers(this->command_pool, this->command_buffers);
        this->device->logical.destroyCommandPool(this->command_pool);
        this->destroy_sync_primitives();
        this->destroy_target_textures();
        this->destroy_tonemap();
    }

    void resize() {
        this->should_reconfigure_swapchain = true;
    }

    TextureId add_texture(const Image& image) {
        return this->texture_manager.add(this->uploader, image);
    }

    void set_texture(TextureId id, const Image& image) {
        this->texture_manager.set(id, this->uploader, image);
    }

    void free_texture(TextureId id) {
        this->texture_manager.free(id);
    }

    MaterialId add_material(const Material& material) {
        return this->material_manager.add(material);
    }

    const Material* get_material(MaterialId id) {
        return this->material_manager.get(id);
    }

    void set_material(MaterialId id, const Material& material) {
        this->material_manager.set(id, material);
    }

    void free_material(MaterialId id) {
        this->material_manager.free(id);
    }
    
    MeshId add_mesh(const Mesh& mesh) {
        return this->mesh_manager.add(this->uploader, mesh);
    }

    void set_mesh(MeshId id, const Mesh& mesh) {
        this->mesh_manager.set(id, this->uploader, mesh);
    }

    void free_mesh(MeshId id) {
        this->mesh_manager.free(id);
    }

    void draw(const Scene& scene) {
        vk::Fence fence = this->fences[this->frame_index];

        vk_expect(
            this->device->logical.waitForFences(
                fence, true,
                std::numeric_limits<uint64_t>::max()
            ),
            "Fence wait failed"
        );

        auto [result1, image] = this->swapchain.acquire_image(
            this->image_available[this->frame_index]
        );

        if (result1 == vk::Result::eErrorOutOfDateKHR) {
            this->reconfigure_render_targets();
            return;
        } else if (result1 != vk::Result::eSuboptimalKHR) {
            vk_expect(result1, "Critical swapchain image acquisition failure");
        }

        vk_expect(this->device->logical.resetFences(fence), "Fence reset failed");
        this->mesh_manager.free_pending();
        this->texture_manager.destroy_pending();
        this->material_manager.flag_dirty_materials(this->texture_manager);
        this->material_manager.update_dirty(this->texture_manager, this->frame_index);
        this->texture_manager.clear_updated();

        vk::CommandBuffer command_buffer = this->begin_frame_commands();
        this->uploader.flush(command_buffer);
        this->texture_manager.flush_mip_maps(command_buffer);
        this->record_frame_commands(command_buffer, image, scene);
        this->submit_frame_commands(command_buffer, image);

        vk::Result result2 = this->swapchain.present(image);
        if (result2 == vk::Result::eSuboptimalKHR 
            || result2 == vk::Result::eErrorOutOfDateKHR 
            || this->should_reconfigure_swapchain) 
        {
            this->reconfigure_render_targets();
        }

        this->frame_counter += 1;
        this->frame_index = this->frame_counter % FRAMES_IN_FLIGHT;

        this->uploader.begin_frame(this->frame_index);
        this->frame_arena.begin_frame(this->frame_index);
        this->texture_manager.begin_frame(this->frame_counter);
        this->mesh_manager.begin_frame(this->frame_counter);
    }

private:

    void reconfigure_render_targets() {
        this->configure_render_targets();
        this->write_tonemap_descriptors();
        this->should_reconfigure_swapchain = false;
    }

    void configure_render_targets() {
        this->device->wait_idle();
        vk_expect(
            this->swapchain.configure(this->device, this->swapchain_info),
            "Failed to configure swapchain"
        );
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            this->create_hdr_image(i);
        }
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            this->create_depth_image(i);
        }
    }

    void create_sync_primitives() {
        this->fences.resize(FRAMES_IN_FLIGHT);
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            auto [result, fence] = this->device->logical.createFence(
                vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled)
            );
            vk_expect(result, "Failed to create in-flight fence");
            this->fences[i] = fence;
        }

        this->image_available.resize(FRAMES_IN_FLIGHT);
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            auto [result, semaphore] = this->device->logical.createSemaphore(vk::SemaphoreCreateInfo());
            vk_expect(result, "Failed to create image available fence");
            this->image_available[i] = semaphore;
        }
    }

    void destroy_sync_primitives() {
        for (auto fence : this->fences) {
            this->device->logical.destroyFence(fence);
        }
        for (auto image_available : this->image_available) {
            this->device->logical.destroySemaphore(image_available);
        }
    }

    void create_command_pool() {
        auto [result, command_pool] = this->device->logical.createCommandPool(
            vk::CommandPoolCreateInfo()
                .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                .setQueueFamilyIndex(this->device->graphics_queue_family)
        );
        vk_expect(result, "Failed to create command pool");
        this->command_pool = command_pool;
    }

    void allocate_command_buffers() {
        auto [result, command_buffers] = this->device->logical.allocateCommandBuffers(
            vk::CommandBufferAllocateInfo()
                .setCommandPool(command_pool)
                .setCommandBufferCount(FRAMES_IN_FLIGHT)
                .setLevel(vk::CommandBufferLevel::ePrimary)
        );
        vk_expect(result, "Failed to allocate command buffers");
        this->command_buffers = command_buffers;
    }

    void setup_tonemap() {
        this->tonemap_sampler = create_tonemap_sampler(this->device->logical);
        this->tonemap_set_layout = create_tonemap_set_layout(this->device->logical);
        this->tonemap_descriptor_pool =
            create_tonemap_descriptor_pool(this->device->logical, FRAMES_IN_FLIGHT);
        this->tonemap_sets = allocate_tonemap_sets(
            this->device->logical,
            this->tonemap_descriptor_pool,
            this->tonemap_set_layout,
            FRAMES_IN_FLIGHT
        );
        this->write_tonemap_descriptors();
    }

    void write_tonemap_descriptors() {
        if (this->tonemap_sets.empty()) {
            return;
        }

        std::vector<vk::DescriptorImageInfo> infos;
        infos.reserve(this->tonemap_sets.size() * 2);
        std::vector<vk::WriteDescriptorSet> writes;
        writes.reserve(this->tonemap_sets.size() * 2);

        for (uint32_t i = 0; i < this->tonemap_sets.size(); i++) {
            infos.push_back(
                vk::DescriptorImageInfo()
                    .setImageView(this->hdr_textures[i].view)
                    .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
            );
            writes.push_back(
                vk::WriteDescriptorSet()
                    .setDstSet(this->tonemap_sets[i])
                    .setDstBinding(0)
                    .setDstArrayElement(0)
                    .setDescriptorType(vk::DescriptorType::eSampledImage)
                    .setImageInfo(infos.back())
            );

            infos.push_back(vk::DescriptorImageInfo().setSampler(this->tonemap_sampler));
            writes.push_back(
                vk::WriteDescriptorSet()
                    .setDstSet(this->tonemap_sets[i])
                    .setDstBinding(1)
                    .setDstArrayElement(0)
                    .setDescriptorType(vk::DescriptorType::eSampler)
                    .setImageInfo(infos.back())
            );
        }

        this->device->logical.updateDescriptorSets(writes, {});
    }

    void destroy_tonemap() {
        this->device->logical.destroyDescriptorPool(this->tonemap_descriptor_pool);
        this->device->logical.destroyDescriptorSetLayout(this->tonemap_set_layout);
        this->device->logical.destroySampler(this->tonemap_sampler);
    }

    void create_tonemap_pipeline() {
        std::vector shader_spirv = read_spirv_file(SHADER_DIRECTORY / "tonemap.spv");
        auto [result1, shader_module] =
            this->device->logical.createShaderModule(vk::ShaderModuleCreateInfo().setCode(shader_spirv));
        vk_expect(result1, "Failed to create shader module");

        std::array shader_stages = {
            vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eVertex)
                .setModule(shader_module)
                .setPName("vertex_main"),
            vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eFragment)
                .setModule(shader_module)
                .setPName("fragment_main"),
        };

        auto vertex_state = vk::PipelineVertexInputStateCreateInfo();

        auto raster_state = vk::PipelineRasterizationStateCreateInfo()
            .setPolygonMode(vk::PolygonMode::eFill)
            .setFrontFace(vk::FrontFace::eCounterClockwise)
            .setCullMode(vk::CullModeFlagBits::eBack)
            .setLineWidth(1.0f);

        auto input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo()
            .setTopology(vk::PrimitiveTopology::eTriangleList);

        vk::SampleMask sample_mask = ~static_cast<vk::SampleMask>(0);
        auto ms_state = vk::PipelineMultisampleStateCreateInfo()
            .setRasterizationSamples(vk::SampleCountFlagBits::e1)
            .setPSampleMask(&sample_mask)
            .setAlphaToCoverageEnable(false)
            .setAlphaToOneEnable(false);

        auto viewport_state = vk::PipelineViewportStateCreateInfo()
            .setScissorCount(1)
            .setViewportCount(1);

        auto blend_attachment = vk::PipelineColorBlendAttachmentState()
            .setBlendEnable(false)
            .setColorWriteMask(
                vk::ColorComponentFlagBits::eR 
                    | vk::ColorComponentFlagBits::eG 
                    | vk::ColorComponentFlagBits::eB 
                    | vk::ColorComponentFlagBits::eA
            );

        auto blend_state = vk::PipelineColorBlendStateCreateInfo()
            .setAttachments(blend_attachment);

        std::array dynamic_states = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
        };
        auto dyn_state = vk::PipelineDynamicStateCreateInfo().setDynamicStates(dynamic_states);

        std::array set_layouts = {this->tonemap_set_layout};
        std::array push_constant_ranges = {
            vk::PushConstantRange()
                .setOffset(0)
                .setSize(sizeof(TonemapConstants))
                .setStageFlags(vk::ShaderStageFlagBits::eFragment)
        };
        auto [result2, pipeline_layout] = this->device->logical.createPipelineLayout(
            vk::PipelineLayoutCreateInfo()
                .setSetLayouts(set_layouts)
                .setPushConstantRanges(push_constant_ranges)
        );
        vk_expect(result2, "Failed to create pipeline layout");

        vk::Format swapchain_format = this->swapchain.format();
        auto [result3, pipeline] = this->device->logical.createGraphicsPipeline(
            vk::PipelineCache(),
            vk::StructureChain{
                vk::GraphicsPipelineCreateInfo()
                    .setLayout(pipeline_layout)
                    .setStages(shader_stages)
                    .setPVertexInputState(&vertex_state)
                    .setPInputAssemblyState(&input_assembly_state)
                    .setPRasterizationState(&raster_state)
                    .setPMultisampleState(&ms_state)
                    .setPViewportState(&viewport_state)
                    .setPColorBlendState(&blend_state)
                    .setPDynamicState(&dyn_state),

                vk::PipelineRenderingCreateInfo()
                    .setColorAttachmentFormats(swapchain_format)
            }
            .get()
        );
        vk_expect(result3, "Failed to create graphics pipeline");
        this->device->logical.destroyShaderModule(shader_module);

        this->tonemap_pipeline = pipeline;
        this->tonemap_pipeline_layout = pipeline_layout;
    }

    void create_pipeline() {
        std::vector shader_spirv = read_spirv_file(SHADER_DIRECTORY / "pbr_forward.spv");
        auto [result1, shader_module] =
            this->device->logical.createShaderModule(vk::ShaderModuleCreateInfo().setCode(shader_spirv));
        vk_expect(result1, "Failed to create shader module");

        std::array shader_stages = {
            vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eVertex)
                .setModule(shader_module)
                .setPName("vertex_main"),
            vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eFragment)
                .setModule(shader_module)
                .setPName("fragment_main"),
        };

        auto vertex_state = vk::PipelineVertexInputStateCreateInfo();

        auto raster_state = vk::PipelineRasterizationStateCreateInfo()
            .setPolygonMode(vk::PolygonMode::eFill)
            .setFrontFace(vk::FrontFace::eCounterClockwise)
            .setCullMode(vk::CullModeFlagBits::eBack)
            .setLineWidth(1.0f);

        auto depth_state = vk::PipelineDepthStencilStateCreateInfo()
            .setDepthTestEnable(true)
            .setDepthWriteEnable(true)
            .setDepthCompareOp(vk::CompareOp::eGreaterOrEqual);

        auto input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo()
            .setTopology(vk::PrimitiveTopology::eTriangleList);

        vk::SampleMask sample_mask = ~static_cast<vk::SampleMask>(0);
        auto ms_state = vk::PipelineMultisampleStateCreateInfo()
            .setRasterizationSamples(vk::SampleCountFlagBits::e1)
            .setPSampleMask(&sample_mask)
            .setAlphaToCoverageEnable(false)
            .setAlphaToOneEnable(false);

        auto viewport_state = vk::PipelineViewportStateCreateInfo()
            .setScissorCount(1)
            .setViewportCount(1);

        auto blend_attachment = vk::PipelineColorBlendAttachmentState()
            .setBlendEnable(false)
            .setColorWriteMask(
                vk::ColorComponentFlagBits::eR 
                    | vk::ColorComponentFlagBits::eG 
                    | vk::ColorComponentFlagBits::eB 
                    | vk::ColorComponentFlagBits::eA
            );

        auto blend_state = vk::PipelineColorBlendStateCreateInfo()
            .setAttachments(blend_attachment);

        std::array dynamic_states = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
        };
        auto dyn_state = vk::PipelineDynamicStateCreateInfo().setDynamicStates(dynamic_states);

        std::array set_layouts = {this->texture_manager.descriptor_set_layout()};
        std::array push_constant_ranges = {
            vk::PushConstantRange()
                .setOffset(0)
                .setSize(sizeof(DrawConstants))
                .setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
        };
        auto [result2, pipeline_layout] = this->device->logical.createPipelineLayout(
            vk::PipelineLayoutCreateInfo()
                .setSetLayouts(set_layouts)
                .setPushConstantRanges(push_constant_ranges)
        );
        vk_expect(result2, "Failed to create pipeline layout");

        auto [result3, pipeline] = this->device->logical.createGraphicsPipeline(
            vk::PipelineCache(),
            vk::StructureChain{
                vk::GraphicsPipelineCreateInfo()
                    .setLayout(pipeline_layout)
                    .setStages(shader_stages)
                    .setPVertexInputState(&vertex_state)
                    .setPInputAssemblyState(&input_assembly_state)
                    .setPRasterizationState(&raster_state)
                    .setPDepthStencilState(&depth_state)
                    .setPMultisampleState(&ms_state)
                    .setPViewportState(&viewport_state)
                    .setPColorBlendState(&blend_state)
                    .setPDynamicState(&dyn_state),

                vk::PipelineRenderingCreateInfo()
                    .setColorAttachmentFormats(this->hdr_textures[0].format)
                    .setDepthAttachmentFormat(vk::Format::eD32Sfloat)
            }
            .get()
        );
        vk_expect(result3, "Failed to create graphics pipeline");
        this->device->logical.destroyShaderModule(shader_module);

        this->pipeline = pipeline;
        this->pipeline_layout = pipeline_layout;
    }

    void create_hdr_image(uint32_t frame_index) {
        if (this->hdr_textures.size() != FRAMES_IN_FLIGHT) {
            this->hdr_textures.resize(FRAMES_IN_FLIGHT);
        }

        Texture texture = this->create_image(
            vk::ImageCreateInfo()
                .setImageType(vk::ImageType::e2D)
                .setExtent(vk::Extent3D(this->swapchain.extent(), 1))
                .setFormat(vk::Format::eR16G16B16A16Sfloat)
                .setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled)
                .setMipLevels(1)
                .setArrayLayers(1)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setTiling(vk::ImageTiling::eOptimal)
        );

        if (!this->hdr_textures[frame_index].is_null()) {
            this->hdr_textures[frame_index].destroy(this->device);
        }

        this->hdr_textures[frame_index] = texture;
    }

    void create_depth_image(uint32_t frame_index) {
        if (this->depth_textures.size() != FRAMES_IN_FLIGHT) {
            this->depth_textures.resize(FRAMES_IN_FLIGHT);
        }

        Texture texture = this->create_image(
            vk::ImageCreateInfo()
                .setImageType(vk::ImageType::e2D)
                .setExtent(vk::Extent3D(this->swapchain.extent(), 1))
                .setFormat(vk::Format::eD32Sfloat)
                .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
                .setMipLevels(1)
                .setArrayLayers(1)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setTiling(vk::ImageTiling::eOptimal)
        );

        if (!this->depth_textures[frame_index].is_null()) {
            this->depth_textures[frame_index].destroy(this->device);
        }

        this->depth_textures[frame_index] = texture;
    }

    void destroy_target_textures() {
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            if (!this->depth_textures[i].is_null()) {
                this->depth_textures[i].destroy(this->device);
            }
        }

        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            if (!this->hdr_textures[i].is_null()) {
                this->hdr_textures[i].destroy(this->device);
            }
        }
    }

    Texture create_image(const vk::ImageCreateInfo& info) {
        return ::create_texture(this->device, info);
    }

    ViewData get_view_data(const Camera& camera) {
        vk::Extent2D viewport_size = this->swapchain.extent();
        auto viewport_width = static_cast<float>(viewport_size.width);
        auto viewport_height = static_cast<float>(viewport_size.height);

        return {
            .world_to_clip = camera.world_to_clip(viewport_width, viewport_height),
            .position = camera.transform.translation,
        };
    }

    void update_scene_draw_data(const Scene& scene) {
        this->scene_data.view = this->get_view_data(scene.camera);
        this->scene_data.clear_color = scene.clear_color;
        this->scene_data.ambient_light = {
            .color = scene.ambient_light.color,
            .illuminance = scene.ambient_light.illuminance
        };

        this->scene_data.point_lights.clear();
        for (const auto& light: scene.point_lights) {
            this->scene_data.point_lights.push_back({
                .position = light.position,
                .radius = light.radius,
                .color = light.color,
                .intensity = light.intensity
            });
        }

        this->scene_data.dir_lights.clear();
        for (const auto& light: scene.directional_lights) {
            this->scene_data.dir_lights.push_back({ 
                .direction = light.direction,
                .color = light.color,
                .illuminance = light.illuminance
            });
        }

        this->scene_data.scratch.clear();
        uint32_t object_id = 0;
        for (const auto& object: scene.mesh_objects) {
            if (this->mesh_manager.is_valid(object.mesh)) {
                this->scene_data.scratch.push_back({
                    .mesh_id = object.mesh,
                    .object_id = object_id,
                });
            }
            object_id += 1;
        }
        std::ranges::sort(
            this->scene_data.scratch,
            [](const auto& a, const auto& b) {
                return a.mesh_id.index < b.mesh_id.index;
            }
        );

        this->scene_data.instances.clear();
        for (const auto& entry: this->scene_data.scratch) {
            const auto& object = scene.mesh_objects[entry.object_id];
            this->scene_data.instances.emplace_back(
                object.transform,
                this->material_manager.get_index(object.material)
            );
        }

        this->scene_data.draws.clear();
        this->scene_data.draw_commands.clear();
        size_t i = 0;
        while (i < this->scene_data.scratch.size()) {
            const auto start = static_cast<uint32_t>(i);
            const MeshId mesh_id = this->scene_data.scratch[i].mesh_id;
            while (i < this->scene_data.scratch.size()
                && this->scene_data.scratch[i].mesh_id.index == mesh_id.index) 
            {
                i++;
            }
            MeshData mesh = this->mesh_manager.get(mesh_id);
            this->scene_data.draws.push_back({
                .vertices = mesh.vertices_address,
                .instance_offset = start
            });
            this->scene_data.draw_commands.push_back(
                vk::DrawIndexedIndirectCommand()
                    .setFirstIndex(mesh.index_offset)
                    .setIndexCount(mesh.index_count)
                    .setFirstInstance(0)
                    .setInstanceCount(i - start)
                    .setVertexOffset(0)
            );
        }
    }

    FrameData write_frame_data(const Scene& scene) {
        this->update_scene_draw_data(scene);

        FrameSubBuffer view_data = this->frame_arena.add(scene_data.view).value();
        FrameSubBuffer light_data = this->frame_arena
            .add(LightData {
                .ambient_color = this->scene_data.ambient_light.color,
                .ambient_illuminance = this->scene_data.ambient_light.illuminance,
                .point_count = static_cast<uint32_t>(this->scene_data.point_lights.size()),
                .directional_count = static_cast<uint32_t>(this->scene_data.dir_lights.size()),
            })
            .value();

        FrameSubBuffer dir_lights = this->frame_arena.add(scene_data.dir_lights).value();
        FrameSubBuffer point_lights = this->frame_arena.add(scene_data.point_lights).value();
        FrameSubBuffer instances = this->frame_arena.add(scene_data.instances).value();
        FrameSubBuffer draws = this->frame_arena.add(scene_data.draws).value();
        FrameSubBuffer draw_commands = this->frame_arena.add(scene_data.draw_commands).value();
        
        return {
            .clear_color = scene_data.clear_color,
            .view_data = view_data,
            .light_data = light_data,
            .dir_lights = dir_lights,
            .point_lights = point_lights,
            .instances = instances,
            .draws = draws,
            .draw_commands = draw_commands,
        };
    }

    void record_frame_commands(
        vk::CommandBuffer command_buffer, 
        const SwapchainImage& image,
        const Scene& scene
    ) {
        FrameData data = this->write_frame_data(scene);

        auto subresource_range = vk::ImageSubresourceRange()
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1);

        this->record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(this->hdr_textures[this->frame_index])
                .setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
                .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                .setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                .setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseMipLevel(0)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1)
                )
        );
        this->record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(this->depth_textures[this->frame_index])
                .setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
                .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                .setDstStageMask(vk::PipelineStageFlagBits2::eEarlyFragmentTests)
                .setDstAccessMask(
                    vk::AccessFlagBits2::eDepthStencilAttachmentRead
                        | vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                )
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eDepthAttachmentOptimal)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eDepth)
                        .setBaseMipLevel(0)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1)
                )
        );
        this->record_draw_commands(command_buffer, data);
        this->record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(image.image)
                .setSrcStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
                .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                .setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                .setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setSubresourceRange(subresource_range)
        );
        this->record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(this->hdr_textures[this->frame_index])
                .setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                .setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
                .setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
                .setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
                .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setBaseMipLevel(0)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1)
                )
        );
        this->record_tonemap_commands(command_buffer, image, scene);
        this->record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(image.image)
                .setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                .setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
                .setDstStageMask(vk::PipelineStageFlagBits2::eNone)
                .setDstAccessMask(vk::AccessFlagBits2::eNone)
                .setOldLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setNewLayout(vk::ImageLayout::ePresentSrcKHR)
                .setSubresourceRange(subresource_range)
        );
    }

    void record_draw_commands(vk::CommandBuffer command_buffer, const FrameData& data) {
        DrawConstants draw_constants = {
            .view_data = data.view_data.address(),
            .light_data = data.light_data.address(),
            .point_lights = data.point_lights.address(),
            .directional_lights = data.dir_lights.address(),
            .materials = this->material_manager.buffer_address(this->frame_index),
            .instances = data.instances.address(),
            .draws = data.draws.address()
        };
        command_buffer.pushConstants<DrawConstants>(
            this->pipeline_layout,
            vk::ShaderStageFlagBits::eVertex 
                | vk::ShaderStageFlagBits::eFragment,
            0, {draw_constants}
        );

        auto color_attachment = vk::RenderingAttachmentInfo()
            .setImageView(this->hdr_textures[this->frame_index].view)
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setClearValue(vk::ClearColorValue(
                data.clear_color.r, 
                data.clear_color.g, 
                data.clear_color.b, 
                1.0f
            ))
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore);

        auto depth_attachment = vk::RenderingAttachmentInfo()
            .setImageView(this->depth_textures[this->frame_index].view)
            .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
            .setClearValue(vk::ClearDepthStencilValue(0.0f))
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore);

        auto rendering = vk::RenderingInfo()
            .setLayerCount(1)
            .setColorAttachments(color_attachment)
            .setPDepthAttachment(&depth_attachment)
            .setRenderArea(this->swapchain.full_area());

        command_buffer.beginRendering(rendering);
        command_buffer.setViewport(0, this->swapchain.full_viewport());
        command_buffer.setScissor(0, this->swapchain.full_area());
        command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, this->pipeline);
        command_buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            this->pipeline_layout, 0,
            {this->texture_manager.descriptor_set()}, {}
        );
        command_buffer.bindIndexBuffer(this->mesh_manager.index_buffer(), 0, vk::IndexType::eUint32);
        if (data.draw_commands.length() > 0) {
            command_buffer.drawIndexedIndirect(
                data.draw_commands.buffer(), 
                data.draw_commands.buffer_offset(),
                static_cast<uint32_t>(data.draw_commands.length()),
                sizeof(vk::DrawIndexedIndirectCommand)
            );
        }
        command_buffer.endRendering();
    }

    void record_tonemap_commands(
        vk::CommandBuffer command_buffer, 
        const SwapchainImage& image,
        const Scene& scene
    ) {
        command_buffer.pushConstants<TonemapConstants>(
            this->tonemap_pipeline_layout,
            vk::ShaderStageFlagBits::eFragment,
            0, {TonemapConstants(scene.post.tonemap, scene.camera.film.exposure)}
        );
        auto color_attachment = vk::RenderingAttachmentInfo()
            .setImageView(image.view)
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStoreOp(vk::AttachmentStoreOp::eStore);

        auto rendering = vk::RenderingInfo()
            .setLayerCount(1)
            .setColorAttachments(color_attachment)
            .setRenderArea(this->swapchain.full_area());

        command_buffer.beginRendering(rendering);
        command_buffer.setViewport(0, this->swapchain.full_viewport());
        command_buffer.setScissor(0, this->swapchain.full_area());
        command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, this->tonemap_pipeline);
        command_buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            this->tonemap_pipeline_layout, 0,
            {this->tonemap_sets[this->frame_index]}, {}
        );
        command_buffer.draw(3, 1, 0, 0);
        command_buffer.endRendering();
    }

    void record_image_barrier(vk::CommandBuffer command_buffer, vk::ImageMemoryBarrier2 barrier) {
        command_buffer.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
    }

    vk::CommandBuffer begin_frame_commands() {
        vk::CommandBuffer command_buffer = this->command_buffers[this->frame_index];
        vk_expect(command_buffer.reset(), "Failed to reset command buffer");
        vk_expect(
            command_buffer.begin(
                vk::CommandBufferBeginInfo()
                    .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
            ),
            "Failed to begin command buffer"
        );

        return command_buffer;
    }

    void submit_frame_commands(vk::CommandBuffer command_buffer, const SwapchainImage& image) {
        vk_expect(command_buffer.end(), "Failed to end command buffer");

        auto command_buffer_info = vk::CommandBufferSubmitInfo().setCommandBuffer(command_buffer);

        auto wait_info = vk::SemaphoreSubmitInfo()
            .setSemaphore(image.available)
            .setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);

        auto signal_info = vk::SemaphoreSubmitInfo()
            .setSemaphore(image.presentable)
            .setStageMask(vk::PipelineStageFlagBits2::eAllCommands);

        vk_expect(
            this->device->graphics_queue.submit2(
                vk::SubmitInfo2()
                    .setWaitSemaphoreInfos(wait_info)
                    .setSignalSemaphoreInfos(signal_info)
                    .setCommandBufferInfos(command_buffer_info),
                this->fences[this->frame_index]
            ),
            "Failed to submit command buffer"
        );
    }

    InstanceHandle instance;
    DeviceHandle device;
    Swapchain swapchain;
    SwapchainConfigureInfo swapchain_info;
    bool should_reconfigure_swapchain = false;

    std::vector<Texture> hdr_textures;
    std::vector<Texture> depth_textures;

    std::vector<vk::Fence> fences;              // One per frame in flight
    std::vector<vk::Semaphore> image_available; // One per frame in flight

    uint32_t frame_index = 0;
    uint64_t frame_counter = 0;

    vk::CommandPool command_pool;
    std::vector<vk::CommandBuffer> command_buffers; // One per frame in flight

    vk::Pipeline pipeline;
    vk::PipelineLayout pipeline_layout;
    
    vk::Pipeline tonemap_pipeline;
    vk::PipelineLayout tonemap_pipeline_layout;
    vk::DescriptorPool tonemap_descriptor_pool;
    vk::DescriptorSetLayout tonemap_set_layout;
    std::vector<vk::DescriptorSet> tonemap_sets;
    vk::Sampler tonemap_sampler;

    Uploader uploader;
    FrameArenaBuffer frame_arena;
    TextureManager texture_manager;
    MaterialManager material_manager;
    MeshManager mesh_manager;
    SceneDrawData scene_data;
};

Renderer::Renderer() = default;

Renderer::Renderer(const std::shared_ptr<Target>& target)
    : inner(std::make_unique<Inner>()) 
{
    this->inner->initialize(target);
}

Renderer::~Renderer() = default;

Renderer::Renderer(Renderer&& other) noexcept {
    this->inner = std::move(other.inner);
}

Renderer& Renderer::operator=(Renderer&& other) noexcept {
    this->inner = std::move(other.inner);
    return *this;
}

void Renderer::resize() {
    this->inner->resize();
}

MaterialId Renderer::add_material(const Material& material) {
    return this->inner->add_material(material);
}

const Material* Renderer::get_material(MaterialId id) {
    return this->inner->get_material(id);
}

void Renderer::set_material(MaterialId id, const Material& material) {
    return this->inner->set_material(id, material);
}

void Renderer::free_material(MaterialId id) {
    return this->inner->free_material(id);
}

TextureId Renderer::add_texture(const Image& image) {
    return this->inner->add_texture(image);
}

void Renderer::set_texture(TextureId id, const Image& image) {
    return this->inner->set_texture(id, image);
}

void Renderer::free_texture(TextureId id) {
    return this->inner->free_texture(id);
}

MeshId Renderer::add_mesh(const Mesh& mesh) {
    return this->inner->add_mesh(mesh);
}

void Renderer::set_mesh(MeshId id, const Mesh& mesh) {
    this->inner->set_mesh(id, mesh);
}

void Renderer::free_mesh(MeshId id) {
    this->inner->free_mesh(id);
}

void Renderer::draw(const Scene& scene) {
    this->inner->draw(scene);
}