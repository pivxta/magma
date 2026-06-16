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
#include "render_target_manager.h"
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
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.hpp>

static constexpr uint32_t MAX_RENDER_TARGETS = 256;
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
    RenderTargetIndices input;
    TonemapType type;
    float exposure;

    AgxConstants agx;

    TonemapConstants(
        RenderTargetIndices input,
        const Tonemap& tonemap, 
        float film_exposure
    ) {
        this->input = input;
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

struct ShadowConstants {
    glm::mat4 light_world_to_clip;
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
    bool shadows_enabled;
    RenderTargetIndices shadow_map;
};

struct DirectionalLightData {
    alignas(16) glm::vec3 direction;
    alignas(16) glm::vec3 color;
    float illuminance;
    bool shadows_enabled;
    glm::mat4 world_to_shadow_clip;
    RenderTargetIndices shadow_map;
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

    Aabb bounds;
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

static glm::mat4 directional_light_matrix(glm::vec3 light_direction, const Aabb& world_bounds) {
    const auto dir = glm::normalize(light_direction);
    const auto up = glm::abs(dir.y) > 0.99f ? 
        glm::vec3(0.0f, 0.0f, 1.0f) : 
        glm::vec3(0.0f, 1.0f, 0.0f);

    glm::vec3 center = world_bounds.center();
    glm::mat4 light_view = glm::lookAtRH(center, center + dir, up);

    Aabb ls = light_view * world_bounds;
    glm::mat4 light_proj = glm::orthoRH_ZO(
        ls.min.x, ls.max.x,
        ls.min.y, ls.max.y,
        -ls.min.z, -ls.max.z
    );
    return light_proj * light_view;
}

struct Renderer::Inner {
    static constexpr uint32_t FRAMES_IN_FLIGHT = 2;
    InstanceHandle instance;
    DeviceHandle device;
    Swapchain swapchain;
    SwapchainConfigureInfo swapchain_info;
    bool should_reconfigure_swapchain = false;

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

    vk::Pipeline shadow_pipeline;
    vk::PipelineLayout shadow_pipeline_layout;

    RenderTargetId swapchain_target;
    RenderTargetId hdr_target;
    RenderTargetId depth_target;
    RenderTargetId shadow_target;
    vk::Extent2D shadow_resolution = vk::Extent2D(1024, 1024);

    Uploader uploader;
    FrameArenaBuffer frame_arena;
    RenderTargetManager targets;
    TextureManager textures;
    MaterialManager materials;
    MeshManager meshes;
    SceneDrawData scene_data;

    ~Inner() {
        this->device->wait_idle();
        this->device->logical.destroyPipeline(this->shadow_pipeline);
        this->device->logical.destroyPipelineLayout(this->shadow_pipeline_layout);
        this->device->logical.destroyPipeline(this->tonemap_pipeline);
        this->device->logical.destroyPipelineLayout(this->tonemap_pipeline_layout);
        this->device->logical.destroyPipeline(this->pipeline);
        this->device->logical.destroyPipelineLayout(this->pipeline_layout);
        this->device->logical.freeCommandBuffers(this->command_pool, this->command_buffers);
        this->device->logical.destroyCommandPool(this->command_pool);
        this->destroy_sync_primitives();
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
        this->targets = RenderTargetManager(
            this->device,
            FRAMES_IN_FLIGHT,
            MAX_RENDER_TARGETS
        );
        this->textures = TextureManager(
            this->device,
            this->uploader,
            FRAMES_IN_FLIGHT,
            MAX_TEXTURES
        );
        this->materials = MaterialManager(
            this->device,
            FRAMES_IN_FLIGHT,
            MAX_MATERIALS
        );
        this->meshes = MeshManager(
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
        this->create_render_targets();
        this->create_pipeline();
        this->create_tonemap_pipeline();
        this->create_shadow_pipeline();
        this->shadow_target = this->targets.add(
            RenderTargetInfo()
                .set_buffering(RenderTargetBuffering::Shared)
                .set_size_policy(FixedSizePolicy(this->shadow_resolution))
                .set_format(vk::Format::eD32Sfloat)
        ).value();
    }

    void create_render_targets() {
        this->swapchain.configure(this->device, this->swapchain_info);
        this->targets.resize_swapchain(this->swapchain.extent());
        this->swapchain_target = this->targets.reserve().value();
        this->depth_target = this->targets.add(
            RenderTargetInfo()
                .set_buffering(RenderTargetBuffering::PerFif)
                .set_size_policy(SwapchainAdjustedSizePolicy())
                .set_format(vk::Format::eD32Sfloat)
        ).value();
        this->hdr_target = this->targets.add(
            RenderTargetInfo()
                .set_buffering(RenderTargetBuffering::PerFif)
                .set_size_policy(SwapchainAdjustedSizePolicy())
                .set_format(vk::Format::eR16G16B16A16Sfloat)
        ).value();
    }

    void reconfigure_render_targets() {
        this->configure_render_targets();
        this->should_reconfigure_swapchain = false;
    }

    void configure_render_targets() {
        this->device->wait_idle();
        vk_expect(
            this->swapchain.configure(this->device, this->swapchain_info),
            "Failed to configure swapchain"
        );
        this->targets.resize_swapchain(this->swapchain.extent());
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

        std::array set_layouts = {
            this->textures.descriptor_set_layout(),
            this->targets.descriptor_set_layout()
        };
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

    void create_shadow_pipeline() {
        std::vector shader_spirv = read_spirv_file(SHADER_DIRECTORY / "shadow.spv");
        auto [result1, shader_module] =
            this->device->logical.createShaderModule(vk::ShaderModuleCreateInfo().setCode(shader_spirv));
        vk_expect(result1, "Failed to create shader module");

        std::array shader_stages = {
            vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eVertex)
                .setModule(shader_module)
                .setPName("vertex_main")
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

        std::array set_layouts = {this->textures.descriptor_set_layout()};
        std::array push_constant_ranges = {
            vk::PushConstantRange()
                .setOffset(0)
                .setSize(sizeof(ShadowConstants))
                .setStageFlags(vk::ShaderStageFlagBits::eVertex)
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
                    .setDepthAttachmentFormat(vk::Format::eD32Sfloat)
            }
            .get()
        );
        vk_expect(result3, "Failed to create graphics pipeline");
        this->device->logical.destroyShaderModule(shader_module);

        this->shadow_pipeline = pipeline;
        this->shadow_pipeline_layout = pipeline_layout;
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

        std::array set_layouts = {
            this->textures.descriptor_set_layout(),
            this->targets.descriptor_set_layout()
        };
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

        vk::Format color_format = this->targets.get_texture(this->hdr_target)->format();
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
                    .setColorAttachmentFormats(color_format)
                    .setDepthAttachmentFormat(vk::Format::eD32Sfloat)
            }
            .get()
        );
        vk_expect(result3, "Failed to create graphics pipeline");
        this->device->logical.destroyShaderModule(shader_module);

        this->pipeline = pipeline;
        this->pipeline_layout = pipeline_layout;
    }

    void draw(const Scene& scene) {
        this->update_scene_draw_data(scene);

        vk::Fence fence = this->fences[this->frame_index];

        vk_expect(
            this->device->logical.waitForFences(
                fence, true,
                std::numeric_limits<uint64_t>::max()
            ),
            "Fence wait failed"
        );

        auto [result1, swapchain_texture] = this->swapchain.acquire_texture(
            this->image_available[this->frame_index]
        );

        if (result1 == vk::Result::eErrorOutOfDateKHR) {
            this->reconfigure_render_targets();
            return;
        } else if (result1 != vk::Result::eSuboptimalKHR) {
            vk_expect(result1, "Critical swapchain image acquisition failure");
        }

        vk_expect(this->device->logical.resetFences(fence), "Fence reset failed");
        this->meshes.free_pending();
        this->textures.update_pending();
        this->targets.destroy_pending();
        this->materials.flag_dirty_materials(this->textures);
        this->materials.update_dirty(this->textures, this->frame_index);
        this->textures.clear_updated();
        this->targets.bind(this->swapchain_target, *swapchain_texture.texture);

        vk::CommandBuffer command_buffer = this->begin_frame_commands();
        this->uploader.flush(command_buffer);
        this->textures.flush_mip_maps(command_buffer);
        this->record_frame_commands(command_buffer, swapchain_texture, scene);
        this->submit_frame_commands(command_buffer, swapchain_texture);

        vk::Result result2 = this->swapchain.present(swapchain_texture);
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
        this->textures.begin_frame(this->frame_counter);
        this->meshes.begin_frame(this->frame_counter);
        this->targets.begin_frame(this->frame_counter);
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

        this->scene_data.bounds = Aabb();
        for (const auto& object: scene.mesh_objects) {
            if (!this->meshes.is_valid(object.mesh)) {
                continue;
            }
            this->scene_data.bounds.merge(
                object.transform.affine_matrix() * this->meshes.get_bounds(object.mesh)
            );
        }

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
        uint32_t dir_light_id = 0;
        for (const auto& light: scene.directional_lights) {
            this->scene_data.dir_lights.push_back({ 
                .direction = light.direction,
                .color = light.color,
                .illuminance = light.illuminance,
                .shadows_enabled = dir_light_id == 0,
                .world_to_shadow_clip = dir_light_id == 0 ?
                    directional_light_matrix(light.direction, this->scene_data.bounds) :
                    glm::identity<glm::mat4>(),
                .shadow_map = dir_light_id == 0 ? 
                    this->targets.get_indices(this->shadow_target, Sampler::LinearRepeat).value() :
                    RenderTargetIndices {}
            });
            dir_light_id++;
        }

        this->scene_data.scratch.clear();
        uint32_t object_id = 0;
        for (const auto& object: scene.mesh_objects) {
            if (this->meshes.is_valid(object.mesh)) {
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
                this->materials.get_index(object.material)
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
            MeshDrawData mesh = this->meshes.get_draw_data(mesh_id);
            this->scene_data.draws.push_back({
                .vertices = mesh.vertices_address,
                .instance_offset = start
            });
            this->scene_data.draw_commands.push_back(
                vk::DrawIndexedIndirectCommand()
                    .setFirstIndex(mesh.index_offset)
                    .setIndexCount(mesh.index_count)
                    .setFirstInstance(0)
                    .setInstanceCount(static_cast<uint32_t>(i) - start)
                    .setVertexOffset(0)
            );
        }
    }

    FrameData write_frame_data() {
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

        this->frame_arena.flush();
        
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
        vk::CommandBuffer cmd, 
        const SwapchainTexture& swapchain_texture,
        const Scene& scene
    ) {
        FrameData data = this->write_frame_data();

        this->record_shadow_commands(cmd, data);
        this->record_draw_commands(cmd, data);
        this->record_tonemap_commands(cmd, swapchain_texture, scene);
    }

    void record_shadow_commands(vk::CommandBuffer cmd, const FrameData& data) {
        this->targets.use(cmd, this->shadow_target, RenderTargetUsage::depth_attachment());

        glm::mat4 light_matrix = this->scene_data.dir_lights.empty()
            ? glm::mat4(1.0f)
            : directional_light_matrix(
                this->scene_data.dir_lights.front().direction,
                this->scene_data.bounds
            );

        ShadowConstants shadow_constants = {
            .light_world_to_clip = light_matrix,
            .instances = data.instances.address(),
            .draws = data.draws.address(),
        };
        cmd.pushConstants<ShadowConstants>(
            this->shadow_pipeline_layout,
            vk::ShaderStageFlagBits::eVertex,
            0, {shadow_constants}
        );

        auto depth_attachment = vk::RenderingAttachmentInfo()
            .setImageView(this->targets.get_texture(this->shadow_target)->default_view())
            .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
            .setClearValue(vk::ClearDepthStencilValue(0.0f))
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore);

        auto rendering = vk::RenderingInfo()
            .setLayerCount(1)
            .setPDepthAttachment(&depth_attachment)
            .setRenderArea(vk::Rect2D(vk::Offset2D(), this->shadow_resolution));

        cmd.beginRendering(rendering);
        cmd.setViewport(0, vk::Viewport(
            0.0f, 0.0f, 
            static_cast<float>(this->shadow_resolution.width),
            static_cast<float>(this->shadow_resolution.height),
            0.0f, 1.0f
        ));
        cmd.setScissor(0, vk::Rect2D(vk::Offset2D(), this->shadow_resolution));
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, this->shadow_pipeline);
        cmd.bindIndexBuffer(this->meshes.index_buffer(), 0, vk::IndexType::eUint32);
        if (data.draw_commands.length() > 0) {
            cmd.drawIndexedIndirect(
                data.draw_commands.buffer(), 
                data.draw_commands.buffer_offset(),
                static_cast<uint32_t>(data.draw_commands.length()),
                sizeof(vk::DrawIndexedIndirectCommand)
            );
        }
        cmd.endRendering();
    }

    void record_draw_commands(vk::CommandBuffer cmd, const FrameData& data) {
        this->targets.use(cmd, this->hdr_target, RenderTargetUsage::color_attachment());
        this->targets.use(cmd, this->depth_target, RenderTargetUsage::depth_attachment());
        this->targets.use(cmd, this->shadow_target, RenderTargetUsage::shader_read());

        DrawConstants draw_constants = {
            .view_data = data.view_data.address(),
            .light_data = data.light_data.address(),
            .point_lights = data.point_lights.address(),
            .directional_lights = data.dir_lights.address(),
            .materials = this->materials.buffer_address(this->frame_index),
            .instances = data.instances.address(),
            .draws = data.draws.address()
        };
        cmd.pushConstants<DrawConstants>(
            this->pipeline_layout,
            vk::ShaderStageFlagBits::eVertex 
                | vk::ShaderStageFlagBits::eFragment,
            0, {draw_constants}
        );

        auto color_attachment = vk::RenderingAttachmentInfo()
            .setImageView(this->targets.get_texture(this->hdr_target)->default_view())
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
            .setImageView(this->targets.get_texture(this->depth_target)->default_view())
            .setImageLayout(vk::ImageLayout::eDepthAttachmentOptimal)
            .setClearValue(vk::ClearDepthStencilValue(0.0f))
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore);

        auto rendering = vk::RenderingInfo()
            .setLayerCount(1)
            .setColorAttachments(color_attachment)
            .setPDepthAttachment(&depth_attachment)
            .setRenderArea(this->swapchain.full_area());

        cmd.beginRendering(rendering);
        cmd.setViewport(0, this->swapchain.full_viewport());
        cmd.setScissor(0, this->swapchain.full_area());
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, this->pipeline);
        cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            this->pipeline_layout, 0,
            {
                this->textures.descriptor_set(),
                this->targets.descriptor_set()
            }, 
            {}
        );
        cmd.bindIndexBuffer(this->meshes.index_buffer(), 0, vk::IndexType::eUint32);
        if (data.draw_commands.length() > 0) {
            cmd.drawIndexedIndirect(
                data.draw_commands.buffer(), 
                data.draw_commands.buffer_offset(),
                static_cast<uint32_t>(data.draw_commands.length()),
                sizeof(vk::DrawIndexedIndirectCommand)
            );
        }
        cmd.endRendering();
    }

    void record_tonemap_commands(
        vk::CommandBuffer cmd, 
        const SwapchainTexture& swapchain_texture,
        const Scene& scene
    ) {
        this->targets.use(cmd, this->hdr_target, RenderTargetUsage::shader_read());
        this->targets.use(cmd, this->swapchain_target, RenderTargetUsage::color_attachment());

        cmd.pushConstants<TonemapConstants>(
            this->tonemap_pipeline_layout,
            vk::ShaderStageFlagBits::eFragment,
            0, {TonemapConstants(
                this->targets.get_indices(this->hdr_target).value(), 
                scene.post.tonemap, 
                scene.camera.film.exposure
            )}
        );
        auto color_attachment = vk::RenderingAttachmentInfo()
            .setImageView(swapchain_texture.texture->default_view())
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStoreOp(vk::AttachmentStoreOp::eStore);

        auto rendering = vk::RenderingInfo()
            .setLayerCount(1)
            .setColorAttachments(color_attachment)
            .setRenderArea(this->swapchain.full_area());

        cmd.beginRendering(rendering);
        cmd.setViewport(0, this->swapchain.full_viewport());
        cmd.setScissor(0, this->swapchain.full_area());
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, this->tonemap_pipeline);
        cmd.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            this->tonemap_pipeline_layout, 0,
            {
                this->textures.descriptor_set(),
                this->targets.descriptor_set()
            }, 
            {}
        );
        cmd.draw(3, 1, 0, 0);
        cmd.endRendering();

        this->targets.use(
            cmd,
            this->swapchain_target,
            RenderTargetUsage::present()
        );
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

    void submit_frame_commands(vk::CommandBuffer cmd, const SwapchainTexture& swapchain_texture) {
        vk_expect(cmd.end(), "Failed to end command buffer");

        auto command_buffer_info = vk::CommandBufferSubmitInfo().setCommandBuffer(cmd);

        auto wait_info = vk::SemaphoreSubmitInfo()
            .setSemaphore(swapchain_texture.available)
            .setStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput);

        auto signal_info = vk::SemaphoreSubmitInfo()
            .setSemaphore(swapchain_texture.presentable)
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
    this->inner->should_reconfigure_swapchain = true;
}

MaterialId Renderer::add_material(const Material& material) {
    return this->inner->materials.add(material);
}

const Material* Renderer::get_material(MaterialId id) const {
    return this->inner->materials.get(id);
}

void Renderer::set_material(MaterialId id, const Material& material) {
    return this->inner->materials.set(id, material);
}

void Renderer::free_material(MaterialId id) {
    return this->inner->materials.free(id);
}

TextureId Renderer::add_texture(const Image& image) {
    return this->inner->textures.add(this->inner->uploader, image);
}

void Renderer::set_texture(TextureId id, const Image& image) {
    return this->inner->textures.set(id, this->inner->uploader, image);
}

void Renderer::free_texture(TextureId id) {
    return this->inner->textures.free(id);
}

MeshId Renderer::add_mesh(const Mesh& mesh) {
    return this->inner->meshes.add(this->inner->uploader, mesh);
}

void Renderer::set_mesh(MeshId id, const Mesh& mesh) {
    this->inner->meshes.set(id, this->inner->uploader, mesh);
}

void Renderer::free_mesh(MeshId id) {
    this->inner->meshes.free(id);
}

void Renderer::draw(const Scene& scene) {
    this->inner->draw(scene);
}