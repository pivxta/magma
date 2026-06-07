#define VMA_IMPLEMENTATION
#include "renderer.h"
#include "swapchain.h"
#include "vk_error.h"
#include "uploader.h"
#include "target.h"
#include "heap_buffer.h"
#include "frame_arena_buffer.h"
#include "camera.h"
#include "image.h"
#include "mesh.h"
#include "texture.h"
#include "buffer.h"
#include "texture_manager.h"
#include "material_manager.h"
#include "time.h"
#include <vector>
#include <optional>
#include <iostream>
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
static constexpr auto FRAME_ARENA_CAPACITY_PER_FIF = vk::DeviceSize(16 * 1024 * 1024);
static constexpr auto UPLOADER_CAPACITY_PER_FIF = vk::DeviceSize(64 * 1024 * 1024);

struct DrawConstants {
    vk::DeviceAddress view_data;
    vk::DeviceAddress light_data;
    vk::DeviceAddress point_lights;
    vk::DeviceAddress directional_lights;
    vk::DeviceAddress instances;
    vk::DeviceAddress vertices;
    vk::DeviceAddress materials;
};

struct VertexData {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;
    glm::vec2 tex_coords;
};

static std::optional<std::vector<VertexData>> interlace_mesh_attributes(const Mesh& mesh) {
    if (mesh.positions.size() != mesh.normals.size()
        || mesh.positions.size() != mesh.tex_coords.size()
        || mesh.positions.size() != mesh.tangents.size())
    {
        spdlog::error("Failed to load mesh, the length of the vertex attribute arrays must the same for all attributes");
        return std::nullopt;
    }

    std::vector<VertexData> vertices;
    vertices.reserve(mesh.positions.size());
    for (size_t i = 0; i < mesh.positions.size(); i++) {
        vertices.push_back({
            .position = mesh.positions[i],
            .normal = mesh.normals[i],
            .tangent = mesh.tangents[i],
            .tex_coords = mesh.tex_coords[i]
        });
    }

    return vertices;
}

struct InstanceData {
    glm::mat4x3 local_to_world;
    uint32_t material_index;

    InstanceData(
        uint32_t material_index,
        glm::vec3 translation, 
        glm::quat rotation = glm::identity<glm::quat>(), 
        float scale = 1.0f
    ) {
        this->material_index = material_index;
        this->local_to_world = glm::mat4x3(scale * glm::mat3_cast(rotation));
        this->local_to_world[3] = translation;
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

struct FrameData {
    FrameSubBuffer<ViewData> view_data;
    FrameSubBuffer<LightData> light_data;
    FrameSubBuffer<DirectionalLightData> dir_lights;
    FrameSubBuffer<PointLightData> point_lights;
    FrameSubBuffer<InstanceData> instances;
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


struct Renderer::Inner {
    static constexpr uint32_t FRAMES_IN_FLIGHT = 2;

    ~Inner() {
        this->destroy();
    }

    void initialize(const std::shared_ptr<Target>& target) {
        this->create_instance(*target);
        this->swapchain = Swapchain(this->instance, target);
        this->swapchain_info = SwapchainConfigureInfo()
            .set_format(vk::Format::eB8G8R8A8Srgb)
            .set_colorspace(vk::ColorSpaceKHR::eSrgbNonlinear)
            .set_present_mode(vk::PresentModeKHR::eMailbox);
        this->pick_physical_device();
        this->create_device_and_queue();
        this->create_allocator();
        this->configure_render_targets();
        this->create_sync_primitives();
        this->create_command_pool();
        this->uploader = Uploader(
            this->device,
            this->allocator,
            FRAMES_IN_FLIGHT,
            UPLOADER_CAPACITY_PER_FIF
        );
        this->texture_manager = TextureManager(
            this->device,
            this->allocator,
            this->uploader,
            FRAMES_IN_FLIGHT,
            MAX_TEXTURES
        );
        this->material_manager = MaterialManager(
            this->device,
            this->allocator,
            FRAMES_IN_FLIGHT,
            MAX_MATERIALS
        );
        this->frame_arena = FrameArenaBuffer(
            this->device,
            this->allocator,
            vk::BufferUsageFlagBits::eStorageBuffer,
            FRAMES_IN_FLIGHT,
            FRAME_ARENA_CAPACITY_PER_FIF
        );
        this->allocate_command_buffers(); 
        this->create_buffers();
        this->create_pipeline(read_spirv_file("bin/shaders/shader.spv"));
    }

    void destroy() {
        vk_expect(this->device.waitIdle(), "Wait for device failed");
        this->device.destroyPipeline(this->pipeline);
        this->device.destroyPipelineLayout(this->pipeline_layout);
        this->device.freeCommandBuffers(this->command_pool, this->command_buffers);
        this->device.destroyCommandPool(this->command_pool);
        this->destroy_sync_primitives();
        this->destroy_depth_textures();
        this->destroy_buffers();
        this->uploader.destroy();
        this->frame_arena.destroy();
        this->texture_manager.destroy();
        this->material_manager.destroy();
        this->allocator.destroy();
        this->swapchain.destroy(this->device);
        this->device.destroy();
        this->instance.destroy();
    }

    void set_camera(const Camera& camera) {
        this->camera = camera;
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

    void use_material(MaterialId id) {
        this->material = id;
    }
    
    void add_mesh(const Mesh& mesh) {
        this->create_mesh_buffers(mesh);
    }

    void draw_frame() {
        vk::Fence fence = this->fences[this->frame_index];

        vk_expect(
            device.waitForFences(
                fence, true, 
                std::numeric_limits<uint64_t>::max()
            ),
            "Fence wait failed"
        );

        auto [result1, image] = this->swapchain.acquire_image(
            this->device, 
            this->image_available[this->frame_index]
        );

        vk_expect(device.resetFences(fence), "Fence reset failed");

        if (result1 == vk::Result::eErrorOutOfDateKHR) {
            this->reconfigure_render_targets();
            return;
        } else if (result1 != vk::Result::eSuboptimalKHR) {
            vk_expect(result1, "Critical swapchain image acquisition failure");
        }

        this->static_buffer.free_pending();
        this->texture_manager.destroy_pending();
        this->material_manager.flag_dirty_materials(this->texture_manager);
        this->material_manager.update_dirty(this->texture_manager, this->frame_index);
        this->texture_manager.clear_updated();

        FrameData frame_data = this->write_frame_data();

        vk::CommandBuffer command_buffer = this->begin_frame_commands();
        this->uploader.flush(command_buffer);
        this->texture_manager.flush_mip_maps(command_buffer);
        this->record_frame_commands(command_buffer, image, frame_data);
        this->submit_frame_commands(command_buffer, image);

        vk::Result result2 = this->swapchain.present(this->queue, image);
        if (result2 == vk::Result::eSuboptimalKHR 
            || result2 == vk::Result::eErrorOutOfDateKHR 
            || this->should_reconfigure_swapchain) 
        {
            this->reconfigure_render_targets();
        }

        this->frame_counter += 1;
        this->frame_index = this->frame_counter % FRAMES_IN_FLIGHT;

        this->uploader.begin_frame(this->frame_index);
        this->texture_manager.begin_frame(this->frame_counter);
        this->frame_arena.begin_frame(this->frame_index);
        this->static_buffer.begin_frame(this->frame_counter);
    }

private:
    void create_instance(const Target& target) {
        std::array instance_layers = {"VK_LAYER_KHRONOS_validation"};
        std::vector instance_extensions = target.required_instance_extensions();

        auto application = vk::ApplicationInfo().setApiVersion(vk::ApiVersion13);
        auto [result, instance] = vk::createInstance(
            vk::InstanceCreateInfo()
                .setPEnabledExtensionNames(instance_extensions)
                .setPEnabledLayerNames(instance_layers)
                .setPApplicationInfo(&application)
        );
        vk_expect(result, "Failed to create instance");
        this->instance = instance;
    }

    void pick_physical_device() {
        auto [result, devices] = this->instance.enumeratePhysicalDevices();
        vk_expect(result, "Failed to enumerate physical devices");

        for (auto device : devices) {
            vk::PhysicalDeviceProperties2 props = device.getProperties2();
            if (props.properties.apiVersion < vk::ApiVersion13) {
                continue;
            }
            if (!device_has_swapchain_ext(device)) {
                continue;
            }
            auto queue_family_index = pick_queue_family(device, this->swapchain.get_surface());
            if (queue_family_index != std::nullopt) {
                spdlog::info(
                    "Selected device: '{}', device ID: {}, vendor ID: {}",
                    props.properties.deviceName.data(),
                    props.properties.deviceID,
                    props.properties.vendorID
                );
                spdlog::info(
                    "Vulkan API version: {}.{}.{}",
                    vk::versionMajor(props.properties.apiVersion),
                    vk::versionMinor(props.properties.apiVersion),
                    vk::versionPatch(props.properties.apiVersion)
                );
                this->physical_device = device;
                this->queue_family_index = *queue_family_index;
                return;
            }
        }

        spdlog::critical("Failed to find suitable physical device");
        std::exit(1);
    }

    static bool device_has_swapchain_ext(vk::PhysicalDevice device) {
        auto [result, extensions] = device.enumerateDeviceExtensionProperties();
        vk_expect(result, "Device extension properties not available");

        for (auto extension : extensions) {
            if (strcmp(extension.extensionName, vk::KHRSwapchainExtensionName) == 0) {
                return true;
            }
        }

        return false;
    }

    static std::optional<uint32_t> pick_queue_family(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
        std::optional<uint32_t> selected_family = std::nullopt;
        std::vector queue_families = device.getQueueFamilyProperties();
        for (uint32_t current_family = 0; current_family < queue_families.size();
             current_family++) {
            vk::QueueFamilyProperties properties = queue_families[current_family];
            if (!(properties.queueFlags & vk::QueueFlagBits::eGraphics)) {
                continue;
            }

            auto [result, support] = device.getSurfaceSupportKHR(current_family, surface);
            if (result != vk::Result::eSuccess || !support) {
                continue;
            }

            selected_family = current_family;
        }

        return selected_family;
    }

    void create_device_and_queue() {
        std::array queue_priorities = {1.0f};
        std::array enabled_extensions = {vk::KHRSwapchainExtensionName};
        auto features = vk::PhysicalDeviceFeatures().setSamplerAnisotropy(true);

        auto [result, device] = this->physical_device.createDevice(
            vk::StructureChain{
                vk::DeviceCreateInfo()
                    .setPEnabledFeatures(&features)
                    .setPEnabledExtensionNames(enabled_extensions)
                    .setQueueCreateInfos(
                        vk::DeviceQueueCreateInfo()
                            .setQueuePriorities(queue_priorities)
                            .setQueueFamilyIndex(this->queue_family_index)
                    ),

                vk::PhysicalDeviceVulkan13Features()
                    .setDynamicRendering(true)
                    .setSynchronization2(true),
                
                vk::PhysicalDeviceVulkan12Features()
                    .setScalarBlockLayout(true)
                    .setBufferDeviceAddress(true)
                    .setDescriptorIndexing(true)
                    .setRuntimeDescriptorArray(true)
                    .setDescriptorBindingPartiallyBound(true)
                    .setDescriptorBindingSampledImageUpdateAfterBind(true)
                    .setShaderSampledImageArrayNonUniformIndexing(true),

                vk::PhysicalDeviceVulkan11Features().setShaderDrawParameters(true)
            }.get()
        );
        vk_expect(result, "Failed to create device");

        this->device = device;
        this->queue = device.getQueue(queue_family_index, 0);
    }

    void reconfigure_render_targets() {
        this->configure_render_targets();
        this->should_reconfigure_swapchain = false;
    }

    void configure_render_targets() {
        this->swapchain.configure(this->physical_device, this->device, this->swapchain_info);
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            this->create_depth_image(i);
        }
    }

    void create_allocator() {
        auto vma_vk_functions = vma::VulkanFunctions()
            .setVkGetInstanceProcAddr(vkGetInstanceProcAddr)
            .setVkGetDeviceProcAddr(vkGetDeviceProcAddr);

        auto vma_info = vma::AllocatorCreateInfo()
            .setFlags(vma::AllocatorCreateFlagBits::eBufferDeviceAddress)
            .setInstance(this->instance)
            .setDevice(this->device)
            .setPhysicalDevice(this->physical_device)
            .setPVulkanFunctions(&vma_vk_functions);

        auto [result, allocator] = vma::createAllocator(vma_info);
        vk_expect(result, "Failed to create allocator");
        this->allocator = allocator;
    }

    void create_sync_primitives() {
        this->fences.resize(FRAMES_IN_FLIGHT);
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            auto [result, fence] = this->device.createFence(
                vk::FenceCreateInfo().setFlags(vk::FenceCreateFlagBits::eSignaled)
            );
            vk_expect(result, "Failed to create in-flight fence");
            this->fences[i] = fence;
        }

        this->image_available.resize(FRAMES_IN_FLIGHT);
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            auto [result, semaphore] = this->device.createSemaphore(vk::SemaphoreCreateInfo());
            vk_expect(result, "Failed to create image available fence");
            this->image_available[i] = semaphore;
        }
    }

    void destroy_sync_primitives() {
        for (auto fence : this->fences) {
            this->device.destroyFence(fence);
        }
        for (auto image_available : this->image_available) {
            this->device.destroySemaphore(image_available);
        }
    }

    void create_command_pool() {
        auto [result, command_pool] = this->device.createCommandPool(
            vk::CommandPoolCreateInfo()
                .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                .setQueueFamilyIndex(queue_family_index)
        );
        vk_expect(result, "Failed to create command pool");
        this->command_pool = command_pool;
    }

    void allocate_command_buffers() {
        auto [result, command_buffers] = this->device.allocateCommandBuffers(
            vk::CommandBufferAllocateInfo()
                .setCommandPool(command_pool)
                .setCommandBufferCount(FRAMES_IN_FLIGHT)
                .setLevel(vk::CommandBufferLevel::ePrimary)
        );
        vk_expect(result, "Failed to allocate command buffers");
        this->command_buffers = command_buffers;
    }

    void create_pipeline(std::vector<uint32_t>&& shader_spirv) {
        auto [result1, shader_module] =
            this->device.createShaderModule(vk::ShaderModuleCreateInfo().setCode(shader_spirv));
        vk_expect(result1, "Failed to create shader module");

        std::array shader_stages = {
            vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eVertex)
                .setModule(shader_module)
                .setPName("vertexMain"),
            vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eFragment)
                .setModule(shader_module)
                .setPName("fragmentMain"),
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
            //this->frame_set_layout,
            this->texture_manager.descriptor_set_layout(),
        };
        std::array push_constant_ranges = {
            vk::PushConstantRange()
                .setOffset(0)
                .setSize(sizeof(DrawConstants))
                .setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
        };
        auto [result2, pipeline_layout] = this->device.createPipelineLayout(
            vk::PipelineLayoutCreateInfo()
                .setSetLayouts(set_layouts)
                .setPushConstantRanges(push_constant_ranges)
        );
        vk_expect(result2, "Failed to create pipeline layout");

        vk::Format color_attachment_format = this->swapchain.get_format();
        auto [result3, pipeline] = this->device.createGraphicsPipeline(
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
                    .setColorAttachmentFormats(color_attachment_format)
                    .setDepthAttachmentFormat(vk::Format::eD32Sfloat)
            }
            .get()
        );
        vk_expect(result3, "Failed to create graphics pipeline");
        this->device.destroyShaderModule(shader_module);

        this->pipeline = pipeline;
        this->pipeline_layout = pipeline_layout;
    }

    void create_depth_image(uint32_t frame_index) {
        if (this->depth_textures.size() != FRAMES_IN_FLIGHT) {
            this->depth_textures.resize(FRAMES_IN_FLIGHT);
        }

        Texture texture = this->create_image(
            vk::ImageCreateInfo()
                .setImageType(vk::ImageType::e2D)
                .setExtent(vk::Extent3D(this->swapchain.get_extent(), 1))
                .setFormat(vk::Format::eD32Sfloat)
                .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
                .setMipLevels(1)
                .setArrayLayers(1)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setQueueFamilyIndices(this->queue_family_index)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setTiling(vk::ImageTiling::eOptimal)
        );

        if (!this->depth_textures[frame_index].is_null()) {
            this->depth_textures[frame_index].destroy(this->device, this->allocator);
        }

        this->depth_textures[frame_index] = texture;
    }

    void destroy_depth_textures() {
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            if (!this->depth_textures[i].is_null()) {
                this->depth_textures[i].destroy(this->device, this->allocator);
            }
        }
    }

    void create_mesh_buffers(const Mesh& mesh) {
        std::vector<VertexData> vertices;
        if (auto mesh_vertices = interlace_mesh_attributes(mesh); mesh_vertices != std::nullopt) {
            vertices = std::move(*mesh_vertices);
        } else {
            return;
        }

        HeapSubBuffer<VertexData> vertices_sub = this->static_buffer
            .allocate<VertexData>(vertices.size())
            .value();
        HeapSubBuffer<uint32_t> indices_sub = this->static_buffer
            .allocate<uint32_t>(mesh.indices.value().size())
            .value();

        this->uploader.upload_buffer(
            BufferUpload()
                .set_buffer(this->static_buffer.buffer())
                .set_offset(vertices_sub.buffer_offset())
                .set_memory(std::span(vertices))
                .set_dst_access_mask(vk::AccessFlagBits2::eShaderStorageRead)
                .set_dst_stage_mask(vk::PipelineStageFlagBits2::eVertexShader)
        );
        this->uploader.upload_buffer(
            BufferUpload()
                .set_buffer(this->static_buffer.buffer())
                .set_offset(indices_sub.buffer_offset())
                .set_memory(std::span(mesh.indices.value()))
                .set_dst_access_mask(vk::AccessFlagBits2::eIndexRead)
                .set_dst_stage_mask(vk::PipelineStageFlagBits2::eIndexInput)
        );

        this->vertices = vertices_sub;
        this->indices = indices_sub;
    }

    void create_buffers() {
        const int layers = 1;
        const int rows = 32;
        const int columns = 32;
        const float spacing = 2.0f;

        for (int layer = 0; layer < layers; layer++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    auto frow = static_cast<float>(row);
                    auto fcolumn = static_cast<float>(column);
                    auto flayer = static_cast<float>(layer);
                    float x = (frow - static_cast<float>(rows) * 0.5f + 0.5f) * spacing;
                    float z = (fcolumn - static_cast<float>(columns) * 0.5f + 0.5f) * spacing;
                    float y = flayer * spacing + 1.0f;
                    this->instances.emplace_back(1, glm::vec3(x, y, z));
                }
            }
        }

        this->static_buffer = HeapBuffer(
            this->device,
            this->allocator,
            FRAMES_IN_FLIGHT,
            256,
            vk::BufferCreateInfo()
                .setSharingMode(vk::SharingMode::eExclusive)
                .setSize(vk::DeviceSize(16 * 1024 * 1024))
                .setUsage(
                    vk::BufferUsageFlagBits::eShaderDeviceAddress
                        | vk::BufferUsageFlagBits::eTransferDst
                        | vk::BufferUsageFlagBits::eIndexBuffer
                ),
            vma::AllocationCreateInfo()
                .setUsage(vma::MemoryUsage::eGpuOnly)
        );
    }

    void destroy_buffers() {
        this->static_buffer.destroy();
    }

    Texture create_image(const vk::ImageCreateInfo& info) {
        return ::create_texture(this->device, this->allocator, info);
    }

    FrameData write_frame_data() {
        vk::Extent2D viewport_size = this->swapchain.get_extent();
        auto viewport_width = static_cast<float>(viewport_size.width);
        auto viewport_height = static_cast<float>(viewport_size.height);

        FrameSubBuffer view_data = this->frame_arena
            .add(ViewData{
                .world_to_clip = this->camera.world_to_clip(viewport_width, viewport_height),
                .position = this->camera.transform.translation,
            })
            .value();

        FrameSubBuffer light_data = this->frame_arena
            .add(LightData {
                .ambient_color = this->ambient_light.color,
                .ambient_illuminance = this->ambient_light.illuminance,
                .point_count = static_cast<uint32_t>(this->point_lights.size()),
                .directional_count = static_cast<uint32_t>(this->directional_lights.size()),
            })
            .value();

        FrameSubBuffer dir_lights = this->frame_arena.add(this->directional_lights).value();
        FrameSubBuffer point_lights = this->frame_arena.add(this->point_lights).value();
        FrameSubBuffer instances = this->frame_arena.add(this->instances).value();
        
        return {
            .view_data = view_data,
            .light_data = light_data,
            .dir_lights = dir_lights,
            .point_lights = point_lights,
            .instances = instances
        };
    }

    void record_frame_commands(
        vk::CommandBuffer command_buffer, 
        const SwapchainImage& image,
        const FrameData& data
    ) {
        DrawConstants draw_constants = {
            .view_data = data.view_data.address(),
            .light_data = data.light_data.address(),
            .point_lights = data.point_lights.address(),
            .directional_lights = data.dir_lights.address(),
            .instances = data.instances.address(),
            .vertices = this->vertices.address(),
            .materials = this->material_manager.buffer_address(this->frame_index)
        };
        command_buffer.pushConstants<DrawConstants>(
            this->pipeline_layout,
            vk::ShaderStageFlagBits::eVertex 
                | vk::ShaderStageFlagBits::eFragment,
            0, {draw_constants}
        );
        auto subresource_range = vk::ImageSubresourceRange()
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1);

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
        this->record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(image.image)
                .setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
                .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                .setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
                .setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
                .setSubresourceRange(subresource_range)
        );
        this->record_draw_commands(command_buffer, image);
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

    void record_draw_commands(vk::CommandBuffer command_buffer, const SwapchainImage& image) {
        glm::vec3 ambient_color = this->ambient_light.color * this->ambient_light.illuminance; 
        auto color_attachment = vk::RenderingAttachmentInfo()
            .setImageView(image.view)
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setClearValue(vk::ClearColorValue(
                ambient_color.r, 
                ambient_color.g, 
                ambient_color.b, 
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
            .setRenderArea(this->swapchain.get_full_area());

        command_buffer.beginRendering(rendering);
        command_buffer.setViewport(0, this->swapchain.get_full_viewport());
        command_buffer.setScissor(0, this->swapchain.get_full_area());
        command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, this->pipeline);
        command_buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics,
            this->pipeline_layout, 0,
            {this->texture_manager.descriptor_set()}, {}
        );
        command_buffer.bindIndexBuffer(
            this->static_buffer.buffer(), 
            this->indices.buffer_offset(), 
            vk::IndexType::eUint32
        );
        command_buffer.drawIndexed(
            this->indices.length(), 
            static_cast<uint32_t>(this->instances.size()), 
            0, 0, 0
        );
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
            this->queue.submit2(
                vk::SubmitInfo2()
                    .setWaitSemaphoreInfos(wait_info)
                    .setSignalSemaphoreInfos(signal_info)
                    .setCommandBufferInfos(command_buffer_info),
                this->fences[this->frame_index]
            ),
            "Failed to submit command buffer"
        );
    }

    vk::Instance instance;
    vk::PhysicalDevice physical_device;
    vk::Device device;
    vk::Queue queue;
    uint32_t queue_family_index = 0;
    vma::Allocator allocator;
    Swapchain swapchain;
    SwapchainConfigureInfo swapchain_info;
    bool should_reconfigure_swapchain = false;

    std::vector<Texture> depth_textures;

    std::vector<vk::Fence> fences;              // One per frame in flight
    std::vector<vk::Semaphore> image_available; // One per frame in flight

    uint32_t frame_index = 0;
    uint64_t frame_counter = 0;

    vk::CommandPool command_pool;
    std::vector<vk::CommandBuffer> command_buffers; // One per frame in flight

    vk::Pipeline pipeline;
    vk::PipelineLayout pipeline_layout;

    /*
    Buffer static_buffer;
    vk::DeviceAddress current_buffer_offset;

    vk::DeviceAddress vertices_offset;
    vk::DeviceAddress vertices_size;
    uint32_t vertex_count;
    vk::DeviceAddress indices_offset;
    vk::DeviceAddress indices_size;
    uint32_t index_count;
    */
    HeapSubBuffer<VertexData> vertices;
    HeapSubBuffer<uint32_t> indices;
    HeapBuffer static_buffer;
    
    Uploader uploader;
    FrameArenaBuffer frame_arena;
    TextureManager texture_manager;
    MaterialManager material_manager;
    MaterialId material;

    Camera camera;
    AmbientLightData ambient_light = {
        .color = glm::vec3(1.0f, 1.0f, 1.0f),
        .illuminance = 0.5f
    };
    std::vector<PointLightData> point_lights = {
        PointLightData { 
            .position = glm::vec3(0.0f, -1.0f, 0.0f),
            .radius = 100.0f,
            .color = glm::vec3(1.0f, 1.0f, 1.0f),
            .intensity = 600.0f
        }
    };
    std::vector<DirectionalLightData> directional_lights = {
        DirectionalLightData { 
            .direction = glm::vec3(0.3f, 1.5f, 0.6f),
            .color = glm::vec3(1.0f, 1.0f, 1.0f),
            .illuminance = 2.0f
        }
    };
    std::vector<InstanceData> instances;
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

void Renderer::use_material(MaterialId id) {
    return this->inner->use_material(id);
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

void Renderer::add_mesh(const Mesh& mesh) {
    return this->inner->add_mesh(mesh);
}

void Renderer::set_camera(const Camera& camera) {
    this->inner->set_camera(camera);
}

void Renderer::draw_frame() {
    this->inner->draw_frame();
}