#define VMA_IMPLEMENTATION
#include "renderer.h"
#include "swapchain.h"
#include "vkerror.h"
#include "stb_image.h"
#include "target.h"
#include "camera.h"
#include "image.h"
#include "mesh.h"
#include "texture.h"
#include "buffer.h"
#include "texturemanager.h"
#include "materialmanager.h"
#include "time.h"
#include <span>
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

static constexpr uint32_t MAX_TEXTURES = 256;
static constexpr uint32_t MAX_SAMPLERS = 128;
static constexpr uint32_t MAX_MATERIALS = 256;
static constexpr size_t MAX_POINT_LIGHTS = 256;
static constexpr size_t MAX_DIRECTIONAL_LIGHTS = 128;

struct AmbientLight {
    glm::vec3 color;
    float illuminance;
};

struct PointLight {
    alignas(16) glm::vec3 position;
    float radius;
    alignas(16) glm::vec3 color;
    float intensity;
};

struct DirectionalLight {
    alignas(16) glm::vec3 direction;
    alignas(16) glm::vec3 color;
    float illuminance;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;
    glm::vec2 tex_coords;

    static void get_attributes(
        std::vector<vk::VertexInputAttributeDescription>& attributes,
        uint32_t binding,
        uint32_t first_location
    ) {
        attributes.push_back(
            vk::VertexInputAttributeDescription()
                .setLocation(first_location)
                .setBinding(binding)
                .setFormat(vk::Format::eR32G32B32Sfloat)
                .setOffset(0)
        );
        attributes.push_back(
            vk::VertexInputAttributeDescription()
                .setLocation(first_location + 1)
                .setBinding(binding)
                .setFormat(vk::Format::eR32G32B32Sfloat)
                .setOffset(3 * sizeof(float))
        );
        attributes.push_back(
            vk::VertexInputAttributeDescription()
                .setLocation(first_location + 2)
                .setBinding(binding)
                .setFormat(vk::Format::eR32G32B32A32Sfloat)
                .setOffset(6 * sizeof(float))
        );
        attributes.push_back(
            vk::VertexInputAttributeDescription()
                .setLocation(first_location + 3)
                .setBinding(binding)
                .setFormat(vk::Format::eR32G32Sfloat)
                .setOffset(10 * sizeof(float))
        );
    }
};

static std::optional<std::vector<Vertex>> interlace_mesh_attributes(const Mesh& mesh) {
    if (mesh.positions.size() != mesh.normals.size()
        || mesh.positions.size() != mesh.tex_coords.size()
        || mesh.positions.size() != mesh.tangents.size())
    {
        spdlog::error("Failed to load mesh, the length of the vertex attribute arrays must the same for all attributes");
        return std::nullopt;
    }

    std::vector<Vertex> vertices;
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

struct Instance {
    glm::mat4x3 local_to_world;
    uint32_t material_index;

    Instance(
        uint32_t material_index,
        glm::vec3 translation, 
        glm::quat rotation = glm::identity<glm::quat>(), 
        float scale = 1.0f
    ) {
        this->material_index = material_index;
        this->local_to_world = glm::mat4x3(scale * glm::mat3_cast(rotation));
        this->local_to_world[3] = translation;
    }

    static void get_attributes(
        std::vector<vk::VertexInputAttributeDescription>& attributes,
        uint32_t binding,
        uint32_t first_location
    ) {
        for (uint32_t i = 0; i < 4; i++) {
            attributes.push_back(
                vk::VertexInputAttributeDescription()
                    .setLocation(first_location + i)
                    .setBinding(binding)
                    .setFormat(vk::Format::eR32G32B32Sfloat)
                    .setOffset(3 * sizeof(float) * static_cast<size_t>(i))
            );
        }
        attributes.push_back(
            vk::VertexInputAttributeDescription()
                .setLocation(first_location + 4)
                .setBinding(binding)
                .setFormat(vk::Format::eR32Uint)
                .setOffset(12 * sizeof(float))
        );
    }
};

struct DrawConstants {
    vk::DeviceAddress instances;
    vk::DeviceAddress vertices;
    vk::DeviceAddress materials;
};

struct ViewUniforms {
    glm::mat4 world_to_clip;
};

struct LightUniforms {
    alignas(16) glm::vec3 view_position;
    alignas(16) glm::vec3 ambient_color;
    float ambient_illuminance;
    uint32_t point_count;
    uint32_t directional_count;
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
        this->texture_manager = TextureManager(
            this->device,
            this->queue,
            this->allocator,
            this->command_pool,
            FRAMES_IN_FLIGHT,
            MAX_TEXTURES
        );
        this->material_manager = MaterialManager(
            this->device,
            this->allocator,
            FRAMES_IN_FLIGHT,
            MAX_MATERIALS
        );
        this->allocate_command_buffers(); 
        this->create_buffers();
        this->create_descriptor_pool();
        this->create_descriptor_sets();
        this->create_pipeline(read_spirv_file("bin/shaders/shader.spv"));
    }

    void destroy() {
        vk_expect(this->device.waitIdle(), "Wait for device failed");
        this->device.destroyPipeline(this->pipeline);
        this->device.destroyPipelineLayout(this->pipeline_layout);
        this->device.destroyDescriptorSetLayout(this->frame_set_layout);
        //this->device.destroyDescriptorSetLayout(this->image_set_layout);
        this->device.destroyDescriptorPool(this->descriptor_pool);
        this->device.freeCommandBuffers(this->command_pool, this->command_buffers);
        this->device.destroyCommandPool(this->command_pool);
        this->destroy_sync_primitives();
        this->destroy_depth_textures();
        this->destroy_buffers();
        this->texture_manager.destroy();
        this->material_manager.destroy();
        //this->destroy_images();
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
        return this->texture_manager.add(
            this->queue,
            this->command_pool,
            image
        );
    }

    void set_texture(TextureId id, const Image& image) {
        this->texture_manager.set(
            id,
            this->queue,
            this->command_pool,
            this->frame_counter,
            image
        );
    }

    void free_texture(TextureId id) {
        this->texture_manager.request_free(id, this->frame_counter);
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

        this->texture_manager.destroy_pending(this->frame_counter);
        this->material_manager.flag_dirty_materials(this->texture_manager);
        this->material_manager.update_dirty(this->texture_manager, this->frame_index);
        this->texture_manager.clear_updated();

        this->update_lights();
        this->set_mapped_uniforms();

        vk::CommandBuffer command_buffer = this->begin_frame_commands();
        this->record_frame_commands(command_buffer, image);
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

        /*
        std::array bindings = {
            
            vk::VertexInputBindingDescription()
                .setBinding(0)
                .setStride(sizeof(Vertex))
                .setInputRate(vk::VertexInputRate::eVertex),
            vk::VertexInputBindingDescription()
                .setBinding(0)
                .setStride(sizeof(Instance))
                .setInputRate(vk::VertexInputRate::eInstance)
        };

        std::vector<vk::VertexInputAttributeDescription> attributes;
        //Vertex::get_attributes(attributes, 0, 0);
        Instance::get_attributes(attributes, 0, 0);*/

        auto vertex_state = vk::PipelineVertexInputStateCreateInfo();
            //.setVertexBindingDescriptions(bindings)
            //.setVertexAttributeDescriptions(attributes);

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
            this->frame_set_layout,
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
        std::vector<Vertex> vertices;
        if (auto mesh_vertices = interlace_mesh_attributes(mesh); mesh_vertices != std::nullopt) {
            vertices = std::move(*mesh_vertices);
        } else {
            return;
        }

        vk::DeviceSize vertices_offset = this->current_buffer_offset;
        vk::DeviceSize vertices_size = vertices.size() * sizeof(Vertex);
        this->current_buffer_offset += vertices_size;

        vk::DeviceSize indices_offset = vertices_size;
        vk::DeviceSize indices_count = mesh.indices.has_value() ? 
            mesh.indices.value().size() : 0;
        vk::DeviceSize indices_size = indices_count * sizeof(uint32_t);
        this->current_buffer_offset += indices_size;

        vk::DeviceSize staging_size = vertices_size + indices_size;
        Buffer staging = create_staging_buffer(
            this->device,
            this->allocator,
            vk::BufferUsageFlagBits::eTransferSrc,
            staging_size
        );
        memcpy(staging.mapped, vertices.data(), vertices_size);
        if (mesh.indices.has_value()) {
            memcpy(
                reinterpret_cast<uint8_t*>(staging.mapped) + indices_offset, 
                mesh.indices.value().data(), 
                indices_size
            );
        }

        vk::CommandBuffer command_buffer = this->begin_one_shot_commands();
        command_buffer.copyBuffer(
            staging,
            this->static_buffer,
            vk::BufferCopy().setSize(staging_size)
        );
        this->submit_one_shot_commands_sync(command_buffer);
        staging.destroy(this->allocator);

        this->vertices_offset = vertices_offset;
        this->vertices_size = vertices_size;
        this->vertex_count = uint32_t(vertices.size());
        this->indices_offset = indices_offset;
        this->indices_size = indices_size;
        this->index_count = indices_count;
    }

    void create_buffers() {
        const int layers = 1;
        const int rows = 32;
        const int columns = 32;
        const float spacing = 2.0f;

        std::vector<Instance> instances;

        for (int layer = 0; layer < layers; layer++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    auto frow = static_cast<float>(row);
                    auto fcolumn = static_cast<float>(column);
                    auto flayer = static_cast<float>(layer);
                    float x = (frow - static_cast<float>(rows) * 0.5f + 0.5f) * spacing;
                    float z = (fcolumn - static_cast<float>(columns) * 0.5f + 0.5f) * spacing;
                    float y = flayer * spacing + 1.0f;
                    instances.emplace_back(1, glm::vec3(x, y, z));
                }
            }
        }

        Buffer static_buffer = create_gpu_buffer(
            this->device,
            this->allocator,
            vk::BufferUsageFlagBits::eIndexBuffer
                | vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst,
            vk::DeviceSize(16 * 1024 * 1024)
        );

        Buffer instance_buffer = create_mapped_buffer(
            this->device,
            this->allocator,
            vk::BufferUsageFlagBits::eStorageBuffer 
                | vk::BufferUsageFlagBits::eTransferDst,
            instances.size() * sizeof(Instance)
        );

        memcpy(instance_buffer.mapped, instances.data(), instances.size() * sizeof(Instance));

        DynOffsetBuffer view_uniform_buffer = this->create_dyn_offset_buffer<ViewUniforms>(
            vk::BufferUsageFlagBits::eUniformBuffer, 1
        );
        DynOffsetBuffer light_uniform_buffer = this->create_dyn_offset_buffer<LightUniforms>(
            vk::BufferUsageFlagBits::eUniformBuffer, 1
        );
        DynOffsetBuffer dir_light_buffer = this->create_dyn_offset_buffer<DirectionalLight>(
            vk::BufferUsageFlagBits::eStorageBuffer, 
            MAX_DIRECTIONAL_LIGHTS
        );
        DynOffsetBuffer point_light_buffer = this->create_dyn_offset_buffer<PointLight>(
            vk::BufferUsageFlagBits::eStorageBuffer, 
            MAX_POINT_LIGHTS
        );

        this->static_buffer = static_buffer;
        this->current_buffer_offset = 0;
        this->instance_count = uint32_t(instances.size());
        this->instance_buffer = instance_buffer;
        this->view_uniform_buffer = view_uniform_buffer;
        this->light_uniform_buffer = light_uniform_buffer;
        this->dir_light_buffer = dir_light_buffer;
        this->point_light_buffer = point_light_buffer;
    }

    void destroy_buffers() {
        this->dir_light_buffer.destroy(this->allocator);
        this->point_light_buffer.destroy(this->allocator);
        this->instance_buffer.destroy(this->allocator);
        this->static_buffer.destroy(this->allocator);
        this->view_uniform_buffer.destroy(this->allocator);
        this->light_uniform_buffer.destroy(this->allocator);
    }

    Texture create_image(const vk::ImageCreateInfo& info) {
        return ::create_texture(this->device, this->allocator, info);
    }

    template<typename T>
    DynOffsetBuffer<T> create_dyn_offset_buffer(
        vk::BufferUsageFlags usage, 
        vk::DeviceSize length,
        vk::DeviceSize count = FRAMES_IN_FLIGHT     
    ) {
        return ::create_dyn_offset_buffer<T>(this->device, this->allocator, usage, length, count);
    }

    void create_descriptor_pool() {
        std::array pool_sizes = {
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eUniformBufferDynamic)
                .setDescriptorCount(2),
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eStorageBufferDynamic)
                .setDescriptorCount(3),
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eSampler)
                .setDescriptorCount(MAX_SAMPLERS),
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eSampledImage)
                .setDescriptorCount(MAX_TEXTURES),
        };
        auto [result, descriptor_pool] = this->device.createDescriptorPool(
            vk::DescriptorPoolCreateInfo()
                .setMaxSets(3)
                .setPoolSizes(pool_sizes)
        );
        vk_expect(result, "Failed to create descriptor pool");
        this->descriptor_pool = descriptor_pool;
    }

    void create_descriptor_sets() {
        this->create_frame_set();
        //this->create_image_set();
        //this->create_material_set();
    }

    void create_frame_set() {
        std::array bindings = {
            vk::DescriptorSetLayoutBinding()
                .setBinding(0)
                .setDescriptorType(vk::DescriptorType::eUniformBufferDynamic)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eVertex),
            vk::DescriptorSetLayoutBinding()
                .setBinding(1)
                .setDescriptorType(vk::DescriptorType::eUniformBufferDynamic)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eFragment),
            vk::DescriptorSetLayoutBinding()
                .setBinding(2)
                .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eFragment),
            vk::DescriptorSetLayoutBinding()
                .setBinding(3)
                .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eFragment),
        };
        auto [result1, layout] = this->device.createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo().setBindings(bindings)
        );
        vk_expect(result1, "Failed to create descriptor set layout");

        auto [result2, sets] = this->device.allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo()
                .setDescriptorPool(this->descriptor_pool)
                .setSetLayouts(layout)
        );
        vk_expect(result2, "Failed to create descriptor set layout");

        std::vector<vk::WriteDescriptorSet> writes;
        
        auto view_uniforms_info = this->view_uniform_buffer.descriptor_info();
        writes.push_back(
            vk::WriteDescriptorSet()
                .setDstSet(sets[0])
                .setDstBinding(0)
                .setDstArrayElement(0)
                .setDescriptorType(vk::DescriptorType::eUniformBufferDynamic)
                .setBufferInfo(view_uniforms_info)
        );

        auto light_uniforms_info = this->light_uniform_buffer.descriptor_info();
        writes.push_back(
            vk::WriteDescriptorSet()
                .setDstSet(sets[0])
                .setDstBinding(1)
                .setDstArrayElement(0)
                .setDescriptorType(vk::DescriptorType::eUniformBufferDynamic)
                .setBufferInfo(light_uniforms_info)
        );

        auto dir_lights_info = this->dir_light_buffer.descriptor_info();
        writes.push_back(
            vk::WriteDescriptorSet()
                .setDstSet(sets[0])
                .setDstBinding(2)
                .setDstArrayElement(0)
                .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
                .setBufferInfo(dir_lights_info)
        );

        auto point_lights_info = this->point_light_buffer.descriptor_info();
        writes.push_back(
            vk::WriteDescriptorSet()
                .setDstSet(sets[0])
                .setDstBinding(3)
                .setDstArrayElement(0)
                .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
                .setBufferInfo(point_lights_info)
        );
        
        this->device.updateDescriptorSets(writes, {});

        this->frame_set = sets[0];
        this->frame_set_layout = layout;
    }

    void set_mapped_uniforms() {
        vk::Extent2D viewport_size = this->swapchain.get_extent();
        auto viewport_width = static_cast<float>(viewport_size.width);
        auto viewport_height = static_cast<float>(viewport_size.height);

        this->view_uniform_buffer.write_one(this->frame_index, ViewUniforms{
            .world_to_clip = this->camera.world_to_clip(viewport_width, viewport_height),
        });
        this->light_uniform_buffer.write_one(this->frame_index, LightUniforms {
            .view_position = this->camera.transform.translation,
            .ambient_color = this->ambient_light.color,
            .ambient_illuminance = this->ambient_light.illuminance,
            .point_count = static_cast<uint32_t>(this->point_lights.size()),
            .directional_count = static_cast<uint32_t>(this->directional_lights.size()),
        });
    }

    void update_lights() {
        this->dir_light_buffer.write_one(this->frame_index, this->directional_lights);
        this->point_light_buffer.write_one(this->frame_index, this->point_lights);
    }

    void record_frame_commands(vk::CommandBuffer command_buffer, const SwapchainImage& image) {
        DrawConstants draw_constants = {
            .instances = this->instance_buffer.address,
            .vertices = this->static_buffer.address + this->vertices_offset,
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
            {
                this->frame_set, 
                this->texture_manager.descriptor_set(), 
            },
            {
                this->view_uniform_buffer.offset(this->frame_index),
                this->light_uniform_buffer.offset(this->frame_index),
                this->dir_light_buffer.offset(this->frame_index),
                this->point_light_buffer.offset(this->frame_index),
            }
        );
        command_buffer.bindIndexBuffer(this->static_buffer, this->indices_offset, vk::IndexType::eUint32);
        command_buffer.drawIndexed(
            this->index_count, 
            this->instance_count, 
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

    vk::CommandBuffer begin_one_shot_commands() {
        auto [result, command_buffer] = this->device.allocateCommandBuffers(
            vk::CommandBufferAllocateInfo()
                .setCommandPool(this->command_pool)
                .setCommandBufferCount(1)
                .setLevel(vk::CommandBufferLevel::ePrimary)
        );
        vk_expect(result, "Failed to create one-shot command buffer");
        vk_expect(
            command_buffer[0].begin(
                vk::CommandBufferBeginInfo()
                    .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
            ),
            "Failed to begin one-shot command buffer"
        );
        return command_buffer[0];
    }

    void submit_one_shot_commands_sync(vk::CommandBuffer command_buffer) {
        vk_expect(command_buffer.end(), "Failed to end one-shot command buffer");

        auto [result, fence] = this->device.createFence(vk::FenceCreateInfo());
        vk_expect(result, "Failed to create fence");

        auto command_buffer_info = vk::CommandBufferSubmitInfo().setCommandBuffer(command_buffer);

        vk_expect(
            this->queue
                .submit2(vk::SubmitInfo2().setCommandBufferInfos(command_buffer_info), fence),
            "Failed to submit one-shot command buffer"
        );
        vk_expect(
            this->device.waitForFences(fence, true, std::numeric_limits<uint64_t>::max()),
            "Wait for fence failed"
        );

        this->device.destroyFence(fence);
        this->device.freeCommandBuffers(this->command_pool, command_buffer);
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

    vk::DescriptorPool descriptor_pool;
    vk::DescriptorSetLayout frame_set_layout;
    vk::DescriptorSet frame_set;

    DynOffsetBuffer<ViewUniforms> view_uniform_buffer;
    DynOffsetBuffer<LightUniforms> light_uniform_buffer;
    DynOffsetBuffer<DirectionalLight> dir_light_buffer;
    DynOffsetBuffer<PointLight> point_light_buffer;

    Buffer instance_buffer;
    uint32_t instance_count;

    Buffer static_buffer;
    vk::DeviceAddress current_buffer_offset;

    vk::DeviceAddress vertices_offset;
    vk::DeviceAddress vertices_size;
    uint32_t vertex_count;
    vk::DeviceAddress indices_offset;
    vk::DeviceAddress indices_size;
    uint32_t index_count;
    
    TextureManager texture_manager;
    MaterialManager material_manager;
    MaterialId material;

    Camera camera;
    AmbientLight ambient_light = {
        .color = glm::vec3(1.0f, 1.0f, 1.0f),
        .illuminance = 0.5f
    };
    std::vector<PointLight> point_lights = {
        PointLight { 
            .position = glm::vec3(0.0f, -1.0f, 0.0f),
            .radius = 100.0f,
            .color = glm::vec3(1.0f, 1.0f, 1.0f),
            .intensity = 600.0f
        }
    };
    std::vector<DirectionalLight> directional_lights = {
        DirectionalLight { 
            .direction = glm::vec3(0.3f, 1.5f, 0.6f),
            .color = glm::vec3(1.0f, 1.0f, 1.0f),
            .illuminance = 4.0f
        }
    };
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