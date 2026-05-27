#define VMA_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "renderer.h"
#include "swapchain.h"
#include "vkerror.h"
#include "target.h"
#include "camera.h"
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
#include <stb_image.h>
#include <vulkan/vulkan.hpp>

static constexpr size_t MAX_POINT_LIGHTS = 256;
static constexpr size_t MAX_DIRECTIONAL_LIGHTS = 128;

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
                .setFormat(vk::Format::eR32G32Sfloat)
                .setOffset(6 * sizeof(float))
        );
    }
};

struct Instance {
    glm::mat4x3 local_to_world;

    Instance(
        glm::vec3 translation, 
        glm::quat rotation = glm::identity<glm::quat>(), 
        float scale = 1.0f
    ) {
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
    }
};

struct Uniforms {
    glm::mat4 world_to_clip;
    alignas(16) glm::vec3 view_position;
    alignas(16) glm::vec3 ambient_light_color;
    uint32_t point_lights;
    uint32_t directional_lights;
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
        this->allocate_command_buffers();
        this->load_images();
        this->create_buffers();
        this->create_descriptor_pool();
        this->create_descriptor_set();
        this->create_pipeline(read_spirv_file("bin/shaders/shader.spv"));
    }

    void destroy() {
        vk_expect(this->device.waitIdle(), "Wait for device failed");
        this->device.destroyPipeline(this->pipeline);
        this->device.destroyPipelineLayout(this->pipeline_layout);
        this->device.destroyDescriptorSetLayout(this->descriptor_set_layout);
        this->device.destroyDescriptorPool(this->descriptor_pool);
        this->device.freeCommandBuffers(this->command_pool, this->command_buffers);
        this->device.destroyCommandPool(this->command_pool);
        this->destroy_sync_primitives();
        this->destroy_depth_images();
        this->destroy_buffers();
        this->destroy_images();
        this->allocator.destroy();
        this->swapchain.destroy(this->device);
        this->device.destroy();
        this->instance.destroy();
    }

    void set_camera(const Camera& camera) {
        this->world_to_clip_matrix = camera.world_to_clip();
        this->view_position = camera.translation;
    }

    void resize() {
        this->should_reconfigure_swapchain = true;
    }

    void draw_frame() {
        auto [result1, image] = this->swapchain.acquire_image(
            this->device, 
            this->fences[this->frame_index], 
            this->image_availible[this->frame_index]
        );

        if (result1 == vk::Result::eErrorOutOfDateKHR) {
            this->reconfigure_render_targets();
            return;
        } else if (result1 != vk::Result::eSuboptimalKHR) {
            vk_expect(result1, "Critical swapchain image acquisition failure");
        }

        this->update_lights();
        this->set_mapped_uniforms(Uniforms{
            .world_to_clip = this->world_to_clip_matrix,
            .view_position = this->view_position,
            .ambient_light_color = this->ambient_light_color,
            .point_lights = static_cast<uint32_t>(this->point_lights.size()),
            .directional_lights = static_cast<uint32_t>(this->directional_lights.size()),
        });

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

        this->frame_index = (this->frame_index + 1) % FRAMES_IN_FLIGHT;
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
        vk_expect(result, "Device extension properties not availible");

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

        this->image_availible.resize(FRAMES_IN_FLIGHT);
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            auto [result, semaphore] = this->device.createSemaphore(vk::SemaphoreCreateInfo());
            vk_expect(result, "Failed to create image availible fence");
            this->image_availible[i] = semaphore;
        }
    }

    void destroy_sync_primitives() {
        for (auto fence : this->fences) {
            this->device.destroyFence(fence);
        }
        for (auto image_availible : this->image_availible) {
            this->device.destroySemaphore(image_availible);
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

        std::array bindings = {
            vk::VertexInputBindingDescription()
                .setBinding(0)
                .setStride(sizeof(Vertex))
                .setInputRate(vk::VertexInputRate::eVertex),
            vk::VertexInputBindingDescription()
                .setBinding(1)
                .setStride(sizeof(Instance))
                .setInputRate(vk::VertexInputRate::eInstance)
        };

        std::vector<vk::VertexInputAttributeDescription> attributes;
        Vertex::get_attributes(attributes, 0, 0);
        Instance::get_attributes(attributes, 1, 3);

        auto vertex_state = vk::PipelineVertexInputStateCreateInfo()
            .setVertexBindingDescriptions(bindings)
            .setVertexAttributeDescriptions(attributes);

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

        auto [result2, pipeline_layout] = this->device.createPipelineLayout(
            vk::PipelineLayoutCreateInfo().setSetLayouts(this->descriptor_set_layout)
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
        if (this->depth_images.size() != FRAMES_IN_FLIGHT) {
            this->depth_images.resize(FRAMES_IN_FLIGHT);
            this->depth_views.resize(FRAMES_IN_FLIGHT);
            this->depth_image_allocs.resize(FRAMES_IN_FLIGHT);
        }

        vk::Format format = vk::Format::eD32Sfloat;

        auto image_info = vk::ImageCreateInfo()
            .setImageType(vk::ImageType::e2D)
            .setExtent(vk::Extent3D(this->swapchain.get_extent(), 1))
            .setFormat(format)
            .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
            .setMipLevels(1)
            .setArrayLayers(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setQueueFamilyIndices(this->queue_family_index)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eOptimal);
        auto alloc_info = vma::AllocationCreateInfo().setUsage(vma::MemoryUsage::eGpuOnly);

        vma::AllocationInfo alloc_stats;
        auto [result1, pair] = this->allocator.createImage(image_info, alloc_info, alloc_stats);
        vk_expect(result1, "Failed to create depth texture");

        auto [alloc, image] = pair;
        auto [result2, view] = device.createImageView(
            vk::ImageViewCreateInfo()
                .setImage(image)
                .setFormat(format)
                .setComponents(vk::ComponentMapping())
                .setViewType(vk::ImageViewType::e2D)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eDepth)
                        .setBaseMipLevel(0)
                        .setLevelCount(1)
                        .setBaseArrayLayer(0)
                        .setLayerCount(1)
                )
        );
        vk_expect(result2, "Failed to create depth image view");

        if (this->depth_images[frame_index] != vk::Image()) {
            this->device.destroyImageView(this->depth_views[frame_index]);
            this->allocator.destroyImage(
                this->depth_images[frame_index],
                this->depth_image_allocs[frame_index]
            );
        }

        this->depth_images[frame_index] = image;
        this->depth_views[frame_index] = view;
        this->depth_image_allocs[frame_index] = alloc;
    }

    void destroy_depth_images() {
        for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
            if (this->depth_images[i] != vk::Image()) {
                this->device.destroyImageView(this->depth_views[i]);
                this->allocator.destroyImage(
                    this->depth_images[i], 
                    this->depth_image_allocs[i]
                );
            }
        }
    }

    void load_images() {
        int width, height, components;
        uint8_t* image_bytes = stbi_load("../images/rocks.jpg", &width, &height, &components, 4);
        if (image_bytes == nullptr) {
            spdlog::critical("Failed to load image");
            std::exit(1);
        }

        vk::Extent3D extent = vk::Extent3D(width, height, 1);
        vk::Format format = vk::Format::eR8G8B8A8Srgb;
        auto image_info = vk::ImageCreateInfo()
            .setImageType(vk::ImageType::e2D)
            .setExtent(extent)
            .setFormat(format)
            .setMipLevels(1)
            .setArrayLayers(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setQueueFamilyIndices(this->queue_family_index)
            .setUsage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eOptimal);

        auto alloc_info = vma::AllocationCreateInfo().setUsage(vma::MemoryUsage::eGpuOnly);

        vma::AllocationInfo alloc_stats;
        auto [result1, pair]= this->allocator.createImage(image_info, alloc_info, alloc_stats);
        vk_expect(result1, "Failed to create image");
        auto [alloc, image] = pair;

        auto [staging, staging_alloc] = this->create_buffer_staging(
            image_bytes,
            static_cast<size_t>(width) * static_cast<size_t>(height) * 4
        );
        stbi_image_free(image_bytes);

        auto subresource_range = vk::ImageSubresourceRange()
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1);

        vk::CommandBuffer command_buffer = this->begin_one_shot_commands();

        this->record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(image)
                .setSrcStageMask(vk::PipelineStageFlagBits2::eNone)
                .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                .setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
                .setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
                .setOldLayout(vk::ImageLayout::eUndefined)
                .setNewLayout(vk::ImageLayout::eTransferDstOptimal)
                .setSubresourceRange(subresource_range)
        );
        
        command_buffer.copyBufferToImage(
            staging, 
            image, 
            vk::ImageLayout::eTransferDstOptimal,
            {
                vk::BufferImageCopy()
                    .setImageExtent(extent)
                    .setImageSubresource(
                        vk::ImageSubresourceLayers()
                            .setAspectMask(vk::ImageAspectFlagBits::eColor)
                            .setBaseArrayLayer(0)
                            .setLayerCount(1)
                            .setMipLevel(0)
                    )
            }
        );

        this->record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(image)
                .setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer)
                .setSrcAccessMask(vk::AccessFlagBits2::eNone)
                .setDstStageMask(vk::PipelineStageFlagBits2::eFragmentShader)
                .setDstAccessMask(vk::AccessFlagBits2::eShaderSampledRead)
                .setOldLayout(vk::ImageLayout::eTransferDstOptimal)
                .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
                .setSubresourceRange(subresource_range)
        );

        this->submit_one_shot_commands_sync(command_buffer);
        
        this->allocator.destroyBuffer(staging, staging_alloc);

        auto [result2, view] = device.createImageView(
            vk::ImageViewCreateInfo()
                .setImage(image)
                .setFormat(format)
                .setComponents(vk::ComponentMapping())
                .setViewType(vk::ImageViewType::e2D)
                .setSubresourceRange(subresource_range)
        );
        vk_expect(result2, "Failed to create image view");

        auto [result3, sampler] = device.createSampler(
            vk::SamplerCreateInfo()
                .setAddressModeU(vk::SamplerAddressMode::eRepeat)
                .setAddressModeV(vk::SamplerAddressMode::eRepeat)
                .setAddressModeW(vk::SamplerAddressMode::eRepeat)
                .setAnisotropyEnable(false)
                .setMinFilter(vk::Filter::eNearest)
                .setMagFilter(vk::Filter::eLinear)
                .setMipmapMode(vk::SamplerMipmapMode::eLinear)
                .setMinLod(0.0f)
                .setMaxLod(64.0f)
        );

        this->image = image;
        this->image_view = view;
        this->image_sampler = sampler;
        this->image_alloc = alloc;
    }

    void destroy_images() {
        this->device.destroySampler(this->image_sampler);
        this->device.destroyImageView(this->image_view);
        this->allocator.destroyImage(this->image, this->image_alloc);
    }

    void create_buffers() {
        const int layers = 2;
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
                    instances.emplace_back(glm::vec3(x, y, z));
                }
            }
        }

        std::array vertices = {
            // -Z face
            Vertex{{-0.5f, -0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {0.0f, 0.0f}},
            Vertex{{-0.5f,  0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {1.0f, 0.0f}},
            Vertex{{ 0.5f,  0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {1.0f, 1.0f}},
            Vertex{{ 0.5f, -0.5f, -0.5f}, { 0.0f,  0.0f, -1.0f}, {0.0f, 1.0f}},
            // +Z face
            Vertex{{-0.5f, -0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {0.0f, 0.0f}},
            Vertex{{ 0.5f, -0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {1.0f, 0.0f}},
            Vertex{{ 0.5f,  0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {1.0f, 1.0f}},
            Vertex{{-0.5f,  0.5f,  0.5f}, { 0.0f,  0.0f,  1.0f}, {0.0f, 1.0f}},
            // -X face
            Vertex{{-0.5f, -0.5f, -0.5f}, {-1.0f,  0.0f,  0.0f}, {0.0f, 0.0f}},
            Vertex{{-0.5f, -0.5f,  0.5f}, {-1.0f,  0.0f,  0.0f}, {1.0f, 0.0f}},
            Vertex{{-0.5f,  0.5f,  0.5f}, {-1.0f,  0.0f,  0.0f}, {1.0f, 1.0f}},
            Vertex{{-0.5f,  0.5f, -0.5f}, {-1.0f,  0.0f,  0.0f}, {0.0f, 1.0f}},
            // +X face
            Vertex{{ 0.5f, -0.5f, -0.5f}, { 1.0f,  0.0f,  0.0f}, {0.0f, 0.0f}},
            Vertex{{ 0.5f,  0.5f, -0.5f}, { 1.0f,  0.0f,  0.0f}, {1.0f, 0.0f}},
            Vertex{{ 0.5f,  0.5f,  0.5f}, { 1.0f,  0.0f,  0.0f}, {1.0f, 1.0f}},
            Vertex{{ 0.5f, -0.5f,  0.5f}, { 1.0f,  0.0f,  0.0f}, {0.0f, 1.0f}},
            // -Y face (top, since +Y is down)
            Vertex{{-0.5f, -0.5f, -0.5f}, { 0.0f, -1.0f,  0.0f}, {0.0f, 0.0f}},
            Vertex{{ 0.5f, -0.5f, -0.5f}, { 0.0f, -1.0f,  0.0f}, {1.0f, 0.0f}},
            Vertex{{ 0.5f, -0.5f,  0.5f}, { 0.0f, -1.0f,  0.0f}, {1.0f, 1.0f}},
            Vertex{{-0.5f, -0.5f,  0.5f}, { 0.0f, -1.0f,  0.0f}, {0.0f, 1.0f}},
            // +Y face (bottom)
            Vertex{{-0.5f,  0.5f, -0.5f}, { 0.0f,  1.0f,  0.0f}, {0.0f, 0.0f}},
            Vertex{{-0.5f,  0.5f,  0.5f}, { 0.0f,  1.0f,  0.0f}, {1.0f, 0.0f}},
            Vertex{{ 0.5f,  0.5f,  0.5f}, { 0.0f,  1.0f,  0.0f}, {1.0f, 1.0f}},
            Vertex{{ 0.5f,  0.5f, -0.5f}, { 0.0f,  1.0f,  0.0f}, {0.0f, 1.0f}},
        };

        std::array<uint16_t, 36> indices = {
            0,  1,  2,   0,  2,  3,  // -Z
            4,  5,  6,   4,  6,  7,  // +Z
            8,  9, 10,   8, 10, 11,  // -X
            12, 13, 14,  12, 14, 15,  // +X
            16, 17, 18,  16, 18, 19,  // -Y
            20, 21, 22,  20, 22, 23,  // +Y
        };

        auto [vertex_buffer, vertex_buffer_alloc] = this->create_buffer(
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            sizeof(vertices)
        );
        auto [index_buffer, index_buffer_alloc] = this->create_buffer(
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            sizeof(indices)
        );
        auto [instance_buffer, instance_buffer_alloc] = this->create_buffer(
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            instances.size() * sizeof(instances[0])
        );
        uint32_t uniform_stride = this->get_uniform_stride(sizeof(Uniforms));
        auto [uniform_buffer, uniform_info, uniform_buffer_alloc] = this->create_buffer_mapped(
            vk::BufferUsageFlagBits::eUniformBuffer, 
            uniform_stride * FRAMES_IN_FLIGHT
        );
        vk::DeviceSize dir_lights_size = MAX_DIRECTIONAL_LIGHTS * sizeof(DirectionalLight);
        uint32_t dir_lights_stride = this->get_storage_stride(dir_lights_size);
        auto [dir_light_buffer, dir_light_info, dir_light_buffer_alloc] = 
            this->create_buffer_mapped(
                vk::BufferUsageFlagBits::eStorageBuffer, 
                dir_lights_stride * FRAMES_IN_FLIGHT
            );
        vk::DeviceSize point_lights_size = MAX_POINT_LIGHTS * sizeof(PointLight);
        uint32_t point_lights_stride = this->get_storage_stride(point_lights_size);
        auto [point_light_buffer, point_light_info, point_light_buffer_alloc] = 
            this->create_buffer_mapped(
                vk::BufferUsageFlagBits::eStorageBuffer, 
                point_lights_stride * FRAMES_IN_FLIGHT
            );

        auto [vertex_staging, vertex_staging_alloc] =
            this->create_buffer_staging(vertices.data(), sizeof(vertices));
        auto [index_staging, index_staging_alloc] =
            this->create_buffer_staging(indices.data(), sizeof(indices));
        auto [instance_staging, instance_staging_alloc] = this->create_buffer_staging(
            instances.data(), instances.size() * sizeof(instances[0])
        );

        vk::CommandBuffer command_buffer = this->begin_one_shot_commands();
        command_buffer.copyBuffer(
            vertex_staging,
            vertex_buffer,
            vk::BufferCopy().setSize(sizeof(vertices))
        );
        command_buffer.copyBuffer(
            index_staging, index_buffer, vk::BufferCopy().setSize(sizeof(indices))
        );
        command_buffer.copyBuffer(
            instance_staging,
            instance_buffer,
            vk::BufferCopy().setSize(instances.size() * sizeof(instances[0]))
        );
        this->submit_one_shot_commands_sync(command_buffer);

        this->allocator.destroyBuffer(vertex_staging, vertex_staging_alloc);
        this->allocator.destroyBuffer(index_staging, index_staging_alloc);
        this->allocator.destroyBuffer(instance_staging, instance_staging_alloc);

        this->vertex_buffer = vertex_buffer;
        this->vertex_buffer_alloc = vertex_buffer_alloc;
        this->vertex_count = vertices.size();
        this->index_buffer = index_buffer;
        this->index_buffer_alloc = index_buffer_alloc;
        this->index_count = indices.size();
        this->instance_buffer = instance_buffer;
        this->instance_buffer_alloc = instance_buffer_alloc;
        this->instance_count = instances.size();
        this->uniform_buffer = uniform_buffer;
        this->uniform_buffer_alloc = uniform_buffer_alloc;
        this->uniform_buffer_mapped = uniform_info.pMappedData;
        this->uniform_stride = uniform_stride;
        this->directional_light_buffer = dir_light_buffer;
        this->directional_light_buffer_alloc = dir_light_buffer_alloc;
        this->directional_light_buffer_stride = dir_lights_stride;
        this->directional_light_buffer_mapped = dir_light_info.pMappedData;
        this->point_light_buffer = point_light_buffer;
        this->point_light_buffer_alloc = point_light_buffer_alloc;
        this->point_light_buffer_stride = point_lights_stride;
        this->point_light_buffer_mapped = point_light_info.pMappedData;
    }

    uint32_t get_uniform_stride(uint32_t uniform_size) {
        vk::DeviceSize align =
            this->physical_device.getProperties().limits.minUniformBufferOffsetAlignment;

        return static_cast<uint32_t>((uniform_size + align - 1) & ~(align - 1));
    }

    uint32_t get_storage_stride(uint32_t storage_size) {
        vk::DeviceSize align =
            this->physical_device.getProperties().limits.minStorageBufferOffsetAlignment;

        return static_cast<uint32_t>((storage_size + align - 1) & ~(align - 1));
    }

    void destroy_buffers() {
        this->allocator.destroyBuffer(this->directional_light_buffer, this->directional_light_buffer_alloc);
        this->allocator.destroyBuffer(this->point_light_buffer, this->point_light_buffer_alloc);
        this->allocator.destroyBuffer(this->instance_buffer, this->instance_buffer_alloc);
        this->allocator.destroyBuffer(this->vertex_buffer, this->vertex_buffer_alloc);
        this->allocator.destroyBuffer(this->index_buffer, this->index_buffer_alloc);
        this->allocator.destroyBuffer(this->uniform_buffer, this->uniform_buffer_alloc);
    }

    std::tuple<vk::Buffer, vma::Allocation> create_buffer(
        vk::BufferUsageFlags usage, 
        vk::DeviceSize buffer_size
    ) {
        std::vector<vk::Buffer> buffers;
        auto buffer_info = vk::BufferCreateInfo()
            .setSize(buffer_size)
            .setUsage(usage)
            .setSharingMode(vk::SharingMode::eExclusive);
        auto alloc_info = vma::AllocationCreateInfo().setUsage(vma::MemoryUsage::eGpuOnly);

        vma::AllocationInfo alloc_stats;
        auto [result, resource] =
            this->allocator.createBuffer(buffer_info, alloc_info, alloc_stats);
        vk_expect(result, "Failed to allocate buffer");

        auto [alloc, buffer] = resource;
        return {buffer, alloc};
    }

    std::tuple<vk::Buffer, vma::Allocation> create_buffer_staging(void* data, size_t size) {
        auto [buffer, alloc_info, alloc] =
            create_buffer_mapped(vk::BufferUsageFlagBits::eTransferSrc, size);
        memcpy(alloc_info.pMappedData, data, size);
        vk_expect(
            this->allocator.flushAllocation(alloc, 0, size), 
            "Failed to flush memory allocation"
        );
        return {buffer, alloc};
    }

    std::tuple<vk::Buffer, vma::AllocationInfo, vma::Allocation> create_buffer_mapped(
        vk::BufferUsageFlags usage, 
        vk::DeviceSize buffer_size
    ) {
        auto buffer_info = vk::BufferCreateInfo()
            .setSize(buffer_size)
            .setUsage(usage)
            .setSharingMode(vk::SharingMode::eExclusive);

        auto alloc_info = vma::AllocationCreateInfo()
            .setUsage(vma::MemoryUsage::eAuto)
            .setFlags(
                vma::AllocationCreateFlagBits::eHostAccessSequentialWrite 
                    | vma::AllocationCreateFlagBits::eMapped
            );

        vma::AllocationInfo alloc_stats;
        auto [result, resource] =
            this->allocator.createBuffer(buffer_info, alloc_info, alloc_stats);
        vk_expect(result, "Failed to allocate buffer");

        auto [alloc, buffer] = resource;
        return {buffer, alloc_stats, alloc};
    }

    void create_descriptor_pool() {
        std::array pool_sizes = {
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eUniformBufferDynamic)
                .setDescriptorCount(FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eStorageBufferDynamic)
                .setDescriptorCount(2 * FRAMES_IN_FLIGHT),
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(FRAMES_IN_FLIGHT),
        };
        auto [result, descriptor_pool] = this->device.createDescriptorPool(
            vk::DescriptorPoolCreateInfo()
                .setMaxSets(FRAMES_IN_FLIGHT)
                .setPoolSizes(pool_sizes)
        );
        vk_expect(result, "Failed to create descriptor pool");
        this->descriptor_pool = descriptor_pool;
    }

    void create_descriptor_set() {
        std::array bindings = {
            vk::DescriptorSetLayoutBinding()
                .setBinding(0)
                .setDescriptorType(vk::DescriptorType::eUniformBufferDynamic)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
            vk::DescriptorSetLayoutBinding()
                .setBinding(1)
                .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eFragment),
            vk::DescriptorSetLayoutBinding()
                .setBinding(2)
                .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eFragment),
            vk::DescriptorSetLayoutBinding()
                .setBinding(3)
                .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eFragment),
        };
        auto [result1, descriptor_set_layout] = this->device.createDescriptorSetLayout(
            vk::DescriptorSetLayoutCreateInfo().setBindings(bindings)
        );
        vk_expect(result1, "Failed to create descriptor set layout");

        auto [result2, descriptor_sets] = this->device.allocateDescriptorSets(
            vk::DescriptorSetAllocateInfo()
                .setDescriptorPool(this->descriptor_pool)
                .setSetLayouts(descriptor_set_layout)
        );
        vk_expect(result2, "Failed to create descriptor set layout");

        auto buffer_info = vk::DescriptorBufferInfo()
            .setBuffer(this->uniform_buffer)
            .setOffset(0)
            .setRange(sizeof(Uniforms));
        auto write_uniforms = vk::WriteDescriptorSet()
            .setDstSet(descriptor_sets[0])
            .setDstBinding(0)
            .setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eUniformBufferDynamic)
            .setBufferInfo(buffer_info);

        auto dir_light_buffer_info = vk::DescriptorBufferInfo()
            .setBuffer(this->directional_light_buffer)
            .setOffset(0)
            .setRange(sizeof(DirectionalLight));
        auto write_dir_lights = vk::WriteDescriptorSet()
            .setDstSet(descriptor_sets[0])
            .setDstBinding(1)
            .setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
            .setBufferInfo(dir_light_buffer_info);

        auto point_light_buffer_info = vk::DescriptorBufferInfo()
            .setBuffer(this->point_light_buffer)
            .setOffset(0)
            .setRange(sizeof(PointLight));
        auto write_point_lights = vk::WriteDescriptorSet()
            .setDstSet(descriptor_sets[0])
            .setDstBinding(2)
            .setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eStorageBufferDynamic)
            .setBufferInfo(point_light_buffer_info);

        auto image_info = vk::DescriptorImageInfo()
            .setImageView(this->image_view)
            .setSampler(this->image_sampler)
            .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
        auto write_image = vk::WriteDescriptorSet()
            .setDstSet(descriptor_sets[0])
            .setDstBinding(3)
            .setDstArrayElement(0)
            .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
            .setImageInfo(image_info);

        this->device.updateDescriptorSets({
            write_uniforms, 
            write_dir_lights,
            write_point_lights,
            write_image,
        }, {});

        this->descriptor_set_layout = descriptor_set_layout;
        this->descriptor_set = descriptor_sets[0];
    }

    void prepare_for_drawing() {
        this->set_mapped_uniforms({this->world_to_clip_matrix});
    }

    void set_mapped_uniforms(const Uniforms& uniforms) {
        auto mapped = reinterpret_cast<Uniforms*>(
            reinterpret_cast<uint8_t*>(this->uniform_buffer_mapped) 
                + static_cast<size_t>(this->uniform_stride) 
                    * static_cast<size_t>(this->frame_index)
        );
        *mapped = uniforms;
    }

    void update_lights() {
        auto mapped_dir_lights = reinterpret_cast<uint8_t*>(this->directional_light_buffer_mapped)
            + static_cast<size_t>(this->directional_light_buffer_stride) 
                * static_cast<size_t>(this->frame_index);
        auto mapped_point_lights = reinterpret_cast<uint8_t*>(this->point_light_buffer_mapped)
            + static_cast<size_t>(this->point_light_buffer_stride) 
                * static_cast<size_t>(this->frame_index);
        std::memcpy(
            mapped_dir_lights, 
            this->directional_lights.data(), 
            this->directional_lights.size() * sizeof(DirectionalLight)
        );
        std::memcpy(
            mapped_point_lights, 
            this->point_lights.data(), 
            this->point_lights.size() * sizeof(PointLight)
        );
    }

    void record_frame_commands(vk::CommandBuffer command_buffer, const SwapchainImage& image) {
        auto subresource_range = vk::ImageSubresourceRange()
            .setAspectMask(vk::ImageAspectFlagBits::eColor)
            .setBaseMipLevel(0)
            .setLevelCount(1)
            .setBaseArrayLayer(0)
            .setLayerCount(1);

        this->record_image_barrier(
            command_buffer,
            vk::ImageMemoryBarrier2()
                .setImage(this->depth_images[this->frame_index])
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
        auto color_attachment = vk::RenderingAttachmentInfo()
            .setImageView(image.view)
            .setImageLayout(vk::ImageLayout::eColorAttachmentOptimal)
            .setClearValue(vk::ClearColorValue(
                this->ambient_light_color.r, 
                this->ambient_light_color.g, 
                this->ambient_light_color.b, 
                1.0f
            ))
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore);

        auto depth_attachment = vk::RenderingAttachmentInfo()
            .setImageView(this->depth_views[this->frame_index])
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
            this->descriptor_set,
            {
                this->uniform_stride * this->frame_index,
                this->directional_light_buffer_stride * this->frame_index,
                this->point_light_buffer_stride * this->frame_index
            }
        );
        command_buffer.bindIndexBuffer(this->index_buffer, 0, vk::IndexType::eUint16);
        command_buffer.bindVertexBuffers(0, this->vertex_buffer, {0});
        command_buffer.bindVertexBuffers(1, this->instance_buffer, {0});
        command_buffer.drawIndexed(
            static_cast<uint32_t>(this->index_count), 
            static_cast<uint32_t>(this->instance_count), 
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
            .setSemaphore(image.availible)
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

    std::vector<vk::Image> depth_images; // One per frame in flight
    std::vector<vk::ImageView> depth_views;
    std::vector<VmaAllocation> depth_image_allocs;

    std::vector<vk::Fence> fences;              // One per frame in flight
    std::vector<vk::Semaphore> image_availible; // One per frame in flight

    uint32_t frame_index = 0;

    vk::CommandPool command_pool;
    std::vector<vk::CommandBuffer> command_buffers; // One per frame in flight

    vk::Pipeline pipeline;
    vk::PipelineLayout pipeline_layout;

    vk::DescriptorPool descriptor_pool;
    vk::DescriptorSetLayout descriptor_set_layout;
    vk::DescriptorSet descriptor_set;

    vk::Buffer uniform_buffer;
    vma::Allocation uniform_buffer_alloc;
    void* uniform_buffer_mapped;
    uint32_t uniform_stride;

    vk::Buffer directional_light_buffer;
    vma::Allocation directional_light_buffer_alloc;
    uint32_t directional_light_buffer_stride;
    void *directional_light_buffer_mapped;

    vk::Buffer point_light_buffer;
    vma::Allocation point_light_buffer_alloc;
    uint32_t point_light_buffer_stride;
    void *point_light_buffer_mapped;

    vk::Buffer instance_buffer;
    vma::Allocation instance_buffer_alloc;
    size_t instance_count = 0;

    vk::Buffer vertex_buffer;
    vma::Allocation vertex_buffer_alloc;
    size_t vertex_count = 0;

    vk::Buffer index_buffer;
    vma::Allocation index_buffer_alloc;
    size_t index_count = 0;

    vk::Image image;
    vk::ImageView image_view;
    vk::Sampler image_sampler;
    vma::Allocation image_alloc;

    glm::mat4 world_to_clip_matrix;
    glm::vec3 view_position = glm::vec3(0.0f);
    glm::vec3 ambient_light_color = glm::vec3(0.5f);
    std::vector<PointLight> point_lights = {
        PointLight { 
            .position = glm::vec3(0.0f, -1.0f, 0.0f),
            .radius = 100.0f,
            .color = glm::vec3(1.0f, 1.0f, 1.0f),
            .intensity = 200.0f
        }
    };
    std::vector<DirectionalLight> directional_lights = {
        DirectionalLight { 
            .direction = glm::vec3(0.3f, 1.5f, 0.6f),
            .color = glm::vec3(1.0f, 1.0f, 1.0f),
            .illuminance = 1.0f
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

void Renderer::set_camera(const Camera& camera) {
    this->inner->set_camera(camera);
}

void Renderer::draw_frame() {
    this->inner->draw_frame();
}