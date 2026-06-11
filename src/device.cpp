#include "device.h"
#include "vk_error.h"
#include "swapchain.h"
#include "target.h"
#include <spdlog/spdlog.h>

#ifdef NDEBUG
static constexpr bool ENABLE_VALIDATION_LAYERS = false;
#else
static constexpr bool ENABLE_VALIDATION_LAYERS = true;
#endif
static constexpr const char* VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation";

static bool validation_layer_available() {
    auto [result, layers] = vk::enumerateInstanceLayerProperties();
    if (result != vk::Result::eSuccess) {
        return false;
    }
    for (const auto& layer : layers) {
        if (strcmp(layer.layerName, VALIDATION_LAYER_NAME) == 0) {
            return true;
        }
    }
    return false;
}

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debug_callback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT,
    const vk::DebugUtilsMessengerCallbackDataEXT* data,
    void*
) {
    const char* id = data->pMessageIdName != nullptr ? data->pMessageIdName : "";
    switch (severity) {
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
            spdlog::error("[vulkan] {}: {}", id, data->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
            spdlog::warn("[vulkan] {}: {}", id, data->pMessage);
            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
            spdlog::debug("[vulkan] {}: {}", id, data->pMessage);
            break;
        default:
            spdlog::trace("[vulkan] {}: {}", id, data->pMessage);
            break;
    }
    return vk::False;
}

static vk::DebugUtilsMessengerCreateInfoEXT make_debug_messenger_create_info() {
    using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;
    return vk::DebugUtilsMessengerCreateInfoEXT()
        .setMessageSeverity(Severity::eWarning | Severity::eError)
        .setMessageType(Type::eGeneral | Type::eValidation | Type::ePerformance)
        .setPfnUserCallback(debug_callback);
}

InstanceHandle create_instance(const Target& target) {
    std::vector<const char*> instance_layers;
    if constexpr (ENABLE_VALIDATION_LAYERS) {
        if (validation_layer_available()) {
            instance_layers.push_back(VALIDATION_LAYER_NAME);
            spdlog::info("Validation layers enabled");
        } else {
            spdlog::warn(
                "Validation layers requested but '{}' is not available "
                "(is the Vulkan SDK installed?); continuing without them",
                VALIDATION_LAYER_NAME
            );
        }
    }

    const bool validation_enabled = !instance_layers.empty();

    std::vector instance_extensions = target.required_instance_extensions();
    if (validation_enabled) {
        instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    auto debug_ci = make_debug_messenger_create_info();

    auto application = vk::ApplicationInfo().setApiVersion(vk::ApiVersion13);
    auto instance_ci = vk::InstanceCreateInfo()
        .setPEnabledExtensionNames(instance_extensions)
        .setPEnabledLayerNames(instance_layers)
        .setPApplicationInfo(&application);
    if (validation_enabled) {
        instance_ci.setPNext(&debug_ci);
    }

    auto [result, instance] = vk::createInstance(instance_ci);
    vk_expect(result, "Failed to create instance");

    vk::DebugUtilsMessengerEXT debug_messenger = {};
    if (validation_enabled) {
        vk::detail::DispatchLoaderDynamic dldi(instance, vkGetInstanceProcAddr);
        auto [msg_result, messenger] =
            instance.createDebugUtilsMessengerEXT(debug_ci, nullptr, dldi);
        vk_expect(msg_result, "Failed to create debug messenger");
        debug_messenger = messenger;
    }

    auto ctx = std::make_shared<InstanceContext>();
    ctx->instance = instance;
    ctx->debug_messenger = debug_messenger;
    return ctx;
}

InstanceContext::~InstanceContext() {
    if (!this->instance) {
        return;
    }
    if (this->debug_messenger) {
        vk::detail::DispatchLoaderDynamic dldi(this->instance, vkGetInstanceProcAddr);
        this->instance.destroyDebugUtilsMessengerEXT(this->debug_messenger, nullptr, dldi);
    }
    this->instance.destroy();
}

static bool device_has_swapchain_ext(vk::PhysicalDevice device) {
    auto [result, extensions] = device.enumerateDeviceExtensionProperties();
    if (result != vk::Result::eSuccess) {
        spdlog::error("Failed to enumerate physical device extension properties");
        return false;
    }

    for (auto extension : extensions) {
        if (strcmp(extension.extensionName, vk::KHRSwapchainExtensionName) == 0) {
            return true;
        }
    }

    return false;
}

static std::optional<uint32_t> pick_graphics_queue_family(
    vk::PhysicalDevice device, 
    vk::SurfaceKHR surface
) {
    std::optional<uint32_t> selected_family = std::nullopt;
    std::vector families = device.getQueueFamilyProperties();
    for (uint32_t family = 0; family < families.size(); family++) {
        vk::QueueFamilyProperties properties = families[family];
        if (!(properties.queueFlags & vk::QueueFlagBits::eGraphics)) {
            continue;
        }

        auto [result, support] = device.getSurfaceSupportKHR(family, surface);
        if (result != vk::Result::eSuccess || !support) {
            continue;
        }

        selected_family = family;
    }

    return selected_family;
}

static bool pick_physical_device(DeviceContext& device, vk::SurfaceKHR surface) {
    auto [result, devices] = device.instance->instance.enumeratePhysicalDevices();
    if (result != vk::Result::eSuccess) {
        spdlog::error("Failed to enumerate physical devices");
        return false;
    }

    for (auto physical : devices) {
        vk::PhysicalDeviceProperties2 props = physical.getProperties2();
        if (props.properties.apiVersion < vk::ApiVersion13) {
            continue;
        }
        if (!device_has_swapchain_ext(physical)) {
            continue;
        }
        auto queue_family_index = pick_graphics_queue_family(physical, surface);
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

            device.physical = physical;
            device.properties = props;
            device.graphics_queue_family = *queue_family_index;
            return true;
        }
    }
    return false;
}

static bool create_device_and_queue(DeviceContext& device) {
    std::array queue_priorities = {1.0f};
    std::array enabled_extensions = {vk::KHRSwapchainExtensionName};
    auto features = vk::PhysicalDeviceFeatures()
        .setSamplerAnisotropy(true)
        .setMultiDrawIndirect(true);

    auto [result, logical] = device.physical.createDevice(
        vk::StructureChain{
            vk::DeviceCreateInfo()
                .setPEnabledFeatures(&features)
                .setPEnabledExtensionNames(enabled_extensions)
                .setQueueCreateInfos(
                    vk::DeviceQueueCreateInfo()
                        .setQueuePriorities(queue_priorities)
                        .setQueueFamilyIndex(device.graphics_queue_family)
                ),

            vk::PhysicalDeviceVulkan13Features()
                .setDynamicRendering(true)
                .setSynchronization2(true),
            
            vk::PhysicalDeviceVulkan12Features()
                .setDrawIndirectCount(true)
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
    if (result != vk::Result::eSuccess) {
        spdlog::error("Failed to create device");
        return false;
    }

    device.logical = logical;
    device.graphics_queue = logical.getQueue(device.graphics_queue_family, 0);

    auto vma_vk_functions = vma::VulkanFunctions()
        .setVkGetInstanceProcAddr(vkGetInstanceProcAddr)
        .setVkGetDeviceProcAddr(vkGetDeviceProcAddr);
    auto vma_info = vma::AllocatorCreateInfo()
        .setFlags(vma::AllocatorCreateFlagBits::eBufferDeviceAddress)
        .setInstance(device.instance->instance)
        .setDevice(device.logical)
        .setPhysicalDevice(device.physical)
        .setPVulkanFunctions(&vma_vk_functions);
    auto [alloc_result, allocator] = vma::createAllocator(vma_info);
    if (alloc_result != vk::Result::eSuccess) {
        spdlog::error("Failed to create allocator");
        return false;
    }
    device.allocator = allocator;

    return true;
}

std::optional<DeviceHandle> create_device(
    const InstanceHandle& instance,
    const Swapchain& swapchain
) {
    auto ctx = std::make_shared<DeviceContext>();
    ctx->instance = instance;
    if (!pick_physical_device(*ctx, swapchain.surface())) {
        return std::nullopt;
    }
    if (!create_device_and_queue(*ctx)) {
        return std::nullopt;
    }
    return ctx;
}

void DeviceContext::wait_idle() const {
    (void)this->logical.waitIdle();
}

DeviceContext::~DeviceContext() {
    if (!this->logical) {
        return;
    }
    this->wait_idle();
    if (this->allocator) {
        this->allocator.destroy();
    }
    this->logical.destroy();
}

