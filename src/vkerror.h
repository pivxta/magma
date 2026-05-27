#pragma once
#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

template<typename... Args>
static inline void vk_expect(vk::Result result, const char* msg) {
    if (result != vk::Result::eSuccess) {
        spdlog::critical(msg);
        spdlog::critical(
            "...Caused by Vulkan error (code {}): {}", 
            static_cast<int>(result), 
            vk::to_string(result)
        );
        std::exit(static_cast<int>(result));
    }
}

template<typename... Args>
static inline void vk_expect(VkResult result, const char* msg) {
    if (result != VK_SUCCESS) {
        spdlog::critical(msg);
        spdlog::critical(
            "...Caused by Vulkan error (code {}): {}",
            static_cast<int>(result),
            vk::to_string(static_cast<vk::Result>(result))
        );
        std::exit(static_cast<int>(result));
    }
}