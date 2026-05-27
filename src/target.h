#pragma once
#include <tuple>
#include <vector>
#include <vulkan/vulkan.hpp>

class Target {
public:
    virtual std::vector<const char*> required_instance_extensions() const = 0;
    virtual std::tuple<vk::Result, vk::SurfaceKHR> target_surface(vk::Instance instance) = 0;
    virtual vk::Extent2D target_extent() = 0;
};