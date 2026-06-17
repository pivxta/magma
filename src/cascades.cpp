#include "cascades.h"
#include <algorithm>

static std::array<float, MAX_CASCADES> get_cascade_depths(
    const CascadeInfo& info, 
    const Camera& camera
) {
    const float near = std::max(camera.near(), info.min_distance);
    const float far = std::min(camera.far(), info.max_distance);
    std::array<float, MAX_CASCADES> depths;
    depths.fill(0.0f);
    for (size_t i = 0; i < info.count; i++) {
        const auto t = static_cast<float>(i + 1) / static_cast<float>(info.count);
        const float uniform = near + (far - near) * t;
        const float log = near * std::pow(far / near, t);
        depths[i] = glm::mix(uniform, log, info.lambda);
    }
    return depths;
}

static glm::mat4 get_cascade_matrix(
    const CascadeInfo& info,
    const DirectionalLight& light,
    const std::array<glm::vec3, 8>& corners
) {
    const float radius_resolution = 16.0f;
    const glm::vec3 center = std::ranges::fold_left(corners, glm::vec3(0.0f), std::plus<>{}) / 8.0f;
    float radius = 0.0f;
    for (const auto& corner: corners) {
        radius = std::max(radius, glm::distance(corner, center));
    }
    radius = std::ceil(radius * radius_resolution) / radius_resolution;

    const auto direction = glm::normalize(light.direction);
    const auto up = glm::abs(direction.y) < 0.99f ?
        glm::vec3(0.0f, -1.0f, 0.0f) :
        glm::vec3(0.0f, 0.0f, 1.0f);
    auto world_to_view = glm::lookAtLH(center - direction * radius, center, up);
    auto view_to_clip = glm::orthoLH_ZO(-radius, radius, -radius, radius, 2.0f * radius, 0.0f);

    const auto origin_clip = view_to_clip * world_to_view * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    const auto half_resolution = 0.5f * static_cast<float>(info.resolution);
    const auto origin_texels = glm::vec2(origin_clip) * half_resolution;
    const auto snap_offset = (glm::round(origin_texels) - origin_texels) / half_resolution;
    view_to_clip[3][0] += snap_offset.x;
    view_to_clip[3][1] += snap_offset.y;

    return view_to_clip * world_to_view;
}

static std::array<glm::mat4, MAX_CASCADES> get_cascade_matrices(
    const CascadeInfo& info,
    const Camera& camera,
    glm::vec2 viewport,
    const DirectionalLight& light,
    const std::array<float, MAX_CASCADES>& depths
) {
    const float near = std::max(camera.near(), info.min_distance);
    std::array<glm::mat4, MAX_CASCADES> matrices;
    matrices.fill(glm::identity<glm::mat4>());
    for (size_t i = 0; i < info.count; i++) {
        const auto cascade_far = depths[i];
        const auto cascade_near = i == 0 ? near : depths[i - 1];
        const auto cascade_corners = camera.frustum_corners(viewport, cascade_near, cascade_far);
        matrices[i] = get_cascade_matrix(info, light, cascade_corners);
    }
    return matrices;
}

Cascades::Cascades(
    const CascadeInfo& info,
    const Camera& camera, 
    glm::vec2 viewport,
    const DirectionalLight& light
) {
    this->count = info.count;
    this->depths = get_cascade_depths(info, camera);
    this->world_to_clip = get_cascade_matrices(info, camera, viewport, light, this->depths);
}