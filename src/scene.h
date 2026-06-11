#pragma once
#include "camera.h"
#include "light.h"
#include "mesh_object.h"
#include "post_processing.h"

struct Scene {
    Camera camera;
    PostProcessing post;
    AmbientLight ambient_light;
    glm::vec3 clear_color;

    std::vector<PointLight> point_lights;
    std::vector<DirectionalLight> directional_lights;
    std::vector<MeshObject> mesh_objects;

    void add(const PointLight& light) {
        this->point_lights.push_back(light);
    }

    void add(const DirectionalLight& light) {
        this->directional_lights.push_back(light);
    }

    void add(const MeshObject& object) {
        this->mesh_objects.push_back(object);
    }
};