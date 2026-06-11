#pragma once
#include "resource.h"
#include "transform.h"

struct MeshObject {
    MeshId mesh;
    MaterialId material;
    Transform transform;

    MeshObject& set_mesh(MeshId mesh) {
        this->mesh = mesh;
        return *this;
    }

    MeshObject& set_material(MaterialId material) {
        this->material = material;
        return *this;
    }

    MeshObject& set_transform(const Transform& transform) {
        this->transform = transform;
        return *this;
    }
};