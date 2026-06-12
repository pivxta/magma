#pragma once
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>
#include <functional>
 
template<typename T>
struct SlotKey {
    uint32_t index = 0;
    uint32_t generation = 0;
};

template<typename T>
class SlotMap {
public:
    SlotMap() = default;

    explicit SlotMap(uint32_t size_limit) {
        this->size_limit = size_limit;
        this->slots.reserve(size_limit);
    }
 
    bool is_valid(SlotKey<T> key) const {
        return key.index < this->slots.size()
            && key.generation % 2 == 1
            && key.generation == this->slots[key.index].generation;
    }
 
    std::optional<SlotKey<T>> reserve() {
        if (!this->free_indices.empty()) {
            uint32_t index = this->free_indices.front();
            uint32_t generation = ++this->slots[index].generation;
            this->free_indices.pop_front();
            return SlotKey<T>{index, generation};
        }
        if (!this->size_limit.has_value() || this->slots.size() < *this->size_limit) {
            T value;
            auto index = static_cast<uint32_t>(this->slots.size());
            this->slots.push_back({value, 1});
            return SlotKey<T>{index, 1};
        }
        return std::nullopt;
    }
 
    std::optional<SlotKey<T>> insert(const T& value) {
        if (auto key = this->reserve(); key.has_value()) {
            this->slots[key->index].value = value;
            return key;
        }
        return std::nullopt;
    }

    void for_each(const std::function<void(SlotKey<T>, const T&)>& pred) const {
        uint32_t index = 0;
        for (const auto& slot: this->slots) {
            if (slot.generation % 2 == 1) {
                pred(SlotKey<T>{index, slot.generation}, slot.value);
            }
            index++;
        }
    }

    void for_each(const std::function<void(SlotKey<T>, T&)>& pred) {
        uint32_t index = 0;
        for (auto& slot: this->slots) {
            if (slot.generation % 2 == 1) {
                pred(SlotKey<T>{index, slot.generation}, slot.value);
            }
            index++;
        }
    }
 
    const T* get(SlotKey<T> key) const {
        if (!this->is_valid(key)) {
            return nullptr;
        }
        return &this->slots[key.index].value;
    }
 
    T* get(SlotKey<T> key) {
        if (!this->is_valid(key)) {
            return nullptr;
        }
        return &this->slots[key.index].value;
    }
 
    bool free(SlotKey<T> key, const std::function<void(T&)>& destroy = {}) {
        if (!this->is_valid(key)) {
            return false;
        }
        if (destroy) {
            destroy(this->slots[key.index].value);
        }
        this->slots[key.index].generation++;
        this->free_indices.push_back(key.index);
        return true;
    }

    void clear(const std::function<void(T&)>& destroy = {}) {
        if (destroy) {
            for (uint32_t i = 0; i < this->slots.size(); i++) {
                Slot& slot = this->slots[i];
                if (slot.generation % 2 == 1) {
                    destroy(slot.value);
                }
            }
        }
        this->slots.clear();
        this->free_indices.clear();
    }

    std::optional<uint32_t> capacity() const {
        return this->size_limit;
    }
 
private:
    struct Slot {
        T value;
        uint32_t generation;
    };

    std::vector<Slot> slots;
    std::deque<uint32_t> free_indices;

    std::optional<uint32_t> size_limit;
};