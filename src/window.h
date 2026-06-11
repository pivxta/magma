#pragma once
#include <memory>
#include <vector>
#include <glm/vec2.hpp>

class Input;
class Target;

enum class WindowEvent: uint8_t {
    CloseRequested,
    Resized
};

struct WindowSize {
    uint32_t width;
    uint32_t height;
};

class Window {
public:
    Window();
    Window(uint32_t width, uint32_t height, const char *title);
    ~Window();

    Window(const Window&) = delete;
    const Window& operator=(const Window&) = delete;
    Window(Window&& other) noexcept;
    Window& operator=(Window&& other) noexcept;

    WindowSize logical_size() const;
    WindowSize physical_size() const;   
    Input input();
    std::shared_ptr<Target> target();

    void set_cursor_locked(bool locked);

    std::vector<WindowEvent> poll_events();

private:    
    struct Inner;
    std::shared_ptr<Inner> inner;
};