#include "window.h"
#include "input.h"
#include "target.h"
#include <bitset>
#include <GLFW/glfw3.h>

static constexpr size_t NUM_KEYS = static_cast<size_t>(Key::Last) + 1;
static constexpr size_t NUM_MOUSE_BUTTONS = static_cast<size_t>(MouseButton::Last) + 1; 

static inline Key glfw_to_key(int key) {
    switch (key) {
        case GLFW_KEY_SPACE: return Key::Space;        
        case GLFW_KEY_APOSTROPHE: return Key::Apostrophe;   
        case GLFW_KEY_COMMA: return Key::Comma;        
        case GLFW_KEY_MINUS: return Key::Minus;        
        case GLFW_KEY_PERIOD: return Key::Period;       
        case GLFW_KEY_SLASH: return Key::Slash;        
        case GLFW_KEY_0: return Key::Num0;         
        case GLFW_KEY_1: return Key::Num1;         
        case GLFW_KEY_2: return Key::Num2;         
        case GLFW_KEY_3: return Key::Num3;         
        case GLFW_KEY_4: return Key::Num4;         
        case GLFW_KEY_5: return Key::Num5;         
        case GLFW_KEY_6: return Key::Num6;         
        case GLFW_KEY_7: return Key::Num7;         
        case GLFW_KEY_8: return Key::Num8;         
        case GLFW_KEY_9: return Key::Num9;         
        case GLFW_KEY_SEMICOLON: return Key::Semicolon;    
        case GLFW_KEY_EQUAL: return Key::Equal;        
        case GLFW_KEY_A: return Key::A;            
        case GLFW_KEY_B: return Key::B;            
        case GLFW_KEY_C: return Key::C;            
        case GLFW_KEY_D: return Key::D;            
        case GLFW_KEY_E: return Key::E;            
        case GLFW_KEY_F: return Key::F;            
        case GLFW_KEY_G: return Key::G;            
        case GLFW_KEY_H: return Key::H;            
        case GLFW_KEY_I: return Key::I;            
        case GLFW_KEY_J: return Key::J;            
        case GLFW_KEY_K: return Key::K;            
        case GLFW_KEY_L: return Key::L;            
        case GLFW_KEY_M: return Key::M;            
        case GLFW_KEY_N: return Key::N;            
        case GLFW_KEY_O: return Key::O;            
        case GLFW_KEY_P: return Key::P;            
        case GLFW_KEY_Q: return Key::Q;            
        case GLFW_KEY_R: return Key::R;            
        case GLFW_KEY_S: return Key::S;            
        case GLFW_KEY_T: return Key::T;            
        case GLFW_KEY_U: return Key::U;            
        case GLFW_KEY_V: return Key::V;            
        case GLFW_KEY_W: return Key::W;            
        case GLFW_KEY_X: return Key::X;            
        case GLFW_KEY_Y: return Key::Y;            
        case GLFW_KEY_Z: return Key::Z;            
        case GLFW_KEY_LEFT_BRACKET: return Key::LeftBracket;  
        case GLFW_KEY_BACKSLASH: return Key::Backslash;    
        case GLFW_KEY_RIGHT_BRACKET: return Key::RightBracket; 
        case GLFW_KEY_GRAVE_ACCENT: return Key::GraveAccent;  
        case GLFW_KEY_WORLD_1: return Key::World1;       
        case GLFW_KEY_WORLD_2: return Key::World2;       
        case GLFW_KEY_ESCAPE: return Key::Escape;       
        case GLFW_KEY_ENTER: return Key::Enter;        
        case GLFW_KEY_TAB: return Key::Tab;          
        case GLFW_KEY_BACKSPACE: return Key::Backspace;    
        case GLFW_KEY_INSERT: return Key::Insert;       
        case GLFW_KEY_DELETE: return Key::Delete;       
        case GLFW_KEY_RIGHT: return Key::Right;        
        case GLFW_KEY_LEFT: return Key::Left;         
        case GLFW_KEY_DOWN: return Key::Down;         
        case GLFW_KEY_UP: return Key::Up;           
        case GLFW_KEY_PAGE_UP: return Key::PageUp;       
        case GLFW_KEY_PAGE_DOWN: return Key::PageDown;     
        case GLFW_KEY_HOME: return Key::Home;         
        case GLFW_KEY_END: return Key::End;          
        case GLFW_KEY_CAPS_LOCK: return Key::CapsLock;     
        case GLFW_KEY_SCROLL_LOCK: return Key::ScrollLock;   
        case GLFW_KEY_NUM_LOCK: return Key::NumLock;      
        case GLFW_KEY_PRINT_SCREEN: return Key::PrintScreen;  
        case GLFW_KEY_PAUSE: return Key::Pause;        
        case GLFW_KEY_F1: return Key::F1;           
        case GLFW_KEY_F2: return Key::F2;           
        case GLFW_KEY_F3: return Key::F3;           
        case GLFW_KEY_F4: return Key::F4;           
        case GLFW_KEY_F5: return Key::F5;           
        case GLFW_KEY_F6: return Key::F6;           
        case GLFW_KEY_F7: return Key::F7;           
        case GLFW_KEY_F8: return Key::F8;           
        case GLFW_KEY_F9: return Key::F9;           
        case GLFW_KEY_F10: return Key::F10;          
        case GLFW_KEY_F11: return Key::F11;          
        case GLFW_KEY_F12: return Key::F12;          
        case GLFW_KEY_F13: return Key::F13;          
        case GLFW_KEY_F14: return Key::F14;          
        case GLFW_KEY_F15: return Key::F15;          
        case GLFW_KEY_F16: return Key::F16;          
        case GLFW_KEY_F17: return Key::F17;          
        case GLFW_KEY_F18: return Key::F18;          
        case GLFW_KEY_F19: return Key::F19;          
        case GLFW_KEY_F20: return Key::F20;            // add F20 to your enum
        case GLFW_KEY_F21: return Key::F21;          
        case GLFW_KEY_F22: return Key::F22;          
        case GLFW_KEY_F23: return Key::F23;          
        case GLFW_KEY_F24: return Key::F24;          
        case GLFW_KEY_F25: return Key::F25;          
        case GLFW_KEY_KP_0: return Key::Numpad0;        
        case GLFW_KEY_KP_1: return Key::Numpad1;        
        case GLFW_KEY_KP_2: return Key::Numpad2;        
        case GLFW_KEY_KP_3: return Key::Numpad3;        
        case GLFW_KEY_KP_4: return Key::Numpad4;        
        case GLFW_KEY_KP_5: return Key::Numpad5;        
        case GLFW_KEY_KP_6: return Key::Numpad6;        
        case GLFW_KEY_KP_7: return Key::Numpad7;        
        case GLFW_KEY_KP_8: return Key::Numpad8;        
        case GLFW_KEY_KP_9: return Key::Numpad9;        
        case GLFW_KEY_KP_DECIMAL: return Key::NumpadDecimal;  
        case GLFW_KEY_KP_DIVIDE: return Key::NumpadDivide;   
        case GLFW_KEY_KP_MULTIPLY: return Key::NumpadMultiply; 
        case GLFW_KEY_KP_SUBTRACT: return Key::NumpadSubtract; 
        case GLFW_KEY_KP_ADD: return Key::NumpadAdd;      
        case GLFW_KEY_KP_ENTER: return Key::NumpadEnter;    
        case GLFW_KEY_KP_EQUAL: return Key::NumpadEqual;    
        case GLFW_KEY_LEFT_SHIFT: return Key::LeftShift;      
        case GLFW_KEY_LEFT_CONTROL: return Key::LeftControl;    
        case GLFW_KEY_LEFT_ALT: return Key::LeftAlt;        
        case GLFW_KEY_LEFT_SUPER: return Key::LeftSuper;      
        case GLFW_KEY_RIGHT_SHIFT: return Key::RightShift;     
        case GLFW_KEY_RIGHT_CONTROL: return Key::RightControl;   
        case GLFW_KEY_RIGHT_ALT: return Key::RightAlt;       
        case GLFW_KEY_RIGHT_SUPER: return Key::RightSuper;     
        case GLFW_KEY_MENU: return Key::Menu;           
        default: return Key::Unknown;
    }
}

static inline MouseButton glfw_to_mouse_button(int button) {
    switch (button) {
        case GLFW_MOUSE_BUTTON_1: return MouseButton::Button1;
        case GLFW_MOUSE_BUTTON_2: return MouseButton::Button2;
        case GLFW_MOUSE_BUTTON_3: return MouseButton::Button3;
        case GLFW_MOUSE_BUTTON_4: return MouseButton::Button4;
        case GLFW_MOUSE_BUTTON_5: return MouseButton::Button5;
        case GLFW_MOUSE_BUTTON_6: return MouseButton::Button6;
        case GLFW_MOUSE_BUTTON_7: return MouseButton::Button7;
        case GLFW_MOUSE_BUTTON_8: return MouseButton::Button8;
        default: return MouseButton::Unknown;
    }
}

struct Input::State {
    std::bitset<NUM_KEYS> keys_pressed;
    std::bitset<NUM_KEYS> keys_just_pressed;
    std::bitset<NUM_KEYS> keys_just_released;
    std::bitset<NUM_MOUSE_BUTTONS> mouse_buttons_pressed;
    std::bitset<NUM_MOUSE_BUTTONS> mouse_buttons_just_pressed;
    std::bitset<NUM_MOUSE_BUTTONS> mouse_buttons_just_released;

    double cursor_x = 0.0;
    double cursor_y = 0.0;
    double cursor_delta_x = 0.0;
    double cursor_delta_y = 0.0;
    double scroll_delta_x = 0.0;
    double scroll_delta_y = 0.0;

    void reset_deltas() {
        this->cursor_delta_x = 0.0;
        this->cursor_delta_y = 0.0;
        this->scroll_delta_x = 0.0;
        this->scroll_delta_y = 0.0;
        this->keys_just_pressed.reset();
        this->keys_just_released.reset();
        this->mouse_buttons_just_pressed.reset();
        this->mouse_buttons_just_released.reset();
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    void handle_cursor(double x, double y) {
        this->cursor_delta_x += x - this->cursor_x;
        this->cursor_delta_y += y - this->cursor_y;
        this->cursor_x = x;    
        this->cursor_y = y;
    }

    void handle_scroll(double x, double y) {
        this->scroll_delta_x += x;    
        this->scroll_delta_y += y;
    }

    void handle_key(int key_glfw, int action) {
        auto key_index = static_cast<size_t>(glfw_to_key(key_glfw));
        switch (action) {
            case GLFW_PRESS:
                this->keys_just_pressed.set(key_index, !this->keys_pressed[key_index]);
                this->keys_pressed.set(key_index, true);
                break;
            case GLFW_RELEASE:
                this->keys_just_released.set(key_index, true);
                this->keys_pressed.set(key_index, false);
                break;
            default:
                break;
        }
    }

    void handle_mouse_button(int button_glfw, int action) {
        auto button_index = static_cast<size_t>(glfw_to_mouse_button(button_glfw));
        switch (action) {
            case GLFW_PRESS:
                this->mouse_buttons_just_pressed.set(
                    button_index, 
                    !this->mouse_buttons_pressed[button_index]
                );
                this->mouse_buttons_pressed.set(button_index, true);
                break;
            case GLFW_RELEASE:
                this->mouse_buttons_just_released.set(button_index, true);
                this->mouse_buttons_pressed.set(button_index, false);
                break;
            default:
                break;
        }
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)
};

Input::Input()
    : state(std::make_shared<State>()) {}

bool Input::key_pressed(Key key) const {
    return this->state->keys_pressed[static_cast<size_t>(key)];
}

bool Input::key_just_pressed(Key key) const {
    return this->state->keys_just_pressed[static_cast<size_t>(key)];
}

bool Input::key_just_released(Key key) const {
    return this->state->keys_just_released[static_cast<size_t>(key)];
}

bool Input::mouse_button_pressed(MouseButton button) const {
    return this->state->mouse_buttons_pressed[static_cast<size_t>(button)];
}

bool Input::mouse_button_just_pressed(MouseButton button) const {
    return this->state->mouse_buttons_just_pressed[static_cast<size_t>(button)];
}

bool Input::mouse_button_just_released(MouseButton button) const {
    return this->state->mouse_buttons_just_released[static_cast<size_t>(button)];
}

glm::vec2 Input::cursor_position() const {
    return { 
        static_cast<float>(this->state->cursor_x), 
        static_cast<float>(this->state->cursor_y)
    };
}

glm::vec2 Input::cursor_delta() const {
    return { 
        static_cast<float>(this->state->cursor_delta_x), 
        static_cast<float>(this->state->cursor_delta_y) 
    };
}

glm::vec2 Input::scroll_delta() const {
    return { 
        static_cast<float>(this->state->scroll_delta_x), 
        static_cast<float>(this->state->scroll_delta_y)
    };
}

static constexpr int MAX_WIDTH = 32768;
static constexpr int MAX_HEIGHT = 32768;

struct Window::Inner: Target {
    GLFWwindow* window = nullptr;
    Input input;
    vk::SurfaceKHR surface;
    std::vector<WindowEvent> events;

    virtual ~Inner() {
        this->destroy();
    }

    void create(uint32_t width, uint32_t height, const char *title) {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        this->window = glfwCreateWindow(
            static_cast<int>(width), 
            static_cast<int>(height), 
            title, nullptr, nullptr
        );
        glfwSetWindowUserPointer(this->window, this);
        glfwSetWindowSizeLimits(this->window, 1, 1, MAX_WIDTH, MAX_HEIGHT);
        glfwSetWindowSizeCallback(this->window, handle_resize);
        glfwSetCursorPosCallback(this->window, handle_cursor);
        glfwSetScrollCallback(this->window, handle_scroll);
        glfwSetKeyCallback(this->window, handle_key);
        glfwSetMouseButtonCallback(this->window, handle_mouse_button);
    }

    void destroy() {
        if (this->window != nullptr) {
            glfwDestroyWindow(this->window);
            this->window = nullptr;
        }
        glfwTerminate();
    }

    WindowSize logical_size() const {
        int width, height;
        glfwGetWindowSize(this->window, &width, &height);
        return { 
            .width = static_cast<uint32_t>(width), 
            .height = static_cast<uint32_t>(height) 
        };
    }

    WindowSize physical_size() const {
        int width, height;
        glfwGetFramebufferSize(this->window, &width, &height);
        return { 
            .width = static_cast<uint32_t>(width), 
            .height = static_cast<uint32_t>(height) 
        };
    }

    void set_cursor_locked(bool locked) {
        int mode = locked ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL; 
        glfwSetInputMode(this->window, GLFW_CURSOR, mode);
    }

    std::vector<WindowEvent> poll_events() {
        if (glfwWindowShouldClose(this->window)) {
            this->events.push_back(WindowEvent::CloseRequested);
        } else {
            this->input.state->reset_deltas();
            glfwPollEvents();
        }
        return std::move(this->events);
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    static void handle_resize(GLFWwindow* window, int, int) {
        auto inner = reinterpret_cast<Inner*>(glfwGetWindowUserPointer(window));
        inner->events.push_back(WindowEvent::Resized);
    }
    
    static void handle_cursor(GLFWwindow* window, double x, double y) {
        auto inner = reinterpret_cast<Inner*>(glfwGetWindowUserPointer(window));
        inner->input.state->handle_cursor(x, y);
    }

    static void handle_scroll(GLFWwindow* window, double x, double y) {
        auto inner = reinterpret_cast<Inner*>(glfwGetWindowUserPointer(window));
        inner->input.state->handle_scroll(x, y);
    }

    static void handle_key(GLFWwindow* window, int key, int, int action, int) {
        auto inner = reinterpret_cast<Inner*>(glfwGetWindowUserPointer(window));
        inner->input.state->handle_key(key, action);
    }

    static void handle_mouse_button(GLFWwindow* window, int button, int action, int) {
        auto inner = reinterpret_cast<Inner*>(glfwGetWindowUserPointer(window));
        inner->input.state->handle_mouse_button(button, action);
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    std::vector<const char*> required_instance_extensions() const override {
        uint32_t instance_ext_count;
        const char** instance_ext_names = glfwGetRequiredInstanceExtensions(&instance_ext_count);
        return {
            instance_ext_names, 
            instance_ext_names + static_cast<size_t>(instance_ext_count)
        };
    }

    std::tuple<vk::Result, vk::SurfaceKHR> target_surface(vk::Instance instance) override {
        vk::Result result = vk::Result::eSuccess;
        if (this->surface == vk::SurfaceKHR()) {
            VkSurfaceKHR surface;
            result = static_cast<vk::Result>(
                glfwCreateWindowSurface(instance, this->window, nullptr, &surface)
            );
            this->surface = surface;
        }
        return {result, this->surface};
    }

    vk::Extent2D target_extent() override {
        int width, height;
        glfwGetFramebufferSize(this->window, &width, &height);
        return { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    }
};

Window::Window() = default;

Window::Window(uint32_t width, uint32_t height, const char *title)
    : inner(std::make_shared<Inner>())
{
    this->inner->create(width, height, title);
}

Window::Window(Window&& other) noexcept {
    this->inner = std::move(other.inner);
}

Window& Window::operator=(Window&& other) noexcept {
    this->inner = std::move(other.inner);
    return *this;
}

Window::~Window() = default;

WindowSize Window::logical_size() const {
    return this->inner->logical_size();
}

WindowSize Window::physical_size() const {
    return this->inner->physical_size();
}

Input Window::input() {
    return this->inner->input;
}

std::shared_ptr<Target> Window::target() {
    return this->inner;
}

void Window::set_cursor_locked(bool locked) {
    this->inner->set_cursor_locked(locked);
}

std::vector<WindowEvent> Window::poll_events() {
    return this->inner->poll_events();
}

