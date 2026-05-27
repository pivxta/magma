#include "time.h"
#include <GLFW/glfw3.h>

double get_current_time_seconds() {
    return glfwGetTime();
}