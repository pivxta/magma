#pragma once
#include <spdlog/spdlog.h>

inline void configure_logging() {
#ifdef NDEBUG
    spdlog::set_level(spdlog::level::info);
#else
    spdlog::set_level(spdlog::level::debug);
#endif
}
