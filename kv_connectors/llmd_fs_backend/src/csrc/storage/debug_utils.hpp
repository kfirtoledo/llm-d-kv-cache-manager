#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <chrono>

// -------------------------------------
// Debugging and timing macros
// -------------------------------------

// Debug print - enabled when STORAGE_CONNECTOR_DEBUG is set and not "0"
#define DEBUG_PRINT(msg)                                                     \
    do {                                                                     \
        const char* env = std::getenv("STORAGE_CONNECTOR_DEBUG");            \
        if (env && std::string(env) != "0")                                 \
            std::cout << "[DEBUG] " << msg << std::endl;                     \
    } while (0)

// Timing macro - measures execution time when STORAGE_CONNECTOR_DEBUG  is set and not "0"
#define TIME_EXPR(label, expr, info_str) ([&]() {                                  \
    const char* env = std::getenv("STORAGE_CONNECTOR_DEBUG");                      \
    if (!(env && std::string(env) != "0")) {                                       \
        return (expr);                                                             \
    }                                                                              \
    auto __t0 = std::chrono::high_resolution_clock::now();                         \
    auto __ret = (expr);                                                           \
    auto __t1 = std::chrono::high_resolution_clock::now();                         \
    double __ms = std::chrono::duration<double, std::milli>(__t1 - __t0).count();  \
    std::cout << "[DEBUG][TIME] " << label << " took " << __ms << " ms | "         \
              << info_str << std::endl;                                            \
    return __ret;                                                                  \
})()
