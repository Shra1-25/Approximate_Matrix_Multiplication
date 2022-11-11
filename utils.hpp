#ifndef UTILS_HPP
#define UTILS_HPP

#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <iostream>

#define MEMORY_ALIGNMENT 64

class Timer {
  public:

    void tic() {
      t_start = std::chrono::high_resolution_clock::now();
    }

    double toc() {
      auto t_end = std::chrono::high_resolution_clock::now();
      return std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
};

#endif