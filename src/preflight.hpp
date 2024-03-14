#pragma once

#include <cstdio>
#include <map>
#include <vector>

namespace preflight {
inline std::map<void *, size_t> arrayAddressToArraySizeMap;
inline std::vector<std::vector<void *>> kernelDataDependencies;

template <typename T>
inline void registerArray(T *array, size_t size) {
  arrayAddressToArraySizeMap[static_cast<void *>(array)] = size;
}

inline void registerKernel(std::vector<void *> arrays) {
  kernelDataDependencies.push_back(arrays);
}

inline void printResult() {
  size_t totalArraySize = 0;
  for (const auto &[_, size] : arrayAddressToArraySizeMap) {
    totalArraySize += size;
  }

  size_t bottleNeckKernelDataDependencySize = 0;
  for (const auto &arrays : kernelDataDependencies) {
    size_t size = 0;
    for (auto array : arrays) {
      size += arrayAddressToArraySizeMap[array];
    }
    bottleNeckKernelDataDependencySize = std::max(bottleNeckKernelDataDependencySize, size);
  }

  printf(
    "[preflight] Total array size (MiB): %.4lf\n",
    static_cast<double>(totalArraySize) / 1024.0 / 1024.0
  );
  printf(
    "[preflight] Bottleneck kernel data dependency size (MiB): %.4lf\n",
    static_cast<double>(bottleNeckKernelDataDependencySize) / 1024.0 / 1024.0
  );
  printf(
    "[preflight] Bottleneck / Total: %.4lf\n",
    static_cast<double>(bottleNeckKernelDataDependencySize) / static_cast<double>(totalArraySize)
  );
}

template <typename T>
inline void wrappedCudaMalloc(T **ptr, size_t size) {
  cudaMalloc(ptr, size);
  registerArray(static_cast<void *>(*ptr), size);
}
}  // namespace preflight