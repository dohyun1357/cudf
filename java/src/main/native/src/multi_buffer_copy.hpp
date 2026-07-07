/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>

namespace cudf {
namespace java {

/**
 * Copy many device buffers with one batched copy kernel.
 *
 * The address and size tables live in host memory; they are uploaded to the device with a
 * single copy that is synchronized with respect to `stream` before this function returns,
 * so the host arrays may be released by the caller on return. The copies themselves are
 * asynchronous on `stream`.
 *
 * @param dst_addrs host array of destination device addresses
 * @param src_addrs host array of source device addresses
 * @param copy_sizes host array of copy sizes in bytes
 * @param num_buffers number of buffers to copy
 * @param stream CUDA stream to use
 */
void multi_buffer_copy_async(int64_t const* dst_addrs,
                             int64_t const* src_addrs,
                             int64_t const* copy_sizes,
                             std::size_t num_buffers,
                             rmm::cuda_stream_view stream);

}  // namespace java
}  // namespace cudf
