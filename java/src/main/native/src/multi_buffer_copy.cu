/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "multi_buffer_copy.hpp"

#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <vector>

namespace cudf {
namespace java {

void multi_buffer_copy_async(int64_t const* dst_addrs,
                             int64_t const* src_addrs,
                             int64_t const* copy_sizes,
                             std::size_t num_buffers,
                             rmm::cuda_stream_view stream)
{
  if (num_buffers == 0) { return; }

  // Upload the address/size table in one copy, then run one batched copy kernel instead
  // of one memcpy call per buffer. make_device_uvector synchronizes the stream after the
  // upload, so the host staging vector may be released on return.
  std::vector<int64_t> host_table;
  host_table.reserve(num_buffers * 3);
  host_table.insert(host_table.end(), src_addrs, src_addrs + num_buffers);
  host_table.insert(host_table.end(), dst_addrs, dst_addrs + num_buffers);
  host_table.insert(host_table.end(), copy_sizes, copy_sizes + num_buffers);
  auto const d_table = cudf::detail::make_device_uvector(
    cudf::host_span<int64_t const>{host_table.data(), host_table.size()},
    stream,
    cudf::get_current_device_resource_ref());

  auto const src_iter  = reinterpret_cast<void* const*>(d_table.data());
  auto const dst_iter  = reinterpret_cast<void* const*>(d_table.data() + num_buffers);
  auto const size_iter = reinterpret_cast<std::size_t const*>(d_table.data() + 2 * num_buffers);
  cudf::detail::batched_memcpy_async(src_iter, dst_iter, size_iter, num_buffers, stream);
}

}  // namespace java
}  // namespace cudf
