// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/tensor/converter.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "arrow/buffer.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/logging_internal.h"
#include "arrow/util/macros.h"
#include "arrow/visit_type_inline.h"

namespace arrow {

class MemoryPool;

namespace internal {
namespace {

template <typename c_index_type>
inline void IncrementRowMajorIndex(std::vector<c_index_type>& coord,
                                   const std::vector<int64_t>& shape) {
  const int64_t ndim = shape.size();
  ++coord[ndim - 1];
  if (coord[ndim - 1] == shape[ndim - 1]) {
    int64_t d = ndim - 1;
    while (d > 0 && coord[d] == shape[d]) {
      coord[d] = 0;
      ++coord[d - 1];
      --d;
    }
  }
}

template <typename c_index_type>
inline void IncrementColumnMajorIndex(std::vector<c_index_type>& coord,
                                      const std::vector<int64_t>& shape) {
  int64_t ndim = shape.size();
  ++coord[0];
  int64_t i = 1;
  while (i < ndim && coord[i - 1] == static_cast<c_index_type>(shape[i - 1])) {
    coord[i - 1] = 0;
    ++coord[i];
    i++;
  }
}

template <bool is_row_major, typename ValueType, typename c_index_type,
          typename c_value_type>
void ConvertContinuousTensor(const Tensor& tensor, c_index_type* indices,
                             c_value_type* values, const int64_t size) {
  const auto ndim = tensor.ndim();
  const auto& shape = tensor.shape();
  const c_value_type* tensor_data =
      reinterpret_cast<const c_value_type*>(tensor.raw_data());

  std::vector<c_index_type> coord(ndim, 0);
  for (int64_t n = tensor.size(); n > 0; --n) {
    const c_value_type x = *tensor_data;
    if (is_not_zero<ValueType>(x)) {
      std::copy(coord.begin(), coord.end(), indices);
      *values++ = x;
      indices += ndim;
    }
    if constexpr (is_row_major) {
      IncrementRowMajorIndex(coord, shape);
    } else {
      IncrementColumnMajorIndex(coord, shape);
    }

    ++tensor_data;
  }
}

template <typename ValueType, typename c_index_type, typename c_value_type>
void ConvertRowMajorTensor(const Tensor& tensor, c_index_type* out_indices,
                           c_value_type* out_values, const int64_t size) {
  ConvertContinuousTensor<true, ValueType>(tensor, out_indices, out_values, size);
}

template <typename ValueType, typename c_index_type, typename c_value_type>
void ConvertColumnMajorTensor(const Tensor& tensor, c_index_type* out_indices,
                              c_value_type* out_values, const int64_t size) {
  ConvertContinuousTensor<false, ValueType>(tensor, out_indices, out_values, size);
  // TODO should We sort Indices for correct Equality Result with Row Major
}

template <typename ValueType, typename c_index_type, typename c_value_type>
void ConvertStridedTensor(const Tensor& tensor, c_index_type* indices,
                          c_value_type* values, const int64_t size) {
  const auto& shape = tensor.shape();
  const auto ndim = tensor.ndim();
  std::vector<int64_t> coord(ndim, 0);

  c_value_type x;
  int64_t i;
  for (int64_t n = tensor.size(); n > 0; --n) {
    x = tensor.Value<ValueType>(coord);
    if (is_not_zero<ValueType>(x)) {
      *values++ = x;
      for (i = 0; i < ndim; ++i) {
        *indices++ = static_cast<c_index_type>(coord[i]);
      }
    }

    IncrementRowMajorIndex(coord, shape);
  }
}

// ----------------------------------------------------------------------
// SparseTensorConverter for SparseCOOIndex

class SparseCOOTensorConverter {
 public:
  SparseCOOTensorConverter(const Tensor& tensor,
                           const std::shared_ptr<DataType>& index_value_type,
                           MemoryPool* pool)
      : tensor_(tensor), index_value_type_(index_value_type), pool_(pool) {}

  template <typename ValueDataType, typename IndexCtype, typename ValueCtype>
  Status Convert(const ValueDataType&, IndexCtype, ValueCtype) {
    RETURN_NOT_OK(::arrow::internal::CheckSparseIndexMaximumValue(index_value_type_,
                                                                  tensor_.shape()));

    const int index_elsize = index_value_type_->byte_width();
    const int value_elsize = tensor_.type()->byte_width();

    const int64_t ndim = tensor_.ndim();
    ARROW_ASSIGN_OR_RAISE(int64_t nonzero_count, tensor_.CountNonZero());

    ARROW_ASSIGN_OR_RAISE(auto indices_buffer,
                          AllocateBuffer(index_elsize * ndim * nonzero_count, pool_));

    ARROW_ASSIGN_OR_RAISE(auto values_buffer,
                          AllocateBuffer(value_elsize * nonzero_count, pool_));

    auto* values = reinterpret_cast<ValueCtype*>(values_buffer->mutable_data());
    const auto* tensor_data = reinterpret_cast<const ValueCtype*>(tensor_.raw_data());
    auto* indices = reinterpret_cast<IndexCtype*>(indices_buffer->mutable_data());
    if (ndim <= 1) {
      const int64_t count = ndim == 0 ? 1 : tensor_.shape()[0];
      for (int64_t i = 0; i < count; ++i) {
        if (is_not_zero<ValueDataType>(*tensor_data)) {
          *indices++ = static_cast<IndexCtype>(i);
          *values++ = *tensor_data;
        }
        ++tensor_data;
      }
    } else if (tensor_.is_row_major()) {
      ConvertRowMajorTensor<ValueDataType>(tensor_, indices, values, nonzero_count);
    } else if (tensor_.is_column_major()) {
      ConvertColumnMajorTensor<ValueDataType>(tensor_, indices, values, nonzero_count);
    } else {
      ConvertStridedTensor<ValueDataType>(tensor_, indices, values, nonzero_count);
    }

    // make results
    const std::vector<int64_t> indices_shape = {nonzero_count, ndim};
    std::vector<int64_t> indices_strides;
    RETURN_NOT_OK(internal::ComputeRowMajorStrides(
        checked_cast<const FixedWidthType&>(*index_value_type_), indices_shape,
        &indices_strides));
    auto coords = std::make_shared<Tensor>(index_value_type_, std::move(indices_buffer),
                                           indices_shape, indices_strides);
    ARROW_ASSIGN_OR_RAISE(sparse_index, SparseCOOIndex::Make(coords, true));
    data = std::move(values_buffer);

    return Status::OK();
  }

  std::shared_ptr<SparseCOOIndex> sparse_index;
  std::shared_ptr<Buffer> data;

 private:
  const Tensor& tensor_;
  const std::shared_ptr<DataType>& index_value_type_;
  MemoryPool* pool_;
};

}  // namespace

int64_t SparseTensorConverterMixin::GetIndexValue(const uint8_t* value_ptr,
                                                  const int elsize) {
  switch (elsize) {
    case 1:
      return *value_ptr;

    case 2:
      return *reinterpret_cast<const uint16_t*>(value_ptr);

    case 4:
      return *reinterpret_cast<const uint32_t*>(value_ptr);

    case 8:
      return *reinterpret_cast<const int64_t*>(value_ptr);

    default:
      return 0;
  }
}

Status MakeSparseCOOTensorFromTensor(const Tensor& tensor,
                                     const std::shared_ptr<DataType>& index_value_type,
                                     MemoryPool* pool,
                                     std::shared_ptr<SparseIndex>* out_sparse_index,
                                     std::shared_ptr<Buffer>* out_data) {
  SparseCOOTensorConverter converter(tensor, index_value_type, pool);
  auto visitor =
      [&]<typename ValueDataType, typename IndexCType, typename ValueCType>(
          const ValueDataType& value_data_type, IndexCType index_ctype,
          ValueCType value_c_type) {
         return converter.Convert(value_data_type, index_ctype, value_c_type);
      };

  ARROW_RETURN_NOT_OK(VisitValueAndIndexType(tensor.type(), index_value_type, visitor));

  *out_sparse_index = checked_pointer_cast<SparseIndex>(converter.sparse_index);
  *out_data = converter.data;
  return Status::OK();
}

Result<std::shared_ptr<Tensor>> MakeTensorFromSparseCOOTensor(
    MemoryPool* pool, const SparseCOOTensor* sparse_tensor) {
  const auto& sparse_index =
      checked_cast<const SparseCOOIndex&>(*sparse_tensor->sparse_index());

  const auto& value_type = checked_cast<const FixedWidthType&>(*sparse_tensor->type());
  const int value_elsize = value_type.byte_width();
  ARROW_ASSIGN_OR_RAISE(auto values_buffer,
                        AllocateBuffer(value_elsize * sparse_tensor->size(), pool));
  auto raw_dense_tensor_values = values_buffer->mutable_data();
  std::memset(raw_dense_tensor_values, 0, value_elsize * sparse_tensor->size());

  std::vector<int64_t> strides;
  RETURN_NOT_OK(ComputeRowMajorStrides(value_type, sparse_tensor->shape(), &strides));

  auto copy_data_from_sparse_tensor_to_dense_tensor =
      [&]<typename DataType, typename IndexCType, typename ValueCType>(
          const DataType&, IndexCType, ValueCType) {
        bool is_indices_row_major = sparse_index.indices()->is_row_major();
        const auto* sparse_tensor_data =
            sparse_tensor->data()->data_as<const ValueCType>();
        const auto* coords_data = sparse_index.indices()->data()->data_as<IndexCType>();
        const int ndim = sparse_tensor->ndim();
        auto non_zero_count = sparse_tensor->non_zero_length();
        for (int64_t i = 0; i < non_zero_count; ++i) {
          int64_t offset = 0;
          if (is_indices_row_major) {
            for (int j = 0; j < ndim; ++j) {
              auto index = *coords_data++;
              offset += index * strides[j];
            }
          } else {
            for (int j = 0; j < ndim; ++j) {
              auto index = *(coords_data + j * non_zero_count);
              offset += index * strides[j];
            }
            ++coords_data;
          }
          *reinterpret_cast<ValueCType*>(raw_dense_tensor_values + offset) =
              *sparse_tensor_data++;
        }
        return Status::OK();
      };

  DCHECK_OK(VisitValueAndIndexType(sparse_tensor->type(), sparse_index.indices()->type(),
                                   copy_data_from_sparse_tensor_to_dense_tensor));
  return std::make_shared<Tensor>(sparse_tensor->type(), std::move(values_buffer),
                                  sparse_tensor->shape(), strides,
                                  sparse_tensor->dim_names());
}

}  // namespace internal
}  // namespace arrow
