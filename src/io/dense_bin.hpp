/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_DENSE_BIN_HPP_
#define LIGHTGBM_IO_DENSE_BIN_HPP_

#include <LightGBM/bin.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace LightGBM {

template <typename VAL_T>
class DenseBin;

template <typename VAL_T>
class DenseBinIterator: public BinIterator {
 public:
  explicit DenseBinIterator(const DenseBin<VAL_T>* bin_data, uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin)
    : bin_data_(bin_data), min_bin_(static_cast<VAL_T>(min_bin)),
    max_bin_(static_cast<VAL_T>(max_bin)),
    most_freq_bin_(static_cast<VAL_T>(most_freq_bin)) {
    if (most_freq_bin_ == 0) {
      offset_ = 1;
    } else {
      offset_ = 0;
    }
  }
  inline uint32_t RawGet(data_size_t idx) override;
  inline uint32_t Get(data_size_t idx) override;
  inline void Reset(data_size_t) override {}

 private:
  const DenseBin<VAL_T>* bin_data_;
  VAL_T min_bin_;
  VAL_T max_bin_;
  VAL_T most_freq_bin_;
  uint8_t offset_;
};
/*!
* \brief Used to store bins for dense feature
* Use template to reduce memory cost
*/
template <typename VAL_T>
class DenseBin: public Bin {
 public:
  friend DenseBinIterator<VAL_T>;
  explicit DenseBin(data_size_t num_data)
    : num_data_(num_data), data_(num_data_, static_cast<VAL_T>(0)) {
  }

  ~DenseBin() {
  }

  void Push(int, data_size_t idx, uint32_t value) override {
    data_[idx] = static_cast<VAL_T>(value);
  }

  void ReSize(data_size_t num_data) override {
    if (num_data_ != num_data) {
      num_data_ = num_data;
      data_.resize(num_data_);
    }
  }

  BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override;

  template<bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN>
  void ConstructHistogramInner(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians, hist_t* out) const {
    data_size_t i = start;
    hist_t* grad = out;
    hist_t* hess = out + 1;
    hist_cnt_t* cnt = reinterpret_cast<hist_cnt_t*>(hess);
    if (USE_PREFETCH) {
      const data_size_t pf_offset = 64 / sizeof(VAL_T);
      const data_size_t pf_end = end - pf_offset;
      for (; i < pf_end; ++i) {
        const auto idx = USE_INDICES ? data_indices[i] : i;
        const auto pf_idx = USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
        PREFETCH_T0(data_.data() + pf_idx);
        const auto ti = static_cast<uint32_t>(data_[idx]) << 1;
        if (USE_HESSIAN) {
          grad[ti] += ordered_gradients[i];
          hess[ti] += ordered_hessians[i];
        } else {
          grad[ti] += ordered_gradients[i];
          ++cnt[ti];
        }
      }
    }
    for (; i < end; ++i) {
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const auto ti = static_cast<uint32_t>(data_[idx]) << 1;
      if (USE_HESSIAN) {
        grad[ti] += ordered_gradients[i];
        hess[ti] += ordered_hessians[i];
      } else {
        grad[ti] += ordered_gradients[i];
        ++cnt[ti];
      }
    }
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override {
    ConstructHistogramInner<true, true, true>(data_indices, start, end, ordered_gradients, ordered_hessians, out);
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians,
    hist_t* out) const override {
    ConstructHistogramInner<false, false, true>(nullptr, start, end, ordered_gradients, ordered_hessians, out);
  }

  void ConstructHistogram(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients,
    hist_t* out) const override {
    ConstructHistogramInner<true, true, false>(data_indices, start, end, ordered_gradients, nullptr, out);
  }

  void ConstructHistogram(data_size_t start, data_size_t end,
    const score_t* ordered_gradients,
    hist_t* out) const override {
    ConstructHistogramInner<false, false, false>(nullptr, start, end, ordered_gradients, nullptr, out);
  }

  void InitSplit(int num_features, int cur_feature, BinType bin_type,
                 uint32_t min_bin, uint32_t max_bin, uint32_t default_bin,
                 uint32_t most_freq_bin,
                 MissingType missing_type) const override {
    if (split_funcs_.size() < static_cast<size_t>(num_features)) {
      split_funcs_.resize(num_features);
    }
    if (bin_type == BinType::CategoricalBin) {
      split_funcs_[cur_feature] =
          [=](const uint32_t* threshold, int num_threshold, bool,
              const data_size_t* data_indices, data_size_t num_data,
              data_size_t* lte_indices, data_size_t* gt_indices) {
            data_size_t lte_count = 0;
            data_size_t gt_count = 0;
            data_size_t* default_indices = gt_indices;
            data_size_t* default_count = &gt_count;
            if (Common::FindInBitset(threshold, num_threshold, most_freq_bin)) {
              default_indices = lte_indices;
              default_count = &lte_count;
            }
            for (data_size_t i = 0; i < num_data; ++i) {
              const data_size_t idx = data_indices[i];
              const uint32_t bin = data_[idx];
              if (bin < min_bin || bin > max_bin) {
                default_indices[(*default_count)++] = idx;
              } else if (Common::FindInBitset(threshold, num_threshold,
                                              bin - min_bin)) {
                lte_indices[lte_count++] = idx;
              } else {
                gt_indices[gt_count++] = idx;
              }
            }
            return lte_count;
          };
    }
    SplitHelper<VAL_T> helper;
    helper.min_bin = static_cast<VAL_T>(min_bin);
    helper.max_bin = static_cast<VAL_T>(max_bin);
    bool most_freq_bin_is_zero_bin = false;
    bool most_freq_bin_is_na = false;
    helper.offset = 0;
    if (most_freq_bin == 0) {
      helper.default_bin = static_cast<VAL_T>(min_bin - 1 + default_bin);
      helper.most_freq_bin = static_cast<VAL_T>(min_bin - 1 + most_freq_bin);
      helper.offset = 1;
    } else {
      helper.default_bin = static_cast<VAL_T>(min_bin + default_bin);
      helper.most_freq_bin = static_cast<VAL_T>(min_bin + most_freq_bin);
    }
    if (missing_type == MissingType::NaN &&
        helper.most_freq_bin == helper.max_bin) {
      most_freq_bin_is_na = true;
    }
    if (most_freq_bin == default_bin) {
      most_freq_bin_is_zero_bin = true;
    }
    if (missing_type == MissingType::None) {
      split_funcs_[cur_feature] =
          [=](const uint32_t* threshold, int, bool default_left,
              const data_size_t* data_indices, data_size_t num_data,
              data_size_t* lte_indices, data_size_t* gt_indices) {
            return this->Split<MissingType::None, false, false>(
                helper, default_left, *threshold, data_indices, num_data,
                lte_indices, gt_indices);
          };
    } else if (missing_type == MissingType::Zero) {
      if (most_freq_bin_is_zero_bin) {
        split_funcs_[cur_feature] =
            [=](const uint32_t* threshold, int, bool default_left,
                const data_size_t* data_indices, data_size_t num_data,
                data_size_t* lte_indices, data_size_t* gt_indices) {
              return this->Split<MissingType::Zero, true, false>(
                  helper, default_left, *threshold, data_indices, num_data,
                  lte_indices, gt_indices);
            };
      } else {
        split_funcs_[cur_feature] =
            [=](const uint32_t* threshold, int, bool default_left,
                const data_size_t* data_indices, data_size_t num_data,
                data_size_t* lte_indices, data_size_t* gt_indices) {
              return this->Split<MissingType::Zero, false, false>(
                  helper, default_left, *threshold, data_indices, num_data,
                  lte_indices, gt_indices);
            };
      }

    } else if (missing_type == MissingType::NaN) {
      if (most_freq_bin_is_na) {
        split_funcs_[cur_feature] =
            [=](const uint32_t* threshold, int, bool default_left,
                const data_size_t* data_indices, data_size_t num_data,
                data_size_t* lte_indices, data_size_t* gt_indices) {
              return this->Split<MissingType::NaN, false, true>(
                  helper, default_left, *threshold, data_indices, num_data,
                  lte_indices, gt_indices);
            };
      } else {
        split_funcs_[cur_feature] =
            [=](const uint32_t* threshold, int, bool default_left,
                const data_size_t* data_indices, data_size_t num_data,
                data_size_t* lte_indices, data_size_t* gt_indices) {
              return this->Split<MissingType::NaN, false, false>(
                  helper, default_left, *threshold, data_indices, num_data,
                  lte_indices, gt_indices);
            };
      }
    }
  }

  template <MissingType TYPE, bool most_freq_bin_is_zero_bin, bool most_freq_bin_is_na>
  data_size_t Split(const SplitHelper<VAL_T>& helper, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t num_data, data_size_t* lte_indices,
                    data_size_t* gt_indices) const {
    VAL_T th = static_cast<VAL_T>(threshold + helper.min_bin - helper.offset);
    data_size_t lte_count = 0;
    data_size_t gt_count = 0;
    data_size_t* default_indices = gt_indices;
    data_size_t* default_count = &gt_count;
    data_size_t* missing_default_indices = gt_indices;
    data_size_t* missing_default_count = &gt_count;
    if (helper.most_freq_bin <= th) {
      default_indices = lte_indices;
      default_count = &lte_count;
    }
    if (TYPE != MissingType::None) {
      if (default_left) {
        missing_default_indices = lte_indices;
        missing_default_count = &lte_count;
      }
    }
    for (data_size_t i = 0; i < num_data; ++i) {
      const data_size_t idx = data_indices[i];
      const VAL_T bin = data_[idx];
      if (TYPE == MissingType::Zero && !most_freq_bin_is_zero_bin) {
        if (bin == helper.default_bin) {
          missing_default_indices[(*missing_default_count)++] = idx;
          continue;
        }
      }
      if (helper.most_freq_bin == bin || bin < helper.min_bin ||
          bin > helper.max_bin) {
        if ((TYPE == MissingType::NaN && most_freq_bin_is_na) ||
            (TYPE == MissingType::Zero && most_freq_bin_is_zero_bin)) {
          missing_default_indices[(*missing_default_count)++] = idx;
        } else {
          default_indices[(*default_count)++] = idx;
        }
      } else if (bin > th) {
        gt_indices[gt_count++] = idx;
      } else {
        lte_indices[lte_count++] = idx;
      }
    }
    return lte_count;
  }

  data_size_t num_data() const override { return num_data_; }

  void FinishLoad() override {}

  void LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices) override {
    const VAL_T* mem_data = reinterpret_cast<const VAL_T*>(memory);
    if (!local_used_indices.empty()) {
      for (int i = 0; i < num_data_; ++i) {
        data_[i] = mem_data[local_used_indices[i]];
      }
    } else {
      for (int i = 0; i < num_data_; ++i) {
        data_[i] = mem_data[i];
      }
    }
  }

  void CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) override {
    auto other_bin = dynamic_cast<const DenseBin<VAL_T>*>(full_bin);
    for (int i = 0; i < num_used_indices; ++i) {
      data_[i] = other_bin->data_[used_indices[i]];
    }
  }

  void SaveBinaryToFile(const VirtualFileWriter* writer) const override {
    writer->Write(data_.data(), sizeof(VAL_T) * num_data_);
  }

  size_t SizesInByte() const override {
    return sizeof(VAL_T) * num_data_;
  }

  DenseBin<VAL_T>* Clone() override;

 private:
  data_size_t num_data_;
  std::vector<VAL_T, Common::AlignmentAllocator<VAL_T, kAlignedSize>> data_;

  DenseBin<VAL_T>(const DenseBin<VAL_T>& other)
    : num_data_(other.num_data_), data_(other.data_) {
  }
};

template<typename VAL_T>
DenseBin<VAL_T>* DenseBin<VAL_T>::Clone() {
  return new DenseBin<VAL_T>(*this);
}

template <typename VAL_T>
uint32_t DenseBinIterator<VAL_T>::Get(data_size_t idx) {
  auto ret = bin_data_->data_[idx];
  if (ret >= min_bin_ && ret <= max_bin_) {
    return ret - min_bin_ + offset_;
  } else {
    return most_freq_bin_;
  }
}

template <typename VAL_T>
inline uint32_t DenseBinIterator<VAL_T>::RawGet(data_size_t idx) {
  return bin_data_->data_[idx];
}

template <typename VAL_T>
BinIterator* DenseBin<VAL_T>::GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const {
  return new DenseBinIterator<VAL_T>(this, min_bin, max_bin, most_freq_bin);
}

}  // namespace LightGBM
#endif   // LightGBM_IO_DENSE_BIN_HPP_
