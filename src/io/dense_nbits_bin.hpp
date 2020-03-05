/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_IO_DENSE_NBITS_BIN_HPP_
#define LIGHTGBM_IO_DENSE_NBITS_BIN_HPP_

#include <LightGBM/bin.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace LightGBM {

class Dense4bitsBin;

class Dense4bitsBinIterator : public BinIterator {
 public:
  explicit Dense4bitsBinIterator(const Dense4bitsBin* bin_data, uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin)
    : bin_data_(bin_data), min_bin_(static_cast<uint8_t>(min_bin)),
    max_bin_(static_cast<uint8_t>(max_bin)),
    most_freq_bin_(static_cast<uint8_t>(most_freq_bin)) {
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
  const Dense4bitsBin* bin_data_;
  uint8_t min_bin_;
  uint8_t max_bin_;
  uint8_t most_freq_bin_;
  uint8_t offset_;
};

class Dense4bitsBin : public Bin {
 public:
  friend Dense4bitsBinIterator;
  explicit Dense4bitsBin(data_size_t num_data)
    : num_data_(num_data) {
    int len = (num_data_ + 1) / 2;
    data_.resize(len, static_cast<uint8_t>(0));
    buf_ = std::vector<uint8_t>(len, static_cast<uint8_t>(0));
  }

  ~Dense4bitsBin() {
  }

  void Push(int, data_size_t idx, uint32_t value) override {
    const int i1 = idx >> 1;
    const int i2 = (idx & 1) << 2;
    const uint8_t val = static_cast<uint8_t>(value) << i2;
    if (i2 == 0) {
      data_[i1] = val;
    } else {
      buf_[i1] = val;
    }
  }

  void ReSize(data_size_t num_data) override {
    if (num_data_ != num_data) {
      num_data_ = num_data;
      const int len = (num_data_ + 1) / 2;
      data_.resize(len);
    }
  }

  inline BinIterator* GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const override;

  template<bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN>
  void ConstructHistogramInner(const data_size_t* data_indices, data_size_t start, data_size_t end,
    const score_t* ordered_gradients, const score_t* ordered_hessians, hist_t* out) const {
    data_size_t i = start;
    hist_t* grad = out;
    hist_t* hess = out + 1;
    hist_cnt_t* cnt = reinterpret_cast<hist_cnt_t*>(hess);
    if (USE_PREFETCH) {
      const data_size_t pf_offset = 64;
      const data_size_t pf_end = end - pf_offset;
      for (; i < pf_end; ++i) {
        const auto idx = USE_INDICES ? data_indices[i] : i;
        const auto pf_idx = USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
        PREFETCH_T0(data_.data() + (pf_idx >> 1));
        const uint8_t bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
        const uint8_t ti = static_cast<uint8_t>(bin) << 1;
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
      const uint8_t bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
      const uint8_t ti = static_cast<uint8_t>(bin) << 1;
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
              const uint32_t bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
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
    SplitHelper<uint8_t> helper;
    helper.min_bin = static_cast<uint8_t>(min_bin);
    helper.max_bin = static_cast<uint8_t>(max_bin);
    bool most_freq_bin_is_zero_bin = false;
    bool most_freq_bin_is_na = false;
    helper.offset = 0;
    if (most_freq_bin == 0) {
      helper.default_bin = static_cast<uint8_t>(min_bin - 1 + default_bin);
      helper.most_freq_bin = static_cast<uint8_t>(min_bin - 1 + most_freq_bin);
      helper.offset = 1;
    } else {
      helper.default_bin = static_cast<uint8_t>(min_bin + default_bin);
      helper.most_freq_bin = static_cast<uint8_t>(min_bin + most_freq_bin);
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

  template <MissingType TYPE, bool most_freq_bin_is_zero_bin,
            bool most_freq_bin_is_na>
  data_size_t Split(const SplitHelper<uint8_t>& helper, bool default_left,
                    uint32_t threshold, const data_size_t* data_indices,
                    data_size_t num_data, data_size_t* lte_indices,
                    data_size_t* gt_indices) const {
    uint8_t th =
        static_cast<uint8_t>(threshold + helper.min_bin - helper.offset);
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
      const uint8_t bin = (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
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


  void FinishLoad() override {
    if (buf_.empty()) { return; }
    int len = (num_data_ + 1) / 2;
    for (int i = 0; i < len; ++i) {
      data_[i] |= buf_[i];
    }
    buf_.clear();
  }

  void LoadFromMemory(const void* memory, const std::vector<data_size_t>& local_used_indices) override {
    const uint8_t* mem_data = reinterpret_cast<const uint8_t*>(memory);
    if (!local_used_indices.empty()) {
      const data_size_t rest = num_data_ & 1;
      for (int i = 0; i < num_data_ - rest; i += 2) {
        // get old bins
        data_size_t idx = local_used_indices[i];
        const auto bin1 = static_cast<uint8_t>((mem_data[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
        idx = local_used_indices[i + 1];
        const auto bin2 = static_cast<uint8_t>((mem_data[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
        // add
        const int i1 = i >> 1;
        data_[i1] = (bin1 | (bin2 << 4));
      }
      if (rest) {
        data_size_t idx = local_used_indices[num_data_ - 1];
        data_[num_data_ >> 1] = (mem_data[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
      }
    } else {
      for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] = mem_data[i];
      }
    }
  }

  void CopySubrow(const Bin* full_bin, const data_size_t* used_indices, data_size_t num_used_indices) override {
    auto other_bin = dynamic_cast<const Dense4bitsBin*>(full_bin);
    const data_size_t rest = num_used_indices & 1;
    for (int i = 0; i < num_used_indices - rest; i += 2) {
      data_size_t idx = used_indices[i];
      const auto bin1 = static_cast<uint8_t>((other_bin->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
      idx = used_indices[i + 1];
      const auto bin2 = static_cast<uint8_t>((other_bin->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf);
      const int i1 = i >> 1;
      data_[i1] = (bin1 | (bin2 << 4));
    }
    if (rest) {
      data_size_t idx = used_indices[num_used_indices - 1];
      data_[num_used_indices >> 1] = (other_bin->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
    }
  }

  void SaveBinaryToFile(const VirtualFileWriter* writer) const override {
    writer->Write(data_.data(), sizeof(uint8_t) * data_.size());
  }

  size_t SizesInByte() const override {
    return sizeof(uint8_t) * data_.size();
  }

  Dense4bitsBin* Clone() override {
    return new Dense4bitsBin(*this);
  }

 protected:
  Dense4bitsBin(const Dense4bitsBin& other)
    : num_data_(other.num_data_), data_(other.data_), buf_(other.buf_) {
  }

  data_size_t num_data_;
  std::vector<uint8_t, Common::AlignmentAllocator<uint8_t, kAlignedSize>> data_;
  std::vector<uint8_t> buf_;
};

uint32_t Dense4bitsBinIterator::Get(data_size_t idx) {
  const auto bin = (bin_data_->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
  if (bin >= min_bin_ && bin <= max_bin_) {
    return bin - min_bin_ + offset_;
  } else {
    return most_freq_bin_;
  }
}

uint32_t Dense4bitsBinIterator::RawGet(data_size_t idx) {
  return (bin_data_->data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
}

inline BinIterator* Dense4bitsBin::GetIterator(uint32_t min_bin, uint32_t max_bin, uint32_t most_freq_bin) const {
  return new Dense4bitsBinIterator(this, min_bin, max_bin, most_freq_bin);
}

}  // namespace LightGBM
#endif   // LIGHTGBM_IO_DENSE_NBITS_BIN_HPP_
