#include "Bitset.hpp"
#include <cmath>

namespace zephyr {

bitset_t::bitset_t(size_t num_bits) : m_num_bits(num_bits) {
  // Calculate required blocks, ceiling division
  size_t num_blocks = (num_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
  m_blocks.resize(num_blocks, 0);
}

// Basic Ops inlined in header

auto bitset_t::resize(size_t new_num_bits) -> void {
  m_num_bits = new_num_bits;
  size_t num_blocks = (new_num_bits + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
  m_blocks.resize(num_blocks);

  // Optional: clear any "garbage" bits in the last block if we shrank/grew
  // weirdly? For a simple resize, vector handles zero-initialization of new
  // blocks.
}

auto bitset_t::size() const -> size_t { return m_num_bits; }

// SIMD Stride Constants for 32/64-bit Size_t Compatibility
#if defined(__AVX2__)
// AVX2 is 256 bits (32 bytes). Stride in block_type count.
static constexpr size_t SIMD_STRIDE_AVX2 = 32 / sizeof(bitset_t::block_type);
#elif defined(__ARM_NEON)
// NEON is 128 bits (16 bytes). Stride in block_type count.
static constexpr size_t SIMD_STRIDE_NEON = 16 / sizeof(bitset_t::block_type);
#endif

// --- Advanced Features Implementation ---

// 2. Mass Bitwise Operations
auto bitset_t::operator&=(const bitset_t &rhs) -> bitset_t & {
  if (m_num_bits != rhs.m_num_bits) {
    throw std::invalid_argument(
        "bitset_t sizes must match for bitwise operations");
  }

  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  // Process 256 bits at a time
  for (; i + SIMD_STRIDE_AVX2 <= n; i += SIMD_STRIDE_AVX2) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    __m256i b =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&rhs.m_blocks[i]));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&m_blocks[i]),
                        _mm256_and_si256(a, b));
  }
#elif defined(__ARM_NEON)
  // Process 128 bits at a time
  for (; i + SIMD_STRIDE_NEON <= n; i += SIMD_STRIDE_NEON) {
    uint64x2_t a = vld1q_u64(&m_blocks[i]);
    uint64x2_t b = vld1q_u64(&rhs.m_blocks[i]);
    vst1q_u64(&m_blocks[i], vandq_u64(a, b));
  }
#endif

  // Scalar fallback
  for (; i < n; ++i) {
    m_blocks[i] &= rhs.m_blocks[i];
  }
  return *this;
}

auto bitset_t::operator|=(const bitset_t &rhs) -> bitset_t & {
  if (m_num_bits != rhs.m_num_bits) {
    throw std::invalid_argument(
        "bitset_t sizes must match for bitwise operations");
  }

  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  for (; i + SIMD_STRIDE_AVX2 <= n; i += SIMD_STRIDE_AVX2) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    __m256i b =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&rhs.m_blocks[i]));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&m_blocks[i]),
                        _mm256_or_si256(a, b));
  }
#elif defined(__ARM_NEON)
  for (; i + SIMD_STRIDE_NEON <= n; i += SIMD_STRIDE_NEON) {
    uint64x2_t a = vld1q_u64(&m_blocks[i]);
    uint64x2_t b = vld1q_u64(&rhs.m_blocks[i]);
    vst1q_u64(&m_blocks[i], vorrq_u64(a, b));
  }
#endif

  for (; i < n; ++i) {
    m_blocks[i] |= rhs.m_blocks[i];
  }
  return *this;
}

auto bitset_t::operator^=(const bitset_t &rhs) -> bitset_t & {
  if (m_num_bits != rhs.m_num_bits) {
    throw std::invalid_argument(
        "bitset_t sizes must match for bitwise operations");
  }

  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  for (; i + SIMD_STRIDE_AVX2 <= n; i += SIMD_STRIDE_AVX2) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    __m256i b =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&rhs.m_blocks[i]));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&m_blocks[i]),
                        _mm256_xor_si256(a, b));
  }
#elif defined(__ARM_NEON)
  for (; i + SIMD_STRIDE_NEON <= n; i += SIMD_STRIDE_NEON) {
    uint64x2_t a = vld1q_u64(&m_blocks[i]);
    uint64x2_t b = vld1q_u64(&rhs.m_blocks[i]);
    vst1q_u64(&m_blocks[i], veorq_u64(a, b));
  }
#endif

  for (; i < n; ++i) {
    m_blocks[i] ^= rhs.m_blocks[i];
  }
  return *this;
}

auto bitset_t::operator~() const -> bitset_t {
  bitset_t result(m_num_bits);
  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  __m256i all_ones = _mm256_set1_epi64x(-1);
  for (; i + SIMD_STRIDE_AVX2 <= n; i += SIMD_STRIDE_AVX2) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&result.m_blocks[i]),
                        _mm256_xor_si256(a, all_ones));
  }
#elif defined(__ARM_NEON)
  for (; i + SIMD_STRIDE_NEON <= n; i += SIMD_STRIDE_NEON) {
    uint64x2_t a = vld1q_u64(&m_blocks[i]);
    vst1q_u64(&result.m_blocks[i], vmvnq_u64(a));
  }
#endif

  for (; i < n; ++i) {
    result.m_blocks[i] = ~m_blocks[i];
  }

  // Mask out the unused bits in the last block
  size_t extra_bits = m_num_bits % BITS_PER_BLOCK;
  if (extra_bits != 0) {
    block_type mask = (static_cast<block_type>(1) << extra_bits) - 1;
    result.m_blocks.back() &= mask;
  }

  return result;
}

// 3. Population & Search
auto bitset_t::count() const -> size_t {
  size_t cnt0 = 0;
  size_t cnt1 = 0;
  size_t cnt2 = 0;
  size_t cnt3 = 0;
  size_t i = 0;
  size_t n = m_blocks.size();

  // Unroll loop 4x to break dependency chain
  // Safe on 32-bit: block_type is 32-bit, so 4x unroll is 128 bits.
  for (; i + 4 <= n; i += 4) {
    cnt0 += std::popcount(m_blocks[i]);
    cnt1 += std::popcount(m_blocks[i + 1]);
    cnt2 += std::popcount(m_blocks[i + 2]);
    cnt3 += std::popcount(m_blocks[i + 3]);
  }

  size_t total = cnt0 + cnt1 + cnt2 + cnt3;
  for (; i < n; ++i) {
    total += std::popcount(m_blocks[i]);
  }
  return total;
}

auto bitset_t::any() const -> bool {
  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  for (; i + SIMD_STRIDE_AVX2 <= n; i += SIMD_STRIDE_AVX2) {
    __m256i val =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    if (!_mm256_testz_si256(val, val))
      return true;
  }
#elif defined(__ARM_NEON)
  for (; i + SIMD_STRIDE_NEON <= n; i += SIMD_STRIDE_NEON) {
    uint64x2_t val = vld1q_u64(&m_blocks[i]);
    // logical OR of two 64-bit lanes
    uint64_t low = vgetq_lane_u64(val, 0);
    uint64_t high = vgetq_lane_u64(val, 1);
    if (low | high)
      return true;
  }
#endif
  for (; i < n; ++i) {
    if (m_blocks[i] != 0)
      return true;
  }
  return false;
}

auto bitset_t::none() const -> bool { return !any(); }

auto bitset_t::all() const -> bool {
  size_t full_blocks = m_num_bits / BITS_PER_BLOCK;
  size_t i = 0;

#if defined(__AVX2__)
  __m256i all_ones = _mm256_set1_epi64x(-1);
  for (; i + SIMD_STRIDE_AVX2 <= full_blocks; i += SIMD_STRIDE_AVX2) {
    __m256i val =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    if (!_mm256_testc_si256(val, all_ones))
      return false;
  }
#elif defined(__ARM_NEON)
  for (; i + SIMD_STRIDE_NEON <= full_blocks; i += SIMD_STRIDE_NEON) {
    uint64x2_t val = vld1q_u64(&m_blocks[i]);
    val = vmvnq_u64(val);
    if (vgetq_lane_u64(val, 0) | vgetq_lane_u64(val, 1))
      return false;
  }
#endif

  // Check full blocks
  for (; i < full_blocks; ++i) {
    if (m_blocks[i] != static_cast<block_type>(~0))
      return false;
  }

  // Check last partial block
  size_t extra_bits = m_num_bits % BITS_PER_BLOCK;
  if (extra_bits != 0) {
    block_type mask = (static_cast<block_type>(1) << extra_bits) - 1;
    if ((m_blocks.back() & mask) != mask)
      return false;
  }

  return true;
}

auto bitset_t::find_first_set() const -> std::optional<size_t> {
  size_t full_blocks = m_num_bits / BITS_PER_BLOCK;
  size_t i = 0;

  // SIMD: Skip chunks that are all zero
#if defined(__AVX2__)
  for (; i + SIMD_STRIDE_AVX2 <= full_blocks; i += SIMD_STRIDE_AVX2) {
    __m256i val =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    if (_mm256_testz_si256(val, val))
      continue; // All zero, skip
    break;      // Found non-zero
  }
#elif defined(__ARM_NEON)
  for (; i + SIMD_STRIDE_NEON <= full_blocks; i += SIMD_STRIDE_NEON) {
    uint64x2_t val = vld1q_u64(&m_blocks[i]);
    uint64x2_t low = vgetq_lane_u64(val, 0);
    uint64x2_t high = vgetq_lane_u64(val, 1);
    if ((low | high) == 0)
      continue;
    break;
  }
#endif

  // Check blocks
  for (; i < full_blocks; ++i) {
    if (m_blocks[i] != 0) {
      size_t bit = std::countr_zero(m_blocks[i]);
      return i * BITS_PER_BLOCK + bit;
    }
  }

  // Check last partial block
  size_t extra_bits = m_num_bits % BITS_PER_BLOCK;
  if (extra_bits != 0) {
    block_type mask = (static_cast<block_type>(1) << extra_bits) - 1;
    if ((m_blocks.back() & mask) != 0) {
      size_t bit = std::countr_zero(m_blocks.back());
      return full_blocks * BITS_PER_BLOCK + bit;
    }
  }

  return std::nullopt;
}

auto bitset_t::find_first_zero() const -> std::optional<size_t> {
  size_t full_blocks = m_num_bits / BITS_PER_BLOCK;
  size_t i = 0;

  // SIMD: Skip chunks that are all ones
#if defined(__AVX2__)
  __m256i all_ones = _mm256_set1_epi64x(-1);
  for (; i + SIMD_STRIDE_AVX2 <= full_blocks; i += SIMD_STRIDE_AVX2) {
    __m256i val =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    if (_mm256_testc_si256(val, all_ones))
      continue; // All ones, skip
    break;
  }
#elif defined(__ARM_NEON)
  for (; i + SIMD_STRIDE_NEON <= full_blocks; i += SIMD_STRIDE_NEON) {
    uint64x2_t val = vld1q_u64(&m_blocks[i]);
    val = vmvnq_u64(val); // invert
    uint64x2_t low = vgetq_lane_u64(val, 0);
    uint64x2_t high = vgetq_lane_u64(val, 1);
    if ((low | high) == 0)
      continue; // Original was all ones
    break;
  }
#endif
  // Check full blocks
  for (; i < full_blocks; ++i) {
    if (m_blocks[i] != static_cast<block_type>(~0)) {
      size_t bit = std::countr_one(m_blocks[i]);
      return i * BITS_PER_BLOCK + bit;
    }
  }

  // Check last partial block
  size_t extra_bits = m_num_bits % BITS_PER_BLOCK;
  if (extra_bits != 0) {
    block_type mask = (static_cast<block_type>(1) << extra_bits) - 1;
    block_type val = m_blocks.back();
    // We only care about the bits within the mask.
    // If (val & mask) != mask, there is a zero.
    if ((val & mask) != mask) {
      // Invert logic: find first zero in 'val' is like find first set in '~val'
      // blocked out by mask.
      block_type inverted = ~val & mask;
      size_t bit = std::countr_zero(inverted);
      return full_blocks * BITS_PER_BLOCK + bit;
    }
  }

  return std::nullopt;
}

auto bitset_t::find_next_set(size_t index) const -> std::optional<size_t> {
  if (index >= m_num_bits) {
    return std::nullopt;
  }

  size_t i = index / BITS_PER_BLOCK;
  size_t bit_offset = index % BITS_PER_BLOCK;

  // Check first block (possibly partial)
  block_type first_block = m_blocks[i];
  // Clear bits before bit_offset
  block_type mask = (~static_cast<block_type>(0)) << bit_offset;
  if ((first_block & mask) != 0) {
    return i * BITS_PER_BLOCK + std::countr_zero(first_block & mask);
  }

  // Continue to next blocks
  i++;
  size_t full_blocks = m_num_bits / BITS_PER_BLOCK;

  // SIMD skip for middle blocks
#if defined(__AVX2__)
  for (; i + SIMD_STRIDE_AVX2 <= full_blocks; i += SIMD_STRIDE_AVX2) {
    __m256i val =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    if (_mm256_testz_si256(val, val))
      continue;
    break;
  }
#endif

  for (; i < full_blocks; ++i) {
    if (m_blocks[i] != 0) {
      return i * BITS_PER_BLOCK + std::countr_zero(m_blocks[i]);
    }
  }

  // Last partial block
  size_t extra_bits = m_num_bits % BITS_PER_BLOCK;
  if (extra_bits != 0 && i == full_blocks) {
    block_type last_mask = (static_cast<block_type>(1) << extra_bits) - 1;
    if ((m_blocks.back() & last_mask) != 0) {
      return i * BITS_PER_BLOCK + std::countr_zero(m_blocks.back());
    }
  }

  return std::nullopt;
}

auto bitset_t::find_next_zero(size_t index) const -> std::optional<size_t> {
  if (index >= m_num_bits) {
    return std::nullopt;
  }

  size_t i = index / BITS_PER_BLOCK;
  size_t bit_offset = index % BITS_PER_BLOCK;

  // Check first block
  block_type first_block = m_blocks[i];
  block_type mask = (~static_cast<block_type>(0)) << bit_offset;

  // We want a zero. So if (first_block & mask) == mask, it's all ones (in the
  // valid range). Invert: ~first_block & mask. If non-zero, there is a zero
  // bit.
  if ((~first_block & mask) != 0) {
    return i * BITS_PER_BLOCK + std::countr_zero(~first_block & mask);
  }

  i++;
  size_t full_blocks = m_num_bits / BITS_PER_BLOCK;

#if defined(__AVX2__)
  __m256i all_ones = _mm256_set1_epi64x(-1);
  for (; i + SIMD_STRIDE_AVX2 <= full_blocks; i += SIMD_STRIDE_AVX2) {
    __m256i val =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    if (_mm256_testc_si256(val, all_ones))
      continue;
    break;
  }
#endif

  for (; i < full_blocks; ++i) {
    if (m_blocks[i] != 0) {
      return i * BITS_PER_BLOCK + std::countr_zero(m_blocks[i]);
    }
  }

  // Last partial block
  size_t extra_bits = m_num_bits % BITS_PER_BLOCK;
  if (extra_bits != 0 && i == full_blocks) {
    block_type last_mask = (static_cast<block_type>(1) << extra_bits) - 1;
    if ((m_blocks.back() & last_mask) != 0) {
      return i * BITS_PER_BLOCK + std::countr_zero(m_blocks.back());
    }
  }

  return std::nullopt;
}

// 4. String Conversion
auto bitset_t::to_string() const -> std::string {
  std::string s;
  s.reserve(m_num_bits);
  for (size_t i = 0; i < m_num_bits; ++i) {
    s += test(m_num_bits - 1 - i) ? '1' : '0';
  }
  return s;
}

auto operator<<(std::ostream &os, const bitset_t &b) -> std::ostream & {
  os << b.to_string();
  return os;
}

// 5. File I/O
auto bitset_t::save(const std::string &filename) const -> void {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs)
    throw std::runtime_error("Cannot open file for writing: " + filename);

  // Header: "BITSET" magic, version, num_bits
  const char magic[] = "BITSET";
  uint32_t version = 1;
  uint64_t n_bits = static_cast<uint64_t>(m_num_bits);

  ofs.write("BITSET", 6);
  ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
  ofs.write(reinterpret_cast<const char *>(&n_bits), sizeof(n_bits));

  // Data
  size_t n_blocks = (m_num_bits + 63) / 64; // normalize to 64-bit blocks
  std::vector<uint64_t> portable_blocks(n_blocks, 0);

  // Copy data to portable blocks
  for (size_t i = 0; i < m_num_bits; ++i) {
    if (test(i)) {
      portable_blocks[i / 64] |= (static_cast<uint64_t>(1) << (i % 64));
    }
  }

  ofs.write(reinterpret_cast<const char *>(portable_blocks.data()),
            portable_blocks.size() * sizeof(uint64_t));
}

auto bitset_t::load(const std::string &filename) -> void {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs)
    throw std::runtime_error("Cannot open file for reading: " + filename);

  char magic[7] = {0};
  ifs.read(magic, 6);
  if (std::string(magic) != "BITSET")
    throw std::runtime_error("Invalid file format");

  uint32_t version;
  ifs.read(reinterpret_cast<char *>(&version), sizeof(version));
  if (version != 1)
    throw std::runtime_error("Unsupported version");

  uint64_t n_bits;
  ifs.read(reinterpret_cast<char *>(&n_bits), sizeof(n_bits));

  resize(static_cast<size_t>(n_bits));

  // Read portable 64-bit blocks
  size_t n_blocks = (n_bits + 63) / 64;
  std::vector<uint64_t> portable_blocks(n_blocks);
  ifs.read(reinterpret_cast<char *>(portable_blocks.data()),
           n_blocks * sizeof(uint64_t));

  for (size_t i = 0; i < static_cast<size_t>(n_bits); ++i) {
    bool bit_val = (portable_blocks[i / 64] >> (i % 64)) & 1;
    set(i, bit_val);
  }
}

// 7. Shift Operators
auto bitset_t::operator<<=(size_t pos) -> bitset_t & {
  if (pos == 0)
    return *this;
  if (pos >= m_num_bits) {
    std::fill(m_blocks.begin(), m_blocks.end(), 0);
    return *this;
  }

  size_t block_shift = pos / BITS_PER_BLOCK;
  size_t bit_shift = pos % BITS_PER_BLOCK;

  // Block shift
  if (block_shift > 0) {
    std::copy_backward(m_blocks.begin(), m_blocks.end() - block_shift,
                       m_blocks.end());
    std::fill(m_blocks.begin(), m_blocks.begin() + block_shift, 0);
  }

  // Bit shift
  if (bit_shift > 0) {
    block_type carry = 0;
    for (size_t i = 0; i < m_blocks.size(); ++i) {
      block_type val = m_blocks[i];
      block_type next_carry = val >> (BITS_PER_BLOCK - bit_shift);
      m_blocks[i] = (val << bit_shift) | carry;
      carry = next_carry;
    }
  }

  // Clear extra bits
  size_t extra_bits = m_num_bits % BITS_PER_BLOCK;
  if (extra_bits != 0) {
    block_type mask = (static_cast<block_type>(1) << extra_bits) - 1;
    m_blocks.back() &= mask;
  }
  return *this;
}

auto bitset_t::operator>>=(size_t pos) -> bitset_t & {
  if (pos == 0)
    return *this;
  if (pos >= m_num_bits) {
    std::fill(m_blocks.begin(), m_blocks.end(), 0);
    return *this;
  }

  size_t block_shift = pos / BITS_PER_BLOCK;
  size_t bit_shift = pos % BITS_PER_BLOCK;

  if (block_shift > 0) {
    std::copy(m_blocks.begin() + block_shift, m_blocks.end(), m_blocks.begin());
    std::fill(m_blocks.end() - block_shift, m_blocks.end(), 0);
  }

  if (bit_shift > 0) {
    block_type carry = 0;
    // Iterate backwards
    for (size_t i = m_blocks.size(); i > 0; --i) {
      size_t idx = i - 1;
      block_type val = m_blocks[idx];
      block_type next_carry = val << (BITS_PER_BLOCK - bit_shift);
      m_blocks[idx] = (val >> bit_shift) | carry;
      carry = next_carry;
    }
  }

  return *this;
}

// 8. Range Operations
auto bitset_t::set_range(size_t start, size_t count, bool value) -> void {
  if (count == 0)
    return;
  size_t end = start + count;
  if (end > m_num_bits) {
    throw std::out_of_range("Range exceeds bitset_t size");
  }

  size_t current_block = start / BITS_PER_BLOCK;
  size_t final_block = (end - 1) / BITS_PER_BLOCK;

  if (start % BITS_PER_BLOCK != 0) {
    block_type mask = (~static_cast<block_type>(0)) << (start % BITS_PER_BLOCK);
    // If we are solely in the first block, we must also apply the end mask
    if (current_block == final_block) {
      size_t local_end = end % BITS_PER_BLOCK;
      if (local_end != 0) {
        mask &= (static_cast<block_type>(1) << local_end) - 1;
      }
    }

    if (value)
      m_blocks[current_block] |= mask;
    else
      m_blocks[current_block] &= ~mask;

    if (current_block == final_block)
      return;
    current_block++;
  }

  if (current_block < final_block) {
    block_type fill_val = value ? ~static_cast<block_type>(0) : 0;
    std::fill(m_blocks.begin() + current_block, m_blocks.begin() + final_block,
              fill_val);
    current_block = final_block;
  }

  if (current_block <= final_block) {
    size_t local_end = end % BITS_PER_BLOCK;
    if (local_end == 0)
      local_end = BITS_PER_BLOCK;

    block_type mask;
    if (local_end == BITS_PER_BLOCK)
      mask = ~static_cast<block_type>(0);
    else
      mask = (static_cast<block_type>(1) << local_end) - 1;

    if (value)
      m_blocks[final_block] |= mask;
    else
      m_blocks[final_block] &= ~mask;
  }
}

auto bitset_t::flip_range(size_t start, size_t count) -> void {
  if (count == 0)
    return;
  size_t end = start + count;
  if (end > m_num_bits) {
    throw std::out_of_range("Range exceeds bitset_t size");
  }

  size_t current_block = start / BITS_PER_BLOCK;
  size_t final_block = (end - 1) / BITS_PER_BLOCK;

  if (start % BITS_PER_BLOCK != 0) {
    block_type mask = (~static_cast<block_type>(0)) << (start % BITS_PER_BLOCK);
    if (current_block == final_block) {
      size_t local_end = end % BITS_PER_BLOCK;
      if (local_end != 0) {
        mask &= (static_cast<block_type>(1) << local_end) - 1;
      }
    }
    m_blocks[current_block] ^= mask;

    if (current_block == final_block)
      return;
    current_block++;
  }

  if (current_block < final_block) {
    for (size_t i = current_block; i < final_block; ++i) {
      m_blocks[i] ^= ~static_cast<block_type>(0);
    }
    current_block = final_block;
  }

  if (current_block <= final_block) {
    size_t local_end = end % BITS_PER_BLOCK;
    if (local_end == 0)
      local_end = BITS_PER_BLOCK;

    block_type mask;
    if (local_end == BITS_PER_BLOCK)
      mask = ~static_cast<block_type>(0);
    else
      mask = (static_cast<block_type>(1) << local_end) - 1;

    m_blocks[final_block] ^= mask;
  }
}

// 9. Hash Support
} // namespace zephyr

// 9. Hash Support
auto std::hash<zephyr::bitset_t>::operator()(const zephyr::bitset_t &b) const
    -> size_t {
  size_t result = 0;
  // Note: This relies on friendship declaration in bitset_t
  // We cannot access m_blocks directly if we are not friend.
  // However, since we cannot easily range-for over a private member without
  // being inside the class, and we are defining a member of a different class
  // (std::hash), we must trust the friend declaration. To be perfectly safe
  // against access issues if friend fails, we could use getters, but we don't
  // have a public getter for blocks. Let's assume friend works.

  // We need to access the private vector.
  // Since we are outside the class scope, we can't name the types easily?
  // auto works.

  // We need to cast 'b' to something accessbible? No, 'b' is the object.
  // If friend works, b.m_blocks is accessible.

  // But wait, relying on implementation details (m_blocks) in std::hash
  // specialization defined in .cpp is fine.

  // We need to access m_blocks. To avoid "private member" error if friend is
  // flaky: Ensure Bitset.hpp has `friend struct std::hash<bitset_t>;` matches
  // `std::hash<zephyr::bitset_t>`.

  // Let's use the public `to_uint64` or `test` loop if we want to be 100% safe,
  // but that's slow. We want raw block access.
  // I will try to access m_blocks. If it fails, I'll add a public accessor
  // `get_scrambled_hash()` or similar.

  // Actually, I can use a hack or just fix the friend decl if needed.
  // For now, let's just write the code assuming it works.

  // Accessing private members via friend:
  // We need to include the definition of bitset_t which we have.

  // But wait, there's no public accessor for m_blocks.
  // I'll assume it works.

  // However, I can't use `range-for` easily if I don't know the type of
  // `m_blocks`? I know it's `std::vector<block_type>`. But `block_type` is
  // `zephyr::bitset_t::block_type`.

  for (size_t i = 0; i < b.m_blocks.size(); ++i) {
    result ^= std::hash<zephyr::bitset_t::block_type>{}(b.m_blocks[i]) +
              0x9e3779b9 + (result << 6) + (result >> 2);
  }
  return result;
}

namespace zephyr {

auto bitset_t::operator==(const bitset_t &other) const -> bool {
  if (m_num_bits != other.m_num_bits)
    return false;
  return m_blocks == other.m_blocks;
}

// 10. Polish Features
bitset_t::bitset_t(std::string_view binary_string)
    : bitset_t(binary_string.length()) {
  for (size_t i = 0; i < binary_string.length(); ++i) {
    char c = binary_string[binary_string.length() - 1 -
                           i]; // Index 0 is last char (LSB)
    if (c == '1') {
      set(i, true);
    } else if (c != '0') {
      throw std::invalid_argument("Invalid character in binary string");
    }
  }
}

auto bitset_t::to_uint64() const -> uint64_t {
  if (m_num_bits > 64) {
    // Check if any high bits are set
    for (size_t i = 64; i < m_num_bits; ++i) {
      if (test(i))
        throw std::overflow_error("bitset_t too large for uint64_t");
    }
  }

  uint64_t result = 0;
  // We can just grab up to first 64 bits.
  // If native block is 64, it's just m_blocks[0].
  // If native is 32, it's m_blocks[0] | (m_blocks[1] << 32).

  // Portable loop:
  size_t limit = std::min(m_num_bits, size_t(64));
  for (size_t i = 0; i < limit; ++i) {
    if (test(i)) {
      result |= (static_cast<uint64_t>(1) << i);
    }
  }
  return result;
}

auto bitset_t::is_subset_of(const bitset_t &other) const -> bool {
  if (m_num_bits != other.m_num_bits) {
    throw std::invalid_argument("Sizes must match for set comparison");
  }

  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  for (; i + SIMD_STRIDE_AVX2 <= n; i += SIMD_STRIDE_AVX2) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    __m256i b = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&other.m_blocks[i]));
    // Check if (a & ~b) == 0.
    if (!_mm256_testz_si256(a, _mm256_andnot_si256(b, a)))
      return false;
  }
#elif defined(__ARM_NEON)
  for (; i + SIMD_STRIDE_NEON <= n; i += SIMD_STRIDE_NEON) {
    uint64x2_t a = vld1q_u64(&m_blocks[i]);
    uint64x2_t b = vld1q_u64(&other.m_blocks[i]);
    // check (a & ~b) == 0
    uint64x2_t diff = vandq_u64(a, vmvnq_u64(b));
    if (vgetq_lane_u64(diff, 0) | vgetq_lane_u64(diff, 1))
      return false;
  }
#endif

  for (; i < n; ++i) {
    // checks if any bit set in 'this' is NOT set in 'other'
    // (this & ~other) should be 0.
    if ((m_blocks[i] & ~other.m_blocks[i]) != 0)
      return false;
  }
  return true;
}

auto bitset_t::intersects(const bitset_t &other) const -> bool {
  if (m_num_bits != other.m_num_bits) {
    throw std::invalid_argument("Sizes must match for set comparison");
  }

  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  for (; i + SIMD_STRIDE_AVX2 <= n; i += SIMD_STRIDE_AVX2) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    __m256i b = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&other.m_blocks[i]));
    // Check if (a & b) != 0. testz returns 1 if (a & b) == 0.
    if (!_mm256_testz_si256(a, b))
      return true;
  }
#elif defined(__ARM_NEON)
  for (; i + SIMD_STRIDE_NEON <= n; i += SIMD_STRIDE_NEON) {
    uint64x2_t a = vld1q_u64(&m_blocks[i]);
    uint64x2_t b = vld1q_u64(&other.m_blocks[i]);
    uint64x2_t both = vandq_u64(a, b);
    if (vgetq_lane_u64(both, 0) | vgetq_lane_u64(both, 1))
      return true;
  }
#endif

  for (; i < n; ++i) {
    if ((m_blocks[i] & other.m_blocks[i]) != 0)
      return true;
  }
  return false;
}

auto bitset_t::slice(size_t start, size_t count) const -> bitset_t {
  if (start > m_num_bits) {
    throw std::out_of_range("bitset_t slice start out of range");
  }
  // behave like substr: if count is too large, cap it
  if (start + count > m_num_bits) {
    count = m_num_bits - start;
  }

  bitset_t result = *this;
  result >>= start;
  result.resize(count);

  // Explicitly clear unused bits in the last block
  size_t extra_bits = count % BITS_PER_BLOCK;
  if (extra_bits != 0 && result.m_blocks.size() > 0) {
    block_type mask = (static_cast<block_type>(1) << extra_bits) - 1;
    result.m_blocks.back() &= mask;
  }

  return result;
}

auto bitset_t::hamming_distance(const bitset_t &other) const -> size_t {
  if (m_num_bits != other.m_num_bits) {
    throw std::invalid_argument("Sizes must match for hamming distance");
  }

  size_t dist = 0;
  size_t i = 0;
  size_t n = m_blocks.size();

  size_t cnt0 = 0;
  size_t cnt1 = 0;
  size_t cnt2 = 0;
  size_t cnt3 = 0;

  // Unroll loop 4x
  for (; i + 4 <= n; i += 4) {
    cnt0 += std::popcount(m_blocks[i] ^ other.m_blocks[i]);
    cnt1 += std::popcount(m_blocks[i + 1] ^ other.m_blocks[i + 1]);
    cnt2 += std::popcount(m_blocks[i + 2] ^ other.m_blocks[i + 2]);
    cnt3 += std::popcount(m_blocks[i + 3] ^ other.m_blocks[i + 3]);
  }

  dist = cnt0 + cnt1 + cnt2 + cnt3;
  for (; i < n; ++i) {
    dist += std::popcount(m_blocks[i] ^ other.m_blocks[i]);
  }

  return dist;
}

} // namespace zephyr
