#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// SIMD Headers
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace zephyr {

class bitset_t;

} // namespace zephyr

namespace std {
template <> struct hash<zephyr::bitset_t> {
  size_t operator()(const zephyr::bitset_t &b) const;
};
} // namespace std

namespace zephyr {

class bitset_t {
public:
  using block_type = std::size_t;
  static constexpr size_t BITS_PER_BLOCK = sizeof(block_type) * 8;

  explicit bitset_t(size_t num_bits);
  ~bitset_t() = default;
  bitset_t(const bitset_t &) = default;
  bitset_t &operator=(const bitset_t &) = default;
  bitset_t(bitset_t &&) noexcept = default;
  bitset_t &operator=(bitset_t &&) noexcept = default;

  // Set bit at index to value
  auto set(size_t index, bool value = true) -> void {
    if (index >= m_num_bits)
      throw std::out_of_range("bitset_t index out of range");
    set_unchecked(index, value);
  }

  auto set_unchecked(size_t index, bool value) -> void {
    block_type mask = static_cast<block_type>(1) << (index % BITS_PER_BLOCK);
    // Branchless set:
    // If value is 1, -value is all 1s (on 2s complement).
    // If value is 0, -value is 0.
    // (blocks & ~mask) clears the bit.
    // ((-value) & mask) sets the bit if value is 1, else 0.
    m_blocks[index / BITS_PER_BLOCK] =
        (m_blocks[index / BITS_PER_BLOCK] & ~mask) |
        (static_cast<block_type>(-static_cast<std::ptrdiff_t>(value)) & mask);
  }

  // Get bit at index
  [[nodiscard]] auto test(size_t index) const -> bool {
    if (index >= m_num_bits)
      throw std::out_of_range("bitset_t index out of range");
    return test_unchecked(index);
  }

  [[nodiscard]] auto test_unchecked(size_t index) const -> bool {
    return (m_blocks[index / BITS_PER_BLOCK] &
            (static_cast<block_type>(1) << (index % BITS_PER_BLOCK))) != 0;
  }

  // Flip bit at index
  auto flip(size_t index) -> void {
    if (index >= m_num_bits)
      throw std::out_of_range("bitset_t index out of range");
    m_blocks[index / BITS_PER_BLOCK] ^=
        (static_cast<block_type>(1) << (index % BITS_PER_BLOCK));
  }

  // Reset bit at index (set to 0)
  auto reset(size_t index) -> void { set(index, false); }

  // Resize the bitset
  auto resize(size_t new_num_bits) -> void;

  // Get current number of bits
  [[nodiscard]] auto size() const -> size_t;

  // --- Advanced Features ---

  // 1. Proxy Reference for operator[]
  class reference_t {
    friend class bitset_t;
    bitset_t *m_bitset;
    size_t m_index;

    reference_t(bitset_t *bitset, size_t index)
        : m_bitset(bitset), m_index(index) {}

  public:
    ~reference_t() = default;

    // Convert to bool
    operator bool() const { return m_bitset->test(m_index); }

    // Assign bool
    auto operator=(bool x) -> reference_t & {
      m_bitset->set(m_index, x);
      return *this;
    }

    // Assign another reference
    auto operator=(const reference_t &rhs) -> reference_t & {
      m_bitset->set(m_index, (bool)rhs);
      return *this;
    }

    auto flip() -> reference_t & {
      m_bitset->flip(m_index);
      return *this;
    }

    auto operator~() const -> bool { return !m_bitset->test(m_index); }
  };

  auto operator[](size_t index) -> reference_t {
    return reference_t(this, index);
  }

  auto operator[](size_t index) const -> bool { return test(index); }

  // 2. Mass Bitwise Operations
  auto operator&=(const bitset_t &rhs) -> bitset_t &;
  auto operator|=(const bitset_t &rhs) -> bitset_t &;
  auto operator^=(const bitset_t &rhs) -> bitset_t &;
  auto operator~() const -> bitset_t;

  friend auto operator&(const bitset_t &lhs, const bitset_t &rhs) -> bitset_t {
    bitset_t result = lhs;
    result &= rhs;
    return result;
  }

  friend auto operator|(const bitset_t &lhs, const bitset_t &rhs) -> bitset_t {
    bitset_t result = lhs;
    result |= rhs;
    return result;
  }

  friend auto operator^(const bitset_t &lhs, const bitset_t &rhs) -> bitset_t {
    bitset_t result = lhs;
    result ^= rhs;
    return result;
  }

  // 3. Population & Search
  [[nodiscard]] auto count() const -> size_t;
  [[nodiscard]] auto any() const -> bool;
  [[nodiscard]] auto none() const -> bool;
  [[nodiscard]] auto all() const -> bool;

  [[nodiscard]] auto find_first_set() const -> std::optional<size_t>;
  [[nodiscard]] auto find_first_zero() const -> std::optional<size_t>;
  [[nodiscard]] auto find_next_set(size_t index) const -> std::optional<size_t>;
  [[nodiscard]] auto find_next_zero(size_t index) const
      -> std::optional<size_t>;

  // 4. String Conversion
  [[nodiscard]] auto to_string() const -> std::string;
  friend auto operator<<(std::ostream &os, const bitset_t &b) -> std::ostream &;

  // 6. Set-Bit Iterator
  class ones_view_t {
    const bitset_t *m_bitset;

  public:
    class iterator_t {
      const bitset_t *m_bitset;
      size_t m_current;

      auto advance() -> void {
        if (m_current >= m_bitset->size())
          return;
        m_current++;

        // Fast forward through zero blocks
        while (m_current < m_bitset->size()) {
          if (m_bitset->test(m_current))
            return;

          // Optimization: if we are at start of a block and it's 0, skip.
          if (m_current % BITS_PER_BLOCK == 0) {
            size_t block_idx = m_current / BITS_PER_BLOCK;
            if (block_idx < m_bitset->m_blocks.size() &&
                m_bitset->m_blocks[block_idx] == 0) {
              m_current += BITS_PER_BLOCK;
              continue;
            }
          }
          m_current++;
        }
      }

    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = size_t;
      using difference_type = std::ptrdiff_t;
      using pointer = const size_t *;
      using reference = const size_t &;

      iterator_t(const bitset_t *bs, size_t start)
          : m_bitset(bs), m_current(start) {
        // If start is not set, find first set
        if (m_current < m_bitset->size() && !m_bitset->test(m_current)) {
          while (m_current < m_bitset->size() && !m_bitset->test(m_current)) {
            if (m_current % BITS_PER_BLOCK == 0) {
              size_t block_idx = m_current / BITS_PER_BLOCK;
              if (block_idx < m_bitset->m_blocks.size() &&
                  m_bitset->m_blocks[block_idx] == 0) {
                m_current += BITS_PER_BLOCK;
                continue;
              }
            }
            m_current++;
          }
        }
      }

      auto operator*() const -> size_t { return m_current; }

      auto operator++() -> iterator_t & {
        advance();
        return *this;
      }

      auto operator++(int) -> iterator_t {
        iterator_t tmp = *this;
        advance();
        return tmp;
      }

      auto operator==(const iterator_t &other) const -> bool {
        // End iterator is marked by index >= size
        bool this_end = m_current >= m_bitset->size();
        bool other_end = other.m_current >= other.m_bitset->size();
        if (this_end && other_end)
          return true;
        return m_current == other.m_current;
      }

      auto operator!=(const iterator_t &other) const -> bool {
        return !(*this == other);
      }
    };

    ones_view_t(const bitset_t *bs) : m_bitset(bs) {}

    auto begin() const -> iterator_t { return iterator_t(m_bitset, 0); }

    auto end() const -> iterator_t {
      return iterator_t(m_bitset, m_bitset->size());
    }
  };

  auto ones() const -> ones_view_t { return ones_view_t(this); }

  // 7. Shift Operators
  auto operator<<=(size_t pos) -> bitset_t &;
  auto operator>>=(size_t pos) -> bitset_t &;

  auto operator<<(size_t pos) const -> bitset_t {
    bitset_t res = *this;
    res <<= pos;
    return res;
  }

  auto operator>>(size_t pos) const -> bitset_t {
    bitset_t res = *this;
    res >>= pos;
    return res;
  }

  // 8. Range Operations
  auto set_range(size_t start, size_t count, bool value) -> void;
  auto flip_range(size_t start, size_t count) -> void;

  // 10. Polish Features
  explicit bitset_t(std::string_view binary_string);
  auto to_uint64() const -> uint64_t;
  auto is_subset_of(const bitset_t &other) const -> bool;
  auto intersects(const bitset_t &other) const -> bool;
  auto slice(size_t start, size_t count) const -> bitset_t;
  auto hamming_distance(const bitset_t &other) const -> size_t;

  class zeros_view_t {
    const bitset_t *m_bitset;

  public:
    class iterator_t {
      const bitset_t *m_bitset;
      size_t m_current;

      auto advance() -> void {
        if (m_current >= m_bitset->size())
          return;
        m_current++;
        // Find next zero bit
        while (m_current < m_bitset->size()) {
          if (!m_bitset->test(m_current))
            return;

          // Optimization: if we are at start of a block and it's all ones (~0),
          // skip.
          if (m_current % BITS_PER_BLOCK == 0) {
            size_t block_idx = m_current / BITS_PER_BLOCK;
            // Be careful with last partial block, but assuming full blocks
            // logic holds for internal storage
            if (block_idx < m_bitset->m_blocks.size() &&
                m_bitset->m_blocks[block_idx] == ~static_cast<block_type>(0)) {
              m_current += BITS_PER_BLOCK;
              continue;
            }
          }
          m_current++;
        }
      }

    public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = size_t;
      using difference_type = std::ptrdiff_t;
      using pointer = const size_t *;
      using reference = const size_t &;

      iterator_t(const bitset_t *bs, size_t start)
          : m_bitset(bs), m_current(start) {
        if (m_current < m_bitset->size() && m_bitset->test(m_current)) {
          // Current is set, find first zero
          while (m_current < m_bitset->size() && m_bitset->test(m_current)) {
            if (m_current % BITS_PER_BLOCK == 0) {
              size_t block_idx = m_current / BITS_PER_BLOCK;
              if (block_idx < m_bitset->m_blocks.size() &&
                  m_bitset->m_blocks[block_idx] ==
                      ~static_cast<block_type>(0)) {
                m_current += BITS_PER_BLOCK;
                continue;
              }
            }
            m_current++;
          }
        }
      }

      auto operator*() const -> size_t { return m_current; }

      auto operator++() -> iterator_t & {
        advance();
        return *this;
      }

      auto operator++(int) -> iterator_t {
        iterator_t tmp = *this;
        advance();
        return tmp;
      }

      auto operator==(const iterator_t &other) const -> bool {
        bool this_end = m_current >= m_bitset->size();
        bool other_end = other.m_current >= other.m_bitset->size();
        if (this_end && other_end)
          return true;
        return m_current == other.m_current;
      }

      auto operator!=(const iterator_t &other) const -> bool {
        return !(*this == other);
      }
    };

    zeros_view_t(const bitset_t *bs) : m_bitset(bs) {}

    auto begin() const -> iterator_t { return iterator_t(m_bitset, 0); }

    auto end() const -> iterator_t {
      return iterator_t(m_bitset, m_bitset->size());
    }
  };

  auto zeros() const -> zeros_view_t { return zeros_view_t(this); }
  friend struct std::hash<bitset_t>;

  auto operator==(const bitset_t &other) const -> bool;
  auto operator!=(const bitset_t &other) const -> bool {
    return !(*this == other);
  }

  // 5. File I/O
  auto save(const std::string &filename) const -> void;
  auto load(const std::string &filename) -> void;

private:
  std::vector<block_type> m_blocks;
  size_t m_num_bits;

  [[nodiscard]] auto get_block_index(size_t bit_index) const -> size_t {
    return bit_index / BITS_PER_BLOCK;
  }

  [[nodiscard]] auto get_bit_offset(size_t bit_index) const -> size_t {
    return bit_index % BITS_PER_BLOCK;
  }

public:
  // Helper to swap bytes for arithmetic types if system is Big Endian
  template <typename T> static auto to_little_endian(T val) -> T {
    if constexpr (std::endian::native == std::endian::big) {
      if constexpr (std::is_arithmetic_v<T>) {
        auto *ptr = reinterpret_cast<uint8_t *>(&val);
        std::reverse(ptr, ptr + sizeof(T));
      }
    }
    return val;
  }

  template <typename T> static auto from_little_endian(T val) -> T {
    // The operation is symmetric
    return to_little_endian(val);
  }

  template <typename T> auto write_object(size_t index, T value) -> void {
    // Enforce Little Endian for arithmetic types
    if constexpr (std::is_arithmetic_v<T>) {
      value = to_little_endian(value);
    }

    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&value);
    for (size_t i = 0; i < sizeof(T); ++i) {
      for (size_t bit = 0; bit < 8; ++bit) {
        bool bit_val = (bytes[i] >> bit) & 1;
        set(index + (i * 8) + bit, bit_val);
      }
    }
  }

  template <typename T> auto read_object(size_t index) const -> T {
    T value;
    uint8_t *bytes = reinterpret_cast<uint8_t *>(&value);

    // Initialize bytes to 0 before reading
    std::memset(bytes, 0, sizeof(T));

    for (size_t i = 0; i < sizeof(T); ++i) {
      for (size_t bit = 0; bit < 8; ++bit) {
        if (test(index + (i * 8) + bit)) {
          bytes[i] |= (1 << bit);
        }
      }
    }

    // Convert back to native endianness if necessary
    if constexpr (std::is_arithmetic_v<T>) {
      value = from_little_endian(value);
    }

    return value;
  }
};

} // namespace zephyr
