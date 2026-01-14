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

class Bitset;
namespace std {
template <> struct hash<Bitset> {
  size_t operator()(const Bitset &b) const;
};
} // namespace std

class Bitset {
public:
  using BlockType = std::size_t;
  static constexpr size_t BitsPerBlock = sizeof(BlockType) * 8;

  explicit Bitset(size_t numBits);

  // Set bit at index to value
  void set(size_t index, bool value = true) {
    if (index >= m_numBits)
      throw std::out_of_range("Bitset index out of range");
    setUnchecked(index, value);
  }

  void setUnchecked(size_t index, bool value) {
    if (value)
      m_blocks[index / BitsPerBlock] |=
          (static_cast<BlockType>(1) << (index % BitsPerBlock));
    else
      m_blocks[index / BitsPerBlock] &=
          ~(static_cast<BlockType>(1) << (index % BitsPerBlock));
  }

  // Get bit at index
  [[nodiscard]] bool test(size_t index) const {
    if (index >= m_numBits)
      throw std::out_of_range("Bitset index out of range");
    return testUnchecked(index);
  }

  [[nodiscard]] bool testUnchecked(size_t index) const {
    return (m_blocks[index / BitsPerBlock] &
            (static_cast<BlockType>(1) << (index % BitsPerBlock))) != 0;
  }

  // Flip bit at index
  void flip(size_t index) {
    if (index >= m_numBits)
      throw std::out_of_range("Bitset index out of range");
    m_blocks[index / BitsPerBlock] ^=
        (static_cast<BlockType>(1) << (index % BitsPerBlock));
  }

  // Reset bit at index (set to 0)
  void reset(size_t index) { set(index, false); }

  // Resize the bitset
  void resize(size_t newNumBits);

  // Get current number of bits
  [[nodiscard]] size_t size() const;

  // --- Advanced Features ---

  // 1. Proxy Reference for operator[]
  class Reference {
    friend class Bitset;
    Bitset *m_bitset;
    size_t m_index;

    Reference(Bitset *bitset, size_t index)
        : m_bitset(bitset), m_index(index) {}

  public:
    ~Reference() = default;

    // Convert to bool
    operator bool() const { return m_bitset->test(m_index); }

    // Assign bool
    Reference &operator=(bool x) {
      m_bitset->set(m_index, x);
      return *this;
    }

    // Assign another reference
    Reference &operator=(const Reference &rhs) {
      m_bitset->set(m_index, (bool)rhs);
      return *this;
    }

    Reference &flip() {
      m_bitset->flip(m_index);
      return *this;
    }

    bool operator~() const { return !m_bitset->test(m_index); }
  };

  Reference operator[](size_t index) { return Reference(this, index); }

  bool operator[](size_t index) const { return test(index); }

  // 2. Mass Bitwise Operations
  Bitset &operator&=(const Bitset &rhs);
  Bitset &operator|=(const Bitset &rhs);
  Bitset &operator^=(const Bitset &rhs);
  Bitset operator~() const;

  friend Bitset operator&(const Bitset &lhs, const Bitset &rhs) {
    Bitset result = lhs;
    result &= rhs;
    return result;
  }

  friend Bitset operator|(const Bitset &lhs, const Bitset &rhs) {
    Bitset result = lhs;
    result |= rhs;
    return result;
  }

  friend Bitset operator^(const Bitset &lhs, const Bitset &rhs) {
    Bitset result = lhs;
    result ^= rhs;
    return result;
  }

  // 3. Population & Search
  [[nodiscard]] size_t count() const;
  [[nodiscard]] bool any() const;
  [[nodiscard]] bool none() const;
  [[nodiscard]] bool all() const;

  [[nodiscard]] std::optional<size_t> findFirstSet() const;
  [[nodiscard]] std::optional<size_t> findFirstZero() const;

  // 4. String Conversion
  [[nodiscard]] std::string toString() const;
  friend std::ostream &operator<<(std::ostream &os, const Bitset &b);

  // 6. Set-Bit Iterator
  class OnesView {
    const Bitset *m_bitset;

  public:
    class Iterator {
      const Bitset *m_bitset;
      size_t m_current;

      void advance() {
        if (m_current >= m_bitset->size())
          return;
        m_current++;
        // Find next set bit
        // We can optimize this by accessing blocks, but using public/private
        // helper is fine. Since this is nested/friend, we can access m_blocks
        // if needed, but let's use a helper for clarity or block scans. For
        // "high performance", we must scan blocks.

        // Fast forward through zero blocks
        while (m_current < m_bitset->size()) {
          if (m_bitset->test(m_current))
            return;

          // Optimization: if we are at start of a block and it's 0, skip.
          if (m_current % BitsPerBlock == 0) {
            size_t blockIdx = m_current / BitsPerBlock;
            if (blockIdx < m_bitset->m_blocks.size() &&
                m_bitset->m_blocks[blockIdx] == 0) {
              m_current += BitsPerBlock;
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

      Iterator(const Bitset *bs, size_t start)
          : m_bitset(bs), m_current(start) {
        // If start is not set, find first set
        if (m_current < m_bitset->size() && !m_bitset->test(m_current)) {
          // Try to find first set bit efficiently
          // Re-use advance logic or separate find logic?
          // Let's just call advance? No, advance moves +1 first.
          // We need "ensure valid".
          // Let's rely on the fact that begin() passes firstSet, or 0.
          // If we pass 0 and 0 is unset, we need to move.

          // Let's duplicate scan logic for "init" vs "advance" or make helper.
          // Actually, let's just use the simple loop here for safety,
          // assuming 'start' is a hint.
          // Or efficient scan:
          while (m_current < m_bitset->size() && !m_bitset->test(m_current)) {
            if (m_current % BitsPerBlock == 0) {
              size_t blockIdx = m_current / BitsPerBlock;
              if (blockIdx < m_bitset->m_blocks.size() &&
                  m_bitset->m_blocks[blockIdx] == 0) {
                m_current += BitsPerBlock;
                continue;
              }
            }
            m_current++;
          }
        }
      }

      size_t operator*() const { return m_current; }

      Iterator &operator++() {
        advance();
        return *this;
      }

      Iterator operator++(int) {
        Iterator tmp = *this;
        advance();
        return tmp;
      }

      bool operator==(const Iterator &other) const {
        // End iterator is marked by index >= size
        bool thisEnd = m_current >= m_bitset->size();
        bool otherEnd = other.m_current >= other.m_bitset->size();
        if (thisEnd && otherEnd)
          return true;
        return m_current == other.m_current;
      }

      bool operator!=(const Iterator &other) const { return !(*this == other); }
    };

    OnesView(const Bitset *bs) : m_bitset(bs) {}

    Iterator begin() const { return Iterator(m_bitset, 0); }

    Iterator end() const { return Iterator(m_bitset, m_bitset->size()); }
  };

  OnesView ones() const { return OnesView(this); }

  // 7. Shift Operators
  Bitset &operator<<=(size_t pos);
  Bitset &operator>>=(size_t pos);

  Bitset operator<<(size_t pos) const {
    Bitset res = *this;
    res <<= pos;
    return res;
  }

  Bitset operator>>(size_t pos) const {
    Bitset res = *this;
    res >>= pos;
    return res;
  }

  // 8. Range Operations
  void setRange(size_t start, size_t count, bool value);

  // 10. Polish Features
  explicit Bitset(std::string_view binaryString);
  uint64_t to_uint64() const;
  bool isSubsetOf(const Bitset &other) const;
  bool intersects(const Bitset &other) const;

  class ZerosView {
    const Bitset *m_bitset;

  public:
    class Iterator {
      const Bitset *m_bitset;
      size_t m_current;

      void advance() {
        if (m_current >= m_bitset->size())
          return;
        m_current++;
        // Find next zero bit
        while (m_current < m_bitset->size()) {
          if (!m_bitset->test(m_current))
            return;

          // Optimization: if we are at start of a block and it's all ones (~0),
          // skip.
          if (m_current % BitsPerBlock == 0) {
            size_t blockIdx = m_current / BitsPerBlock;
            // Be careful with last partial block, but assuming full blocks
            // logic holds for internal storage
            if (blockIdx < m_bitset->m_blocks.size() &&
                m_bitset->m_blocks[blockIdx] == ~static_cast<BlockType>(0)) {
              m_current += BitsPerBlock;
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

      Iterator(const Bitset *bs, size_t start)
          : m_bitset(bs), m_current(start) {
        if (m_current < m_bitset->size() && m_bitset->test(m_current)) {
          // Current is set, find first zero
          while (m_current < m_bitset->size() && m_bitset->test(m_current)) {
            if (m_current % BitsPerBlock == 0) {
              size_t blockIdx = m_current / BitsPerBlock;
              if (blockIdx < m_bitset->m_blocks.size() &&
                  m_bitset->m_blocks[blockIdx] == ~static_cast<BlockType>(0)) {
                m_current += BitsPerBlock;
                continue;
              }
            }
            m_current++;
          }
        }
      }

      size_t operator*() const { return m_current; }

      Iterator &operator++() {
        advance();
        return *this;
      }

      Iterator operator++(int) {
        Iterator tmp = *this;
        advance();
        return tmp;
      }

      bool operator==(const Iterator &other) const {
        bool thisEnd = m_current >= m_bitset->size();
        bool otherEnd = other.m_current >= other.m_bitset->size();
        if (thisEnd && otherEnd)
          return true;
        return m_current == other.m_current;
      }

      bool operator!=(const Iterator &other) const { return !(*this == other); }
    };

    ZerosView(const Bitset *bs) : m_bitset(bs) {}

    Iterator begin() const { return Iterator(m_bitset, 0); }

    Iterator end() const { return Iterator(m_bitset, m_bitset->size()); }
  };

  ZerosView zeros() const { return ZerosView(this); }
  friend struct std::hash<Bitset>;

  bool operator==(const Bitset &other) const;
  bool operator!=(const Bitset &other) const { return !(*this == other); }

  // 5. File I/O
  void save(const std::string &filename) const;
  void load(const std::string &filename);

private:
  std::vector<BlockType> m_blocks;
  size_t m_numBits;

  [[nodiscard]] size_t getBlockIndex(size_t bitIndex) const {
    return bitIndex / BitsPerBlock;
  }

  [[nodiscard]] size_t getBitOffset(size_t bitIndex) const {
    return bitIndex % BitsPerBlock;
  }

public:
  // Helper to swap bytes for arithmetic types if system is Big Endian
  template <typename T> static T toLittleEndian(T val) {
    if constexpr (std::endian::native == std::endian::big) {
      if constexpr (std::is_arithmetic_v<T>) {
        auto *ptr = reinterpret_cast<uint8_t *>(&val);
        std::reverse(ptr, ptr + sizeof(T));
      }
    }
    return val;
  }

  template <typename T> static T fromLittleEndian(T val) {
    // The operation is symmetric
    return toLittleEndian(val);
  }

  template <typename T> void writeObject(size_t index, T value) {
    // Enforce Little Endian for arithmetic types
    if constexpr (std::is_arithmetic_v<T>) {
      value = toLittleEndian(value);
    }

    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&value);
    for (size_t i = 0; i < sizeof(T); ++i) {
      for (size_t bit = 0; bit < 8; ++bit) {
        bool bitVal = (bytes[i] >> bit) & 1;
        set(index + (i * 8) + bit, bitVal);
      }
    }
  }

  template <typename T> T readObject(size_t index) const {
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
      value = fromLittleEndian(value);
    }

    return value;
  }
};
