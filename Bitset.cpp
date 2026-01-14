#include "Bitset.hpp"
#include <cmath>

Bitset::Bitset(size_t numBits) : m_numBits(numBits) {
  // Calculate required blocks, ceiling division
  size_t numBlocks = (numBits + BitsPerBlock - 1) / BitsPerBlock;
  m_blocks.resize(numBlocks, 0);
}

// Basic Ops inlined in header

void Bitset::resize(size_t newNumBits) {
  m_numBits = newNumBits;
  size_t numBlocks = (newNumBits + BitsPerBlock - 1) / BitsPerBlock;
  m_blocks.resize(numBlocks);

  // Optional: clear any "garbage" bits in the last block if we shrank/grew
  // weirdly? For a simple resize, vector handles zero-initialization of new
  // blocks. However, if we shrank, the last partial block might have extra bits
  // set. We strictly enforce bounds in accessors, so this might not be
  // critical, but cleaning up high bits is good practice if we implement bulk
  // operations later. skipping for now as per "basic" req.
}

size_t Bitset::size() const { return m_numBits; }

// --- Advanced Features Implementation ---

// 2. Mass Bitwise Operations
Bitset &Bitset::operator&=(const Bitset &rhs) {
  if (m_numBits != rhs.m_numBits) {
    throw std::invalid_argument(
        "Bitset sizes must match for bitwise operations");
  }

  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  // Process 256 bits (4 x 64-bit blocks) at a time
  for (; i + 4 <= n; i += 4) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    __m256i b =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&rhs.m_blocks[i]));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&m_blocks[i]),
                        _mm256_and_si256(a, b));
  }
#elif defined(__ARM_NEON)
  // Process 128 bits (2 x 64-bit blocks) at a time
  for (; i + 2 <= n; i += 2) {
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

Bitset &Bitset::operator|=(const Bitset &rhs) {
  if (m_numBits != rhs.m_numBits) {
    throw std::invalid_argument(
        "Bitset sizes must match for bitwise operations");
  }

  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  for (; i + 4 <= n; i += 4) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    __m256i b =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&rhs.m_blocks[i]));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&m_blocks[i]),
                        _mm256_or_si256(a, b));
  }
#elif defined(__ARM_NEON)
  for (; i + 2 <= n; i += 2) {
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

Bitset &Bitset::operator^=(const Bitset &rhs) {
  if (m_numBits != rhs.m_numBits) {
    throw std::invalid_argument(
        "Bitset sizes must match for bitwise operations");
  }

  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  for (; i + 4 <= n; i += 4) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    __m256i b =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&rhs.m_blocks[i]));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&m_blocks[i]),
                        _mm256_xor_si256(a, b));
  }
#elif defined(__ARM_NEON)
  for (; i + 2 <= n; i += 2) {
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

Bitset Bitset::operator~() const {
  Bitset result(m_numBits);
  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  __m256i allOnes = _mm256_set1_epi64x(-1);
  for (; i + 4 <= n; i += 4) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&result.m_blocks[i]),
                        _mm256_xor_si256(a, allOnes));
  }
#elif defined(__ARM_NEON)
  for (; i + 2 <= n; i += 2) {
    uint64x2_t a = vld1q_u64(&m_blocks[i]);
    vst1q_u64(&result.m_blocks[i], vmvnq_u64(a));
  }
#endif

  for (; i < n; ++i) {
    result.m_blocks[i] = ~m_blocks[i];
  }

  // Mask out the unused bits in the last block
  size_t extraBits = m_numBits % BitsPerBlock;
  if (extraBits != 0) {
    BlockType mask = (static_cast<BlockType>(1) << extraBits) - 1;
    result.m_blocks.back() &= mask;
  }

  return result;
}

// 3. Population & Search
size_t Bitset::count() const {
  size_t cnt = 0;
  for (const auto &block : m_blocks) {
    cnt += std::popcount(block);
  }
  return cnt;
}

bool Bitset::any() const {
  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  for (; i + 4 <= n; i += 4) {
    __m256i val =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    if (!_mm256_testz_si256(val, val))
      return true;
  }
#elif defined(__ARM_NEON)
  for (; i + 2 <= n; i += 2) {
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

bool Bitset::none() const { return !any(); }

bool Bitset::all() const {
  size_t fullBlocks = m_numBits / BitsPerBlock;
  size_t i = 0;

#if defined(__AVX2__)
  __m256i allOnes = _mm256_set1_epi64x(-1);
  for (; i + 4 <= fullBlocks; i += 4) {
    __m256i val =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    // Check if any bit is ZERO. testc(a,b) returns 1 if b has all bits of a.
    // So testc(val, allOnes) checks if val has all bits of allOnes... wait.
    // testc(a,b) : (b & ~a) == 0 ?? No.
    // _mm256_testc_si256(a, b) returns 1 if (a & ~b) == 0.
    // We want (allOnes & ~val) == 0. So testc(allOnes, val).
    if (!_mm256_testc_si256(val, allOnes))
      return false;
  }
#elif defined(__ARM_NEON)
  for (; i + 2 <= fullBlocks; i += 2) {
    uint64x2_t val = vld1q_u64(&m_blocks[i]);
    // check if all bits are 1. Invert and check if 0.
    val = vmvnq_u64(val);
    if (vgetq_lane_u64(val, 0) | vgetq_lane_u64(val, 1))
      return false;
  }
#endif

  // Check full blocks
  for (; i < fullBlocks; ++i) {
    if (m_blocks[i] != static_cast<BlockType>(~0))
      return false;
  }

  // Check last partial block
  size_t extraBits = m_numBits % BitsPerBlock;
  if (extraBits != 0) {
    BlockType mask = (static_cast<BlockType>(1) << extraBits) - 1;
    if ((m_blocks.back() & mask) != mask)
      return false;
  }

  return true;
}

std::optional<size_t> Bitset::findFirstSet() const {
  size_t fullBlocks = m_numBits / BitsPerBlock;
  size_t i = 0;

  // SIMD: Skip 256-bit / 128-bit chunks that are all zero
#if defined(__AVX2__)
  for (; i + 4 <= fullBlocks; i += 4) {
    __m256i val =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    if (_mm256_testz_si256(val, val))
      continue; // All zero, skip
    break;      // Found non-zero
  }
#elif defined(__ARM_NEON)
  for (; i + 2 <= fullBlocks; i += 2) {
    uint64x2_t val = vld1q_u64(&m_blocks[i]);
    // logical OR of two 64-bit lanes
    uint64_t low = vgetq_lane_u64(val, 0);
    uint64_t high = vgetq_lane_u64(val, 1);
    if ((low | high) == 0)
      continue;
    break;
  }
#endif

  // Check blocks
  for (; i < fullBlocks; ++i) {
    if (m_blocks[i] != 0) {
      size_t bit = std::countr_zero(m_blocks[i]);
      return i * BitsPerBlock + bit;
    }
  }

  // Check last partial block
  size_t extraBits = m_numBits % BitsPerBlock;
  if (extraBits != 0) {
    BlockType mask = (static_cast<BlockType>(1) << extraBits) - 1;
    if ((m_blocks.back() & mask) != 0) {
      size_t bit = std::countr_zero(m_blocks.back());
      return fullBlocks * BitsPerBlock + bit;
    }
  }

  return std::nullopt;
}

std::optional<size_t> Bitset::findFirstZero() const {
  size_t fullBlocks = m_numBits / BitsPerBlock;
  size_t i = 0;

  // SIMD: Skip chunks that are all ones
#if defined(__AVX2__)
  __m256i allOnes = _mm256_set1_epi64x(-1);
  for (; i + 4 <= fullBlocks; i += 4) {
    __m256i val =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    // Check if (val & ~allOnes) == 0 ? No. Check if (allOnes & ~val) == 0.
    if (_mm256_testc_si256(val, allOnes))
      continue; // All ones, skip
    break;
  }
#elif defined(__ARM_NEON)
  for (; i + 2 <= fullBlocks; i += 2) {
    uint64x2_t val = vld1q_u64(&m_blocks[i]);
    val = vmvnq_u64(val); // invert
    uint64_t low = vgetq_lane_u64(val, 0);
    uint64_t high = vgetq_lane_u64(val, 1);
    if ((low | high) == 0)
      continue; // Original was all ones
    break;
  }
#endif

  // Check full blocks
  for (; i < fullBlocks; ++i) {
    if (m_blocks[i] != static_cast<BlockType>(~0)) {
      size_t bit = std::countr_one(m_blocks[i]);
      return i * BitsPerBlock + bit;
    }
  }

  // Check last partial block
  size_t extraBits = m_numBits % BitsPerBlock;
  if (extraBits != 0) {
    BlockType mask = (static_cast<BlockType>(1) << extraBits) - 1;
    BlockType val = m_blocks.back();
    // We only care about the bits within the mask.
    // If (val & mask) != mask, there is a zero.
    if ((val & mask) != mask) {
      // Invert logic: find first zero in 'val' is like find first set in '~val'
      // blocked out by mask.
      BlockType inverted = ~val & mask;
      size_t bit = std::countr_zero(inverted);
      return fullBlocks * BitsPerBlock + bit;
    }
  }

  return std::nullopt;
}

// 4. String Conversion
std::string Bitset::toString() const {
  std::string s;
  s.reserve(m_numBits);
  for (size_t i = 0; i < m_numBits; ++i) {
    s += test(m_numBits - 1 - i)
             ? '1'
             : '0'; // Most significant bit first usually? Or LSB?
                    // std::bitset prints MSB at index 0 (left).
    // Let's stick to index 0 is right-most (LSB), but string usually reads
    // left-to-right MSB?? Wait, standard `std::bitset::to_string()`: "The
    // character at position i in the string yields the value of the bit at
    // position N - 1 - i" So yes, MSB first.
  }
  return s;
}

std::ostream &operator<<(std::ostream &os, const Bitset &b) {
  os << b.toString();
  return os;
}

// 5. File I/O
void Bitset::save(const std::string &filename) const {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs)
    throw std::runtime_error("Cannot open file for writing: " + filename);

  // Header: "BITSET" magic, version, numBits
  const char magic[] = "BITSET";
  uint32_t version = 1;
  uint64_t nBits =
      static_cast<uint64_t>(m_numBits); // Fixed width for portability

  ofs.write("BITSET", 6);
  ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
  ofs.write(reinterpret_cast<const char *>(&nBits), sizeof(nBits));

  // Data
  // We write the raw blocks which are machine-dependent (BlockType size).
  // For true cross-platform portability we should standardized on uint64_t or
  // similar. But requirement was "cross compatible between 32 and 64 bit
  // systems LOGICALLY", file format might differ? User asked "cross compatible
  // ... bitset". If I write size_t blocks from 64-bit machine and read on
  // 32-bit, size_t is different. So I MUST normalize storage for file I/O.
  // Let's write as constant uint64_t blocks.

  size_t nBlocks = (m_numBits + 63) / 64; // normalize to 64-bit blocks
  std::vector<uint64_t> portableBlocks(nBlocks, 0);

  // Copy data to portable blocks
  for (size_t i = 0; i < m_numBits; ++i) {
    if (test(i)) {
      portableBlocks[i / 64] |= (static_cast<uint64_t>(1) << (i % 64));
    }
  }

  ofs.write(reinterpret_cast<const char *>(portableBlocks.data()),
            portableBlocks.size() * sizeof(uint64_t));
}

void Bitset::load(const std::string &filename) {
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

  uint64_t nBits;
  ifs.read(reinterpret_cast<char *>(&nBits), sizeof(nBits));

  resize(static_cast<size_t>(nBits));

  // Read portable 64-bit blocks
  size_t nBlocks = (nBits + 63) / 64;
  std::vector<uint64_t> portableBlocks(nBlocks);
  ifs.read(reinterpret_cast<char *>(portableBlocks.data()),
           nBlocks * sizeof(uint64_t));

  // Reconstruct internal blocks
  // Efficient way: clear and set. Or just copy bits.
  // Reset handled by resize effectively (if we assume clean state, but resize
  // might keep old data?) Actually resize keeps data. So we should probably
  // clear first or overwrite. Let's just overwrite using set which is safe but
  // slower, or optimized block copy. Since internal BlockType might be 32 or
  // 64, manual bit/block copy is safest.

  for (size_t i = 0; i < static_cast<size_t>(nBits); ++i) {
    bool bitVal = (portableBlocks[i / 64] >> (i % 64)) & 1;
    set(i, bitVal);
  }
}

// 7. Shift Operators
Bitset &Bitset::operator<<=(size_t pos) {
  if (pos == 0)
    return *this;
  if (pos >= m_numBits) {
    std::fill(m_blocks.begin(), m_blocks.end(), 0);
    return *this;
  }

  size_t blockShift = pos / BitsPerBlock;
  size_t bitShift = pos % BitsPerBlock;

  if (bitShift == 0) {
    for (size_t i = m_blocks.size(); i > blockShift; --i) {
      m_blocks[i - 1] = m_blocks[i - 1 - blockShift];
    }
  } else {
    for (size_t i = m_blocks.size(); i > blockShift; --i) {
      size_t destIdx = i - 1;
      size_t srcIdx = destIdx - blockShift;
      BlockType val = m_blocks[srcIdx] << bitShift;
      if (srcIdx > 0) {
        val |= (m_blocks[srcIdx - 1] >> (BitsPerBlock - bitShift));
      }
      m_blocks[destIdx] = val;
    }
  }
  std::fill(m_blocks.begin(), m_blocks.begin() + blockShift, 0);
  if (blockShift < m_blocks.size()) {
    m_blocks[blockShift] &= (~static_cast<BlockType>(0)) << bitShift;
  }
  size_t extraBits = m_numBits % BitsPerBlock;
  if (extraBits != 0) {
    BlockType mask = (static_cast<BlockType>(1) << extraBits) - 1;
    m_blocks.back() &= mask;
  }
  return *this;
}

Bitset &Bitset::operator>>=(size_t pos) {
  if (pos == 0)
    return *this;
  if (pos >= m_numBits) {
    std::fill(m_blocks.begin(), m_blocks.end(), 0);
    return *this;
  }

  size_t blockShift = pos / BitsPerBlock;
  size_t bitShift = pos % BitsPerBlock;

  if (bitShift == 0) {
    for (size_t i = 0; i < m_blocks.size() - blockShift; ++i) {
      m_blocks[i] = m_blocks[i + blockShift];
    }
  } else {
    for (size_t i = 0; i < m_blocks.size() - blockShift; ++i) {
      size_t destIdx = i;
      size_t srcIdx = destIdx + blockShift;

      BlockType val = m_blocks[srcIdx] >> bitShift;
      if (srcIdx + 1 < m_blocks.size()) {
        val |= (m_blocks[srcIdx + 1] << (BitsPerBlock - bitShift));
      }
      m_blocks[destIdx] = val;
    }
  }
  std::fill(m_blocks.end() - blockShift, m_blocks.end(), 0);
  size_t extraBits = m_numBits % BitsPerBlock;
  if (extraBits != 0) {
    BlockType mask = (static_cast<BlockType>(1) << extraBits) - 1;
    m_blocks.back() &= mask;
  }
  return *this;
}

// 8. Range Operations
void Bitset::setRange(size_t start, size_t count, bool value) {
  if (count == 0)
    return;
  size_t end = start + count;
  if (end > m_numBits) {
    throw std::out_of_range("Range exceeds bitset size");
  }

  size_t currentBlock = start / BitsPerBlock;
  size_t finalBlock = (end - 1) / BitsPerBlock;

  if (start % BitsPerBlock != 0) {
    BlockType mask = (~static_cast<BlockType>(0)) << (start % BitsPerBlock);
    // If we are solely in the first block, we must also apply the end mask
    if (currentBlock == finalBlock) {
      size_t localEnd = end % BitsPerBlock;
      if (localEnd != 0) {
        mask &= (static_cast<BlockType>(1) << localEnd) - 1;
      }
    }

    if (value)
      m_blocks[currentBlock] |= mask;
    else
      m_blocks[currentBlock] &= ~mask;

    if (currentBlock == finalBlock)
      return;
    currentBlock++;
  }

  if (currentBlock < finalBlock) {
    BlockType fillVal = value ? ~static_cast<BlockType>(0) : 0;
    std::fill(m_blocks.begin() + currentBlock, m_blocks.begin() + finalBlock,
              fillVal);
    currentBlock = finalBlock;
  }

  if (currentBlock <= finalBlock) {
    size_t localEnd = end % BitsPerBlock;
    if (localEnd == 0)
      localEnd = BitsPerBlock;

    BlockType mask;
    if (localEnd == BitsPerBlock)
      mask = ~static_cast<BlockType>(0);
    else
      mask = (static_cast<BlockType>(1) << localEnd) - 1;

    if (value)
      m_blocks[finalBlock] |= mask;
    else
      m_blocks[finalBlock] &= ~mask;
  }
}

// 9. Hash Support
namespace std {
size_t hash<Bitset>::operator()(const Bitset &b) const {
  size_t result = 0;
  for (const auto &block : b.m_blocks) {
    result ^= std::hash<Bitset::BlockType>{}(block) + 0x9e3779b9 +
              (result << 6) + (result >> 2);
  }
  return result;
}
} // namespace std

bool Bitset::operator==(const Bitset &other) const {
  if (m_numBits != other.m_numBits)
    return false;
  return m_blocks == other.m_blocks;
}

// 10. Polish Features
Bitset::Bitset(std::string_view binaryString) : Bitset(binaryString.length()) {
  for (size_t i = 0; i < binaryString.length(); ++i) {
    char c = binaryString[binaryString.length() - 1 -
                          i]; // Index 0 is last char (LSB)
    if (c == '1') {
      set(i, true);
    } else if (c != '0') {
      throw std::invalid_argument("Invalid character in binary string");
    }
  }
}

uint64_t Bitset::to_uint64() const {
  if (m_numBits > 64) {
    // Check if any high bits are set
    for (size_t i = 64; i < m_numBits; ++i) {
      if (test(i))
        throw std::overflow_error("Bitset too large for uint64_t");
    }
  }

  uint64_t result = 0;
  // We can just grab up to first 64 bits.
  // If native block is 64, it's just m_blocks[0].
  // If native is 32, it's m_blocks[0] | (m_blocks[1] << 32).

  // Portable loop:
  size_t limit = std::min(m_numBits, size_t(64));
  for (size_t i = 0; i < limit; ++i) {
    if (test(i)) {
      result |= (static_cast<uint64_t>(1) << i);
    }
  }
  return result;
}

bool Bitset::isSubsetOf(const Bitset &other) const {
  if (m_numBits != other.m_numBits) {
    throw std::invalid_argument("Sizes must match for set comparison");
  }

  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  for (; i + 4 <= n; i += 4) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    __m256i b = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&other.m_blocks[i]));
    // Check if (a & ~b) == 0.
    if (!_mm256_testz_si256(a, _mm256_andnot_si256(b, a)))
      return false;
  }
#elif defined(__ARM_NEON)
  for (; i + 2 <= n; i += 2) {
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

bool Bitset::intersects(const Bitset &other) const {
  if (m_numBits != other.m_numBits) {
    throw std::invalid_argument("Sizes must match for set comparison");
  }

  size_t i = 0;
  size_t n = m_blocks.size();

#if defined(__AVX2__)
  for (; i + 4 <= n; i += 4) {
    __m256i a =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&m_blocks[i]));
    __m256i b = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(&other.m_blocks[i]));
    // Check if (a & b) != 0. testz returns 1 if (a & b) == 0.
    if (!_mm256_testz_si256(a, b))
      return true;
  }
#elif defined(__ARM_NEON)
  for (; i + 2 <= n; i += 2) {
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

Bitset Bitset::slice(size_t start, size_t count) const {
  if (start > m_numBits) {
    throw std::out_of_range("Bitset slice start out of range");
  }
  // behave like substr: if count is too large, cap it
  if (start + count > m_numBits) {
    count = m_numBits - start;
  }

  Bitset result = *this;
  result >>= start;
  result.resize(count);

  // Explicitly clear unused bits in the last block
  size_t extraBits = count % BitsPerBlock;
  if (extraBits != 0 && result.m_blocks.size() > 0) {
    BlockType mask = (static_cast<BlockType>(1) << extraBits) - 1;
    result.m_blocks.back() &= mask;
  }

  return result;
}

size_t Bitset::hammingDistance(const Bitset &other) const {
  if (m_numBits != other.m_numBits) {
    throw std::invalid_argument("Sizes must match for hamming distance");
  }

  size_t dist = 0;
  size_t i = 0;
  size_t n = m_blocks.size();

  // SIMD-friendly loop (compiler auto-vectorization target)
  for (; i < n; ++i) {
    dist += std::popcount(m_blocks[i] ^ other.m_blocks[i]);
  }

  return dist;
}
