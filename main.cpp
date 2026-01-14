#include "Bitset.hpp"
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

void testProxy() {
  std::cout << "\n--- Proxy Test ---" << std::endl;
  Bitset bs(10);
  bs[0] = true;
  bs[5] = true;
  if (bs[0] && !bs[1] && bs[5])
    std::cout << "Proxy Read/Write PASSED" << std::endl;
}

void testBitwise() {
  std::cout << "\n--- Bitwise Test ---" << std::endl;
  Bitset a(4);
  a[0] = 1;
  a[2] = 1;
  Bitset b(4);
  b[1] = 1;
  b[2] = 1;
  Bitset c = a & b;
  if (!c[0] && !c[1] && c[2])
    std::cout << "Bitwise Ops PASSED" << std::endl;
}

void testSearch() {
  std::cout << "\n--- Search Test ---" << std::endl;
  Bitset bs(100);
  bs[10] = 1;
  if (bs.count() == 1 && *bs.findFirstSet() == 10)
    std::cout << "Search PASSED" << std::endl;
}

void testIO() {
  std::cout << "\n--- I/O Test ---" << std::endl;
  Bitset bs(128);
  bs[0] = 1;
  bs[127] = 1;
  bs.save("test_bitset.bin");
  Bitset loaded(0);
  loaded.load("test_bitset.bin");
  if (loaded.size() == 128 && loaded[0])
    std::cout << "I/O PASSED" << std::endl;
}

void testIterator() {
  std::cout << "\n--- Iterator Test ---" << std::endl;
  Bitset bs(100);
  bs[10] = 1;
  bs[50] = 1;
  std::vector<size_t> found;
  for (auto i : bs.ones())
    found.push_back(i);
  if (found.size() == 2 && found[0] == 10)
    std::cout << "Ones Iterator PASSED" << std::endl;
}

void testShiftRangeHash() {
  std::cout << "\n--- Shift/Range/Hash Test ---" << std::endl;
  Bitset bs(64);
  bs[0] = 1;
  bs <<= 1;
  if (bs[1])
    std::cout << "Shift PASSED" << std::endl;

  bs.setRange(10, 10, true);
  if (bs[10] && bs[19])
    std::cout << "Range PASSED" << std::endl;

  std::unordered_map<Bitset, int> map;
  map[bs] = 42;
  if (map[bs] == 42)
    std::cout << "Hash PASSED" << std::endl;
}

void testPolish() {
  std::cout << "\n--- Polish Features Test ---" << std::endl;

  // String Ctor
  Bitset s("101"); // 5
  if (s.size() == 3 && s[0] && !s[1] && s[2])
    std::cout << "String Ctor PASSED" << std::endl;
  else
    std::cout << "String Ctor FAILED" << std::endl;

  // To UInt64
  if (s.to_uint64() == 5)
    std::cout << "to_uint64 PASSED" << std::endl;
  else
    std::cout << "to_uint64 FAILED: " << s.to_uint64() << std::endl;

  // Relations
  Bitset a("101");
  Bitset b("111");
  if (a.isSubsetOf(b) && !b.isSubsetOf(a) && a.intersects(b))
    std::cout << "Relations PASSED" << std::endl;
  else
    std::cout << "Relations FAILED" << std::endl;

  // Zeros Iterator
  Bitset z(5); // 00000
  z[2] = 1;    // 00100
  std::cout << "Zeros: ";
  std::vector<size_t> zeros;
  for (auto i : z.zeros()) {
    std::cout << i << " ";
    zeros.push_back(i);
  }
  std::cout << std::endl;
  // Expect 0, 1, 3, 4
  if (zeros.size() == 4 && zeros[2] == 3)
    std::cout << "Zeros Iterator PASSED" << std::endl;
  else
    std::cout << "Zeros Iterator FAILED" << std::endl;
}

int main() {
  std::cout << "Starting All Tests..." << std::endl;
  testProxy();
  testBitwise();
  testSearch();
  testIO();
  testIterator();
  testShiftRangeHash();
  testPolish();
  // Benchmark
  std::cout << "\n--- Performance Benchmark ---" << std::endl;
  Bitset bm(100000000); // 100 million bits
  auto start = std::chrono::high_resolution_clock::now();

  // Dense set
  for (size_t i = 0; i < 100000000; i += 2) {
    bm.set(i);
  }

  // Dense test
  size_t count = 0;
  for (size_t i = 0; i < 100000000; ++i) {
    if (bm.test(i))
      count++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "Bitset Time:         " << diff.count() << " s "
            << "(Check: " << count << ")" << std::endl;
  std::cout << "Bitset Ops/sec:      " << (200.0 / diff.count()) << " M/s"
            << std::endl;

  // std::vector<bool> Benchmark
  std::cout << "\n--- std::vector<bool> Benchmark ---" << std::endl;
  std::vector<bool> vec(100000000); // 100 million bits
  start = std::chrono::high_resolution_clock::now();

  // Dense set
  for (size_t i = 0; i < 100000000; i += 2) {
    vec[i] = true;
  }

  // Dense test
  count = 0;
  for (size_t i = 0; i < 100000000; ++i) {
    if (vec[i])
      count++;
  }

  end = std::chrono::high_resolution_clock::now();
  diff = end - start;
  std::cout << "std::vector<bool> Time:    " << diff.count() << " s "
            << "(Check: " << count << ")" << std::endl;
  std::cout << "std::vector<bool> Ops/sec: " << (200.0 / diff.count()) << " M/s"
            << std::endl;

  std::cout << "\nAll Tests Completed." << std::endl;
  return 0;
}
