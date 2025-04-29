// file included directly
enum class Validation {
  None,
  Boolean,
  Result,
};

template <endianness big_endian, Validation validation>
result icelake_convert_utf16_to_latin1(const char16_t *src, size_t len,
                                       char *latin1_output) {

  const char16_t *beg = src;
  const char16_t *end = src + len;

  const __m512i zero = _mm512_setzero_si512();
  const __m512i gather_byte0 = _mm512_set_epi8(
      126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98,
      96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62,
      60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26,
      24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  const __m512i gather_byte1 = _mm512_set_epi8(
      127, 125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99,
      97, 95, 93, 91, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67, 65, 63,
      61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27,
      25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);

  constexpr size_t unrolled_size = 2 * 32;
  const size_t blocks = len / unrolled_size;

  for (size_t i = 0; i < blocks; i++) {
    const auto v0 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + 0 * 32));
    const auto v1 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + 1 * 32));

    const auto latin1 = _mm512_permutex2var_epi8(
        v0, big_endian ? gather_byte1 : gather_byte0, v1);

    if (validation != Validation::None) {
      const auto msb = _mm512_permutex2var_epi8(
          v0, big_endian ? gather_byte0 : gather_byte1, v1);
      const auto err = _mm512_cmpgt_epu8_mask(msb, zero);
      if (err != 0) {
        switch (validation) {
        case Validation::None:
          return result(error_code::OTHER, 0);

        case Validation::Boolean:
          return result(error_code::TOO_LARGE, 0);

        case Validation::Result:
          return result(error_code::TOO_LARGE,
                        src - beg + trailing_zeroes(err));
        }
      }
    }

    _mm512_storeu_si512(reinterpret_cast<__m512i *>(latin1_output), latin1);

    src += 2 * 32;
    latin1_output += 64;
  }

  const auto tail = len - blocks * unrolled_size;

  switch (validation) {
  case Validation::None:
    scalar::utf16_to_latin1::convert_valid<big_endian>(src, tail,
                                                       latin1_output);
    return result(error_code::SUCCESS, len);

  case Validation::Boolean: {
    const auto ret =
        scalar::utf16_to_latin1::convert<big_endian>(src, tail, latin1_output);
    if (ret == 0) {
      return result(error_code::TOO_LARGE, 0);
    } else {
      return result(error_code::SUCCESS, len);
    }
  }

  case Validation::Result: {
    auto ret = scalar::utf16_to_latin1::convert_with_errors<big_endian>(
        src, tail, latin1_output);

    // Since we load & store the same amount of code points in the vectorized code,
    // thus we don't need to check whether `res` is OK or not.
    ret.count += blocks * unrolled_size;
    return ret;
  }
  }
}
