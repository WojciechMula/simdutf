#include <utility>

#include <immintrin.h>

#include "simdutf/westmere/begin.h"
namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {
namespace {
#ifndef SIMDUTF_WESTMERE_H
#error "westmere.h must be included"
#endif
using namespace simd;


// AVX512 code
__m512i avx512_utf8_to_utf32__aux(__m512i utf8);

template <typename EXPANDFN>
size_t avx512_utf8_to_utf32(EXPANDFN expandfn, const char* str, size_t len, char16_t* words) {
    const char* ptr = str;
    const char* end = ptr + len;
    
    char16_t* output = words;

    while (ptr + 16 + 3 < end) {
        /* 1. Load all possible 4-byte substring into an AVX512 register
              For example if we have bytes abcdefgh... we create following 32-bit lanes

              [abcd|bcde|cdef|defg|efgh|...]
               ^                          ^
               byte 0 of reg              byte 63 of reg

              It means, we are touching 16+3 = 19 bytes

              XXX: gather is slow, we may consider some scalar code or -- better --
                   use _mm512_permutex2var_epi8()
        */
        const __m512i input = expandfn(ptr);

        /*
            2. Classify which words contain valid UTF-8 characters.
               We test if the 0th byte is not a continuation byte (0b10xxxxxx) */
        __mmask16 valid;
        {
            const __m512i t0 = _mm512_and_si512(input, _mm512_set1_epi32(0xc0));
            valid = _mm512_cmpneq_epu32_mask(t0, _mm512_set1_epi32(0x80));
        }
        const int valid_count = __builtin_popcount(valid);

        // 3. Convert words into UCS-32
        //    (XXX: would passing `valid` mask speed things up?)
        const __m512i utf32 = avx512_utf8_to_utf32__aux(input);

        // 4. Pack only valid words
        const __m512i out = _mm512_mask_compress_epi32(_mm512_setzero_si512(), valid, utf32);

        // 5. Store them
        _mm256_storeu_si256((__m256i*)output, _mm512_cvtepi32_epi16(out));

        output += valid_count;
        ptr += 16;
    }

    // TODO: process the tail

    return output - words;
}

__m512i avx512_expand__version2(const char* ptr) {
    // load bytes 0..15 (16)
    const __m128i tmp0 = _mm_loadu_si128((const __m128i*)ptr);
    const __m512i t0 = _mm512_broadcast_i64x2(tmp0);

    // load bytes 16..19 (4)
    const uint32_t tmp1 = *(uint32_t*)(ptr + 16);
    const __m512i t1 = _mm512_set1_epi32(tmp1);

    // In the last lane we need bytes 13..19, so we're placing
    // 32-bit word from t1 at 0th position of the lane

    const __m512i t2 = _mm512_mask_mov_epi32(t0, 0x1000, t1);

    /*
    lane{0,1,2} have got bytes: [  0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15]
    lane3 has got bytes:        [ 16, 17, 18, 19,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15]
    expand_ver2 = [
        # lane 0:
        0, 1, 2, 3,
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6,
        # lane 1:
        4, 5, 6, 7,
        5, 6, 7, 8,
        6, 7, 8, 9,
        7, 8, 9, 10,
        # lane 2:
         8,  9, 10, 11,
         9, 10, 11, 12,
        10, 11, 12, 13,
        11, 12, 13, 14,

        # lane 3 order: 13, 14, 15, 16 14, 15, 16, 17, 15, 16, 17, 18, 16, 17, 18, 19
        12, 13, 14, 15,
        13, 14, 15,  0,
        14, 15,  0,  1,
        15,  0,  1,  2,
    ] */
    const __m512i expand_ver2 = _mm512_setr_epi64(
        0x0403020103020100,
        0x0605040305040302,
        0x0807060507060504,
        0x0a09080709080706,
        0x0c0b0a090b0a0908,
        0x0e0d0c0b0d0c0b0a,
        0x000f0e0d0f0e0d0c,
        0x0201000f01000f0e
    );

    return _mm512_shuffle_epi8(t2, expand_ver2);
}

__m512i avx512_utf8_to_utf32__aux(__m512i utf8) {
    /*
        Input:
        - utf8: bytes stored at separate 32-bit words
        - valid: which words have valid UTF-8 characters

        Bit layout of single word. We show 4 cases for each possible
        UTF-8 character encoding. The `?` denotes bits we must not
        assume their value.

        |10dd.dddd|10cc.cccc|10bb.bbbb|1111.0aaa| 4-byte char
        |????.????|10cc.cccc|10bb.bbbb|1110.aaaa| 3-byte char
        |????.????|????.????|10bb.bbbb|110a.aaaa| 2-byte char
        |????.????|????.????|????.????|0aaa.aaaa| ASCII char
          byte 3    byte 2    byte 1     byte 0
    */

    /* 1. Swap bytes within the 32-bit lanes

        |1111.0aaa|10bb.bbbb|10cc.cccc|10dd.dddd|
        |1110.aaaa|10bb.bbbb|10cc.cccc|????.????|
        |110a.aaaa|10bb.bbbb|????.????|????.????|
        |0aaa.aaaa|????.????|????.????|????.????| */
    __m512i values;
    // 4 * [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]
    const __m512i bswap_lookup = _mm512_setr_epi64(
        0x0405060700010203,
        0x0c0d0e0f08090a0b,
        0x0405060700010203,
        0x0c0d0e0f08090a0b,
        0x0405060700010203,
        0x0c0d0e0f08090a0b,
        0x0405060700010203,
        0x0c0d0e0f08090a0b
    );

    values = _mm512_shuffle_epi8(utf8, bswap_lookup);

    /* 2. Shift continuation byte B (byte #2) left by 2 and merge with the value

        |1111.0aaa|bbbb.bb??|10cc.cccc|10dd.dddd|
        |1110.aaaa|bbbb.bb??|10cc.cccc|????.????|
        |110a.aaaa|bbbb.bb??|????.????|????.????|
        |0aaa.aaaa|????.????|????.????|????.????| */
    const uint8_t merge_function = 0xca; // ((mask and value) or (not mask and field))
    {
        const __m512i mask = _mm512_set1_epi32(0x00fc0000);
        __m512i B = _mm512_slli_epi32(values, 2);
        values = _mm512_ternarylogic_epi32(mask, B, values, merge_function);
    }

    /* 3. Shift continuation byte C (byte #1) left by 4 and merge with the value

        |1111.0aaa|bbbb.bbcc|cccc.????|10dd.dddd|
        |1110.aaaa|bbbb.bbcc|cccc.????|????.????|
        |110a.aaaa|bbbb.bb??|????.????|????.????|
        |0aaa.aaaa|????.????|????.????|????.????| */
    {
        const __m512i mask = _mm512_set1_epi32(0x0003f000);
        __m512i C = _mm512_slli_epi32(values, 4);
        values = _mm512_ternarylogic_epi32(mask, C, values, merge_function);
    }

    /* 4. Shift continuation byte D (byte #0) left by 6 and merge with the value

        |1111.0aaa|bbbb.bbcc|cccc.dddd|dd??.????|
        |1110.aaaa|bbbb.bbcc|cccc.????|????.????|
        |110a.aaaa|bbbb.bb??|????.????|????.????|
        |0aaa.aaaa|????.????|????.????|????.????| */
    {
        const __m512i mask = _mm512_set1_epi32(0x00000fc0);
        __m512i D = _mm512_slli_epi32(values, 6);
        values = _mm512_ternarylogic_epi32(mask, D, values, merge_function);
    }

    /* 5. Get the 4 most significant bits from byte #0 -- these values
          will be used by to distinguish character classes

        |0000.0000|0000.0000|0000.0000|0000.1111|
        |0000.0000|0000.0000|0000.0000|0000.1110|
        |0000.0000|0000.0000|0000.0000|0000.110a|
        |0000.0000|0000.0000|0000.0000|0000.0aaa| */
    const __m512i char_type = _mm512_and_si512(_mm512_srli_epi32(utf8, 4),
                                               _mm512_set1_epi32(0x0000000f));

    /* 6. Shift left the values by variable amounts to reset highest UTF-8 bits 

        |aaab.bbbb|bccc.cccd|dddd.d???|???0.0000| shift left by 5
        |aaaa.bbbb|bbcc.cccc|????.????|????.0000| shift left by 4
        |aaaa.abbb|bbb?.????|????.????|????.?000| shift left by 3
        |aaaa.aaa?|????.????|????.????|????.???0| shift left by 1 */
    {
        // 4 * [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 4, 5]
        const __m512i shift_left = _mm512_setr_epi64(
            0x0101010101010101,
            0x0504030300000000,
            0x0101010101010101,
            0x0504030300000000,
            0x0101010101010101,
            0x0504030300000000,
            0x0101010101010101,
            0x0504030300000000
        );

        __m512i shift = _mm512_shuffle_epi8(shift_left, char_type);
        shift = _mm512_and_si512(shift, _mm512_set1_epi32(0xff));
        values = _mm512_sllv_epi32(values, shift);
    }

    /* 7. Shift right the values by variable amounts to reset lowest bits

        |0000.0000|000a.aabb|bbbb.cccc|ccdd.dddd| shift right by 11
        |0000.0000|0000.0000|aaaa.bbbb|bbcc.cccc| shift right by 16
        |0000.0000|0000.0000|0000.0aaa|aabb.bbbb| shift right by 21
        |0000.0000|0000.0000|0000.0000|0aaa.aaaa| shift right by 25 */
    {
        // 4 * [25, 25, 25, 25, 25, 25, 25, 25, 0, 0, 0, 0, 21, 21, 16, 11]
        const __m512i shift_right = _mm512_setr_epi64(
            0x1919191919191919,
            0x0b10151500000000,
            0x1919191919191919,
            0x0b10151500000000,
            0x1919191919191919,
            0x0b10151500000000,
            0x1919191919191919,
            0x0b10151500000000
        );

        __m512i shift = _mm512_shuffle_epi8(shift_right, char_type);
        shift = _mm512_and_si512(shift, _mm512_set1_epi32(0xff));
        values = _mm512_srlv_epi32(values, shift);
    }

    /*
        Note about shifts: since the left shift max is 5, and right shift max is 25,
        we may pack them in single lookup: 3 bits for left, and 5 bits for right shift.
        Then we would need only one _mm512_shuffle_epi8(). But still we need
        _mm512_and_si512() to convert the lookup result into 32-bit numbers. It would
        look like:

        __m512i shift = _mm512_shuffle_epi8(shift_left_right, char_type);

        __m512i shift_left = _mm512_and_si512(shift, _mm512_set1_epi32(0x03));
        __m512i shift_right = _mm512_srli_epi32(shift, 3);
        shift_right _mm512_and_si512(shift_right, _mm512_set1_epi32(0x1f));

        values = _mm512_sllv_epi32(values, shift_left);
        values = _mm512_srlv_epi32(values, shift_right);

        We would replace _mm512_shuffle_epi8() with _mm512_srli_epi32().
    */

    return values;
}
// AVX512 end


simdutf_really_inline bool is_ascii(const simd8x64<uint8_t>& input) {
  return input.reduce_or().is_ascii();
}

simdutf_unused simdutf_really_inline simd8<bool> must_be_continuation(const simd8<uint8_t> prev1, const simd8<uint8_t> prev2, const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_second_byte = prev1.saturating_sub(0b11000000u-1); // Only 11______ will be > 0
  simd8<uint8_t> is_third_byte  = prev2.saturating_sub(0b11100000u-1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte = prev3.saturating_sub(0b11110000u-1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_second_byte | is_third_byte | is_fourth_byte) > int8_t(0);
}

simdutf_really_inline simd8<bool> must_be_2_3_continuation(const simd8<uint8_t> prev2, const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_third_byte  = prev2.saturating_sub(0b11100000u-1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte = prev3.saturating_sub(0b11110000u-1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_third_byte | is_fourth_byte) > int8_t(0);
}

#include "westmere/sse_convert_utf8_to_utf16.cpp"
#include "westmere/sse_validate_utf16le.cpp"
#include "westmere/sse_convert_utf16_to_utf8.cpp"

// UTF-16 => UTF-8 conversion

} // unnamed namespace
} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

#include "generic/buf_block_reader.h"
#include "generic/utf8_validation/utf8_lookup4_algorithm.h"
#include "generic/utf8_validation/utf8_validator.h"
// transcoding from UTF-8 to UTF-16
#include "generic/utf8_to_utf16/valid_utf8_to_utf16.h"
#include "generic/utf8_to_utf16/utf8_to_utf16.h"
// other functions
#include "generic/utf8.h"
#include "generic/utf16.h"
//
// Implementation-specific overrides
//

namespace simdutf {
namespace SIMDUTF_IMPLEMENTATION {

simdutf_warn_unused bool implementation::validate_utf8(const char *buf, size_t len) const noexcept {
  return westmere::utf8_validation::generic_validate_utf8(buf, len);
}

simdutf_warn_unused bool implementation::validate_utf16(const char16_t *buf, size_t len) const noexcept {
  const char16_t* tail = sse_validate_utf16le(buf, len);
  if (tail) {
    return scalar::utf16::validate(tail, len - (tail - buf));
  } else {
    return false;
  }
}

simdutf_warn_unused size_t implementation::convert_utf8_to_utf16(const char* buf, size_t len, char16_t* utf16_output) const noexcept {
  utf8_to_utf16::validating_transcoder converter;
  return converter.convert(buf, len, utf16_output);
}

simdutf_warn_unused size_t implementation::convert_valid_utf8_to_utf16(const char* input, size_t size,
    char16_t* utf16_output) const noexcept {
  return avx512_utf8_to_utf32(avx512_expand__version2, input, size, utf16_output);
}

simdutf_warn_unused size_t implementation::convert_utf16_to_utf8(const char16_t* buf, size_t len, char* utf8_output) const noexcept {
  std::pair<const char16_t*, char*> ret = sse_convert_utf16_to_utf8(buf, len, utf8_output);
  if (ret.first == nullptr) { return 0; }
  size_t saved_bytes = ret.second - utf8_output;
  if (ret.first != buf + len) {
    const size_t scalar_saved_bytes = scalar::utf16_to_utf8::convert(
                                        ret.first, len - (ret.first - buf), ret.second);
    if (scalar_saved_bytes == 0) { return 0; }
    saved_bytes += scalar_saved_bytes;
  }
  return saved_bytes;
}

simdutf_warn_unused size_t implementation::convert_valid_utf16_to_utf8(const char16_t* buf, size_t len, char* utf8_output) const noexcept {
  return convert_utf16_to_utf8(buf, len, utf8_output);
}

simdutf_warn_unused size_t implementation::count_utf16(const char16_t * input, size_t length) const noexcept {
  return utf16::count_code_points(input, length);
}

simdutf_warn_unused size_t implementation::count_utf8(const char * input, size_t length) const noexcept {
  return utf8::count_code_points(input, length);
}

simdutf_warn_unused size_t implementation::utf8_length_from_utf16(const char16_t * input, size_t length) const noexcept {
  return utf16::utf8_length_from_utf16(input, length);
}

simdutf_warn_unused size_t implementation::utf16_length_from_utf8(const char * input, size_t length) const noexcept {
  return utf8::utf16_length_from_utf8(input, length);
}

} // namespace SIMDUTF_IMPLEMENTATION
} // namespace simdutf

#include "simdutf/westmere/end.h"
