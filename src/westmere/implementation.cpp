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
__m512i avx512_utf8_to_utf32__aux__version3(__m512i utf8);


size_t avx512_utf8_to_utf32(const char* str, size_t len, char16_t* words) {
    const char* ptr = str;
    const char* end = ptr + len;
    
    char16_t* output = words;

    if (len > 64) {

    __m128i prev128 =_mm_loadu_si128((const __m128i*)ptr);
    __mmask16 ascii = _mm_movemask_epi8(prev128);
    __m512i prev = _mm512_broadcast_i64x2(prev128);
    ptr += 16;

    while (ptr + 16 < end) {
        /* 1. Load all possible 4-byte substring into an AVX512 register
              For example if we have bytes abcdefgh... we create following 32-bit lanes

              [abcd|bcde|cdef|defg|efgh|...]
               ^                          ^
               byte 0 of reg              byte 63 of reg

              It means, we are touching 16+3 = 19 bytes

              XXX: gather is slow, we may consider some scalar code or -- better --
                   use _mm512_permutex2var_epi8()
        */
        const __m128i tmp =_mm_loadu_si128((const __m128i*)ptr);
        __m512i curr = _mm512_broadcast_i64x2(_mm_loadu_si128((const __m128i*)ptr));
        if (ascii == 0) {
            _mm256_storeu_si256((__m256i*)output, _mm256_cvtepu8_epi16(tmp));
            output += 16;
            ptr += 16;
            prev = curr;
            prev128 = tmp;
            ascii = _mm_movemask_epi8(tmp);
            continue;
        }
        
        ascii = _mm_movemask_epi8(tmp);

		const __m512i merged = _mm512_mask_mov_epi32(prev, 0x1000, curr);
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

		const __m512i input = _mm512_shuffle_epi8(merged, expand_ver2);
        prev = curr;

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
        const __m512i utf32 = avx512_utf8_to_utf32__aux__version3(input);

        const __mmask16 surrogate_pairs = _mm512_mask_cmpgt_epu32_mask(valid, utf32, _mm512_set1_epi32(0xffff));
        if (simdutf_likely(surrogate_pairs == 0)) {
            // 4. Pack only valid words
            const __m512i out = _mm512_mask_compress_epi32(_mm512_setzero_si512(), valid, utf32);

            // 5. Store them
            _mm256_storeu_si256((__m256i*)output, _mm512_cvtepi32_epi16(out));

            output += valid_count;
            ptr += 16;
        } else {
            // handle surrogate pairs: in an unefficient way
            uint32_t buf[16];
            _mm512_storeu_si512((__m512i*)buf, utf32);
            int i = 0;
            uint32_t mask = valid;
            for (/**/; mask != 0; mask >>= 1, i++) {
                if (mask & 1) {
                    if (buf[i] < 0xffff)
                        *output++ = buf[i];
                    else {
                        *output++ = (buf[i] >> 10) | 0xd800;
                        *output++ = (buf[i] & 0x3ff) | 0xdc00;
                    }
                }
            }

            ptr += 16;
        }
    }
    }

    // TODO: process the tail

    return output - words;
}

#include "avx512.cpp"
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
  return avx512_validating_utf8_to_utf16(buf, len, utf16_output);
}

simdutf_warn_unused size_t implementation::convert_valid_utf8_to_utf16(const char* input, size_t size,
    char16_t* utf16_output) const noexcept {
  return avx512_utf8_to_utf32(input, size, utf16_output);
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
