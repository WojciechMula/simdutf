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

__m512i avx512_utf8_to_utf32__aux__version3(__m512i utf8) {
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

    /* 1. Reset control bits of continuation bytes and the MSB
          of the leading byte; this makes all bytes unsigned (and
          does not alter ASCII char).
        
        |00dd.dddd|00cc.cccc|00bb.bbbb|0111.0aaa| 4-byte char
        |00??.????|00cc.cccc|00bb.bbbb|0110.aaaa| 3-byte char
        |00??.????|00??.????|00bb.bbbb|010a.aaaa| 2-byte char
        |00??.????|00??.????|00??.????|0aaa.aaaa| ASCII char
         ^^        ^^        ^^        ^
    */
    __m512i values;
    values = _mm512_and_si512(utf8, _mm512_set1_epi32(0x3f3f3f7f));
    
    /* 2. Swap and join fields A-B and C-D

        |0000.cccc|ccdd.dddd|0001.110a|aabb.bbbb| 4-byte char
        |0000.cccc|cc??.????|0001.10aa|aabb.bbbb| 3-byte char
        |0000.????|????.????|0001.0aaa|aabb.bbbb| 2-byte char
        |0000.????|????.????|000a.aaaa|aa??.????| ASCII char */
    values = _mm512_maddubs_epi16(values, _mm512_set1_epi32(0x01400140));

    /* 3. Swap and join field AB & CD

        |0000.0001|110a.aabb|bbbb.cccc|ccdd.dddd| 4-byte char
        |0000.0001|10aa.aabb|bbbb.cccc|cc??.????| 3-byte char
        |0000.0001|0aaa.aabb|bbbb.????|????.????| 2-byte char
        |0000.000a|aaaa.aa??|????.????|????.????| ASCII char */
    values = _mm512_madd_epi16(values, _mm512_set1_epi32(0x00011000));

    /* 4. Get the 4 most significant bits from byte #0 -- these values
          will be used by to distinguish character classes

        |0000.0000|0000.0000|0000.0000|0000.1111|
        |0000.0000|0000.0000|0000.0000|0000.1110|
        |0000.0000|0000.0000|0000.0000|0000.110a|
        |0000.0000|0000.0000|0000.0000|0000.0aaa| */
    const __m512i char_type = _mm512_and_si512(_mm512_srli_epi32(utf8, 4),
                                               _mm512_set1_epi32(0x0000000f));

    /* 5. Shift left the values by variable amounts to reset highest UTF-8 bits 
        |0000000aaaaaaa??????????????????| 
        |aaabbbbbbccccccdddddd00000000000| 4-byte char -- by 11
        |aaaabbbbbbcccccc??????0000000000| 3-byte char -- by 10
        |aaaaabbbbbb????????????000000000| 2-byte char -- by 9
        |aaaaaaa??????????????????0000000| ASCII char -- by 7 */
    {
        // 4 * [7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 9, 9, 10, 11]
        const __m512i shift_left_v3 = _mm512_setr_epi64(
            0x0707070707070707,
            0x0b0a090900000000,
            0x0707070707070707,
            0x0b0a090900000000,
            0x0707070707070707,
            0x0b0a090900000000,
            0x0707070707070707,
            0x0b0a090900000000
        );

        __m512i shift = _mm512_shuffle_epi8(shift_left_v3, char_type);
        shift = _mm512_and_si512(shift, _mm512_set1_epi32(0xff));
        values = _mm512_sllv_epi32(values, shift);
    }

    /* 5. Shift right the values by variable amounts to reset lowest bits
        |0000000aaaaaaa??????????????????| 
        |aaabbbbbbccccccdddddd00000000000| 4-byte char -- by 11
        |aaaabbbbbbcccccc??????0000000000| 3-byte char -- by 16
        |aaaaabbbbbb????????????000000000| 2-byte char -- by 21
        |aaaaaaa??????????????????0000000| ASCII char -- by 25 */
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

    return values;
}


__m512i avx512_utf8_to_utf32__aux__version3(__m512i utf8, __m512i char_type) {
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

    /* 1. Reset control bits of continuation bytes and the MSB
          of the leading byte; this makes all bytes unsigned (and
          does not alter ASCII char).
        
        |00dd.dddd|00cc.cccc|00bb.bbbb|0111.0aaa| 4-byte char
        |00??.????|00cc.cccc|00bb.bbbb|0110.aaaa| 3-byte char
        |00??.????|00??.????|00bb.bbbb|010a.aaaa| 2-byte char
        |00??.????|00??.????|00??.????|0aaa.aaaa| ASCII char
         ^^        ^^        ^^        ^
    */
    __m512i values;
    values = _mm512_and_si512(utf8, _mm512_set1_epi32(0x3f3f3f7f));
    
    /* 2. Swap and join fields A-B and C-D

        |0000.cccc|ccdd.dddd|0001.110a|aabb.bbbb| 4-byte char
        |0000.cccc|cc??.????|0001.10aa|aabb.bbbb| 3-byte char
        |0000.????|????.????|0001.0aaa|aabb.bbbb| 2-byte char
        |0000.????|????.????|000a.aaaa|aa??.????| ASCII char */
    values = _mm512_maddubs_epi16(values, _mm512_set1_epi32(0x01400140));

    /* 3. Swap and join field AB & CD

        |0000.0001|110a.aabb|bbbb.cccc|ccdd.dddd| 4-byte char
        |0000.0001|10aa.aabb|bbbb.cccc|cc??.????| 3-byte char
        |0000.0001|0aaa.aabb|bbbb.????|????.????| 2-byte char
        |0000.000a|aaaa.aa??|????.????|????.????| ASCII char */
    values = _mm512_madd_epi16(values, _mm512_set1_epi32(0x00011000));

    /* 5. Shift left the values by variable amounts to reset highest UTF-8 bits 
        |0000000aaaaaaa??????????????????| 
        |aaabbbbbbccccccdddddd00000000000| 4-byte char -- by 11
        |aaaabbbbbbcccccc??????0000000000| 3-byte char -- by 10
        |aaaaabbbbbb????????????000000000| 2-byte char -- by 9
        |aaaaaaa??????????????????0000000| ASCII char -- by 7 */
    {
        // 4 * [7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 9, 9, 10, 11]
        const __m512i shift_left_v3 = _mm512_setr_epi64(
            0x0707070707070707,
            0x0b0a090900000000,
            0x0707070707070707,
            0x0b0a090900000000,
            0x0707070707070707,
            0x0b0a090900000000,
            0x0707070707070707,
            0x0b0a090900000000
        );

        const __m512i shift = _mm512_shuffle_epi8(shift_left_v3, char_type);
        values = _mm512_sllv_epi32(values, shift);
    }

    /* 5. Shift right the values by variable amounts to reset lowest bits
        |0000000aaaaaaa??????????????????| 
        |aaabbbbbbccccccdddddd00000000000| 4-byte char -- by 11
        |aaaabbbbbbcccccc??????0000000000| 3-byte char -- by 16
        |aaaaabbbbbb????????????000000000| 2-byte char -- by 21
        |aaaaaaa??????????????????0000000| ASCII char -- by 25 */
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

        const __m512i shift = _mm512_shuffle_epi8(shift_right, char_type);
        values = _mm512_srlv_epi32(values, shift);
    }

    return values;
}

namespace {
    __attribute__((__aligned__(64))) uint32_t rotate_left_idx[16][16] = {
        { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
        {15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14},
        {14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13},
        {13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12},
        {12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11},
        {11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10},
        {10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9},
        { 9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8},
        { 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7},
        { 7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6},
        { 6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5},
        { 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4},
        { 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3},
        { 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2},
        { 2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1},
        { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0}
    };
}

__m512i uint32_rotate_left(__m512i v, unsigned amount) {
    const __m512i idx = _mm512_load_si512((const __m512i*)(&rotate_left_idx[amount & 0xf]));
    return _mm512_permutexvar_epi32(idx, v);
}

// RegisterAccumulator accumulates the unpacked UTF-8 words until
// an AVX512 is full and only then calls the conversion routine.
class RegisterAccumulator {
    struct {
        __m512i vector;
        int     count; // active items 0..active     
    } current;

public:
    RegisterAccumulator() {
        current.vector = _mm512_setzero_si512();
        current.count  = 0;
    }

    // input:
    // - value    - 16 x 32-bit values
    // - mask     - which items are active
    // - callback - fun(__m512 utf32, int count) called when at least 16 values got accumulated
    template <typename CALLBACK>
    void add(__m512i utf8, uint16_t valid, CALLBACK callback) {
        const auto     count    = __builtin_popcount(valid);
        const uint32_t new_mask = (uint32_t(1 << count) - 1) << current.count;

        __m512i tmp;

        tmp = _mm512_mask_compress_epi32(_mm512_setzero_si512(), valid, utf8);
        tmp = uint32_rotate_left(tmp, current.count);
        current.vector = _mm512_mask_mov_epi32(current.vector, (new_mask & 0xffff), tmp);

        if (count + current.count < 16) {
            // there are less than 16 items in register, nothing to do
            current.count += count;
            return;
        }

        // the whole vector is full
        callback(current.vector, 16);

        // keep the remining items from `utf8`
        const auto tail = 16 - current.count;
        current.vector = tmp;
        current.count  = count - tail;
    }

    template <typename CALLBACK>
    void flush(CALLBACK callback) {
        if (current.count)
            callback(current.vector, current.count);

        current.count = 0;
    }
};

size_t avx512_utf8_to_utf32__version4(const char* str, size_t len, char16_t* dwords) {
    const char* ptr = str;
    const char* end = ptr + len;

    uint16_t* output = (uint16_t*)dwords;

    if (len > 64) {

        // load 16 bytes: [abcdefghijklmnop]
        __m512i prev = _mm512_broadcast_i64x2(_mm_loadu_si128((const __m128i*)ptr));
        ptr += 16;

        RegisterAccumulator acc;

        auto convert_to_utf32 = [&output](__m512i utf8, int count) {
            // 3. Convert words into UCS-32
            const __m512i utf32 = avx512_utf8_to_utf32__aux__version3(utf8);

            // 4. Store them
            _mm256_storeu_si256((__m256i*)output, _mm512_cvtepi32_epi16(utf32));

            output += count;
        };

        while (ptr + 16 < end) {
            // load next 16 bytes: [qrstu....]
            const __m512i curr = _mm512_broadcast_i64x2(_mm_loadu_si128((const __m128i*)ptr));

            /* 1. Load all possible 4-byte substring into an AVX512 register
                  from bytes a..p (16) and q..t (4). For these bytes
                  we create following 32-bit lanes

                  [abcd|bcde|cdef|defg|efgh|...]
                   ^                          ^
                   byte 0 of reg              byte 63 of reg */

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

            acc.add(input, valid, convert_to_utf32);
            ptr += 16;
        }

        acc.flush(convert_to_utf32);
    }

    // TODO: process the tail

    return output - (uint16_t*)dwords;
}

bool avx512_utf8_structure_validate_16_bytes(__m128i bytes_ap_128, __m128i bytes_qt_128);
bool avx512_utf8_structure_validate_16_bytes(__m128i bytes_ap_128, __m128i bytes_qt_128, uint16_t& leading_bytes, uint16_t& valid_chars);
bool avx512_utf8_structure_validate_16_bytes(const char* ptr, uint16_t& leading_bytes, uint16_t& valid_chars);

bool avx512_utf8_structure_validate_16_bytes(const char* ptr, uint16_t& leading_bytes, uint16_t& valid_chars) {
    // 1. Load leading 16 bytes (a..p)
    const __m128i bytes_ap_128 = _mm_loadu_si128((const __m128i*)ptr);

    // 2. Load the tail 4 bytes (q..t)
    const __m128i bytes_qt_128 = _mm_loadu_si128((const __m128i*)(ptr + 16));

    return avx512_utf8_structure_validate_16_bytes(bytes_ap_128, bytes_qt_128, leading_bytes, valid_chars);
}

bool avx512_utf8_structure_validate_16_bytes(__m128i bytes_ap_128, __m128i bytes_qt_128) {
    uint16_t unused1;
    uint16_t unused2;

    return avx512_utf8_structure_validate_16_bytes(bytes_ap_128, bytes_qt_128, unused1, unused2);
}

bool avx512_utf8_structure_validate_16_bytes(__m128i bytes_ap_128, __m128i bytes_qt_128, uint16_t& leading_bytes, uint16_t& valid_chars) {

    // 3. Produce bitmask of continuation bytes for input in range b..q
    //    Continuation bytes have the following bit pattern: 10xxx.xxxx
    uint32_t continuation;
    {
        const __m128i t0 = _mm_and_si128(_mm_set1_epi8(char(0xc0)), bytes_ap_128);
        const __m128i t1 = _mm_cmpeq_epi8(_mm_set1_epi8(char(0x80)), t0);
        continuation = _mm_movemask_epi8(t1);
    }
    {
        const __m128i t0 = _mm_and_si128(_mm_set1_epi8(char(0xc0)), bytes_qt_128);
        const __m128i t1 = _mm_cmpeq_epi8(_mm_set1_epi8(char(0x80)), t0);
        continuation |= uint32_t(_mm_movemask_epi8(t1)) << 16;
    }

    // 4. Populate the continuation bitmask
    __m512i v_continuation = _mm512_set1_epi32(continuation);

    // 5. Move 4-bit sub-words to proper positions
    v_continuation = _mm512_srlv_epi32(v_continuation, _mm512_setr_epi32(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ));

    // 6. Get expected mask based on character class
    __m128i mask;
    constexpr uint8_t mask_continuation = 0x00;
    constexpr uint8_t mask_ascii   = 0x01;
    constexpr uint8_t mask_2_bytes = 0x03;
    constexpr uint8_t mask_3_bytes = 0x07;
    constexpr uint8_t mask_4_bytes = 0x0f;

    const __m128i mask_lookup = _mm_setr_epi8(
        /* 0000 */ mask_ascii,
        /* 0001 */ mask_ascii,
        /* 0010 */ mask_ascii,
        /* 0011 */ mask_ascii,
        /* 0100 */ mask_ascii,
        /* 0101 */ mask_ascii,
        /* 0110 */ mask_ascii,
        /* 0111 */ mask_ascii,
        /* 1000 */ mask_continuation,
        /* 1001 */ mask_continuation,
        /* 1010 */ mask_continuation,
        /* 1011 */ mask_continuation,
        /* 1100 */ mask_2_bytes,
        /* 1101 */ mask_2_bytes,
        /* 1110 */ mask_3_bytes,
        /* 1111 */ mask_4_bytes
    );
    {
        const __m128i t0 = _mm_srli_epi16(bytes_ap_128, 4);
        const __m128i t1 = _mm_and_si128(t0, _mm_set1_epi8(0x0f));
        mask = _mm_shuffle_epi8(mask_lookup, t1);
    }

    __m128i expected = mask;
    expected = _mm_srli_epi16(expected, 1);
    expected = _mm_and_si128(expected, _mm_set1_epi8(0x0f));

    // 7. Compare the expected continuation mask with the actual mask
    const __m128i t0 = _mm512_cvtepi32_epi8(v_continuation);
    const __m128i t1 = _mm_and_si128(mask, t0);
    const __m128i valid = _mm_cmpeq_epi8(t1, expected);

    // 8. Prepare bitmask for leading bytes and valid chars
    leading_bytes = ~uint16_t(_mm_movemask_epi8(_mm_cmpeq_epi8(mask, _mm_setzero_si128())));
    valid_chars = _mm_movemask_epi8(valid);

    return (leading_bytes | valid_chars);
    return (leading_bytes & 0x1) // detect strings starting with continuation bytes
       and (leading_bytes == valid_chars);
}



/*
    Valid ranges:
                        min         max
    ASCII               0x00'0000   0x00'007f
    2-byte chars        0x00'0080   0x00'07ff
    3-byte chars        0x00'0800   0x00'ffff
    4-byte chars        0x01'0000   0x10'ffff

    Excluded range for surrogate pairs:

                        0x00'd800   0x00'dfff
*/
__mmask16 avx512_utf8_validate_ranges(__m512i char_class, __m512i decoded) {
    // 1. Build minimum value
    //    We start with the min for 4-byte char (single bit set)
    //    and shift it right by the amount depeding on character class.
    __m512i min = _mm512_set1_epi32(0x0100000);

    /** pshufb

        continuation = 0x80
        ascii    = 32       # 0x0100000 >> 32 = 0x0000 - set to zero (vpsrlv guarantees this)
        _2_bytes = 13       # 0x0100000 >> 13 = 0x0800
        _3_bytes = 9        # 0x0100000 >> 9  = 0x8000
        _4_bytes = 0        # 0x0100000                - keep intact

        min_shifts = 4 * [
            ascii, # 0000
            ascii, # 0001
            ascii, # 0010
            ascii, # 0011
            ascii, # 0100
            ascii, # 0101
            ascii, # 0110
            ascii, # 0111
            continuation, # 1000
            continuation, # 1001
            continuation, # 1010
            continuation, # 1011
            _2_bytes, # 1100
            _2_bytes, # 1101
            _3_bytes, # 1110
            _4_bytes, # 1111
        ]
    */
    const __m512i min_shifts = _mm512_setr_epi64(
        0x2020202020202020,
        0x00090d0d80808080,
        0x2020202020202020,
        0x00090d0d80808080,
        0x2020202020202020,
        0x00090d0d80808080,
        0x2020202020202020,
        0x00090d0d80808080
    );

    {
        const __m512i shift = _mm512_shuffle_epi8(min_shifts, char_class);
        min = _mm512_srlv_epi32(min, shift);
    }

    // 2. Build maximum value
    //    We start with the max for 4-byte char and shift it **left** by the amount
    //    depeding on character class. Then the negated-and-shifted mask is used
    //    to reset higher order bits: max andnot (max << k)
    __m512i max = _mm512_set1_epi32(0x010ffff);

    /** pshufb

        continuation = 0x80
        ascii    = 7        # 0x010ffff andnot 0x087fff80 = 0x0000007f
        _2_bytes = 11       # 0x010ffff andnot 0x87fff800 = 0x000007ff
        _3_bytes = 16       # 0x010ffff andnot 0xffff0000 = 0x0000ffff
        _4_bytes = 32       # 0x010ffff andnot 0x00000000 = 0x0010ffff -- keep intact

        max_shifts = 4 * [
            ascii, # 0000
            ascii, # 0001
            ascii, # 0010
            ascii, # 0011
            ascii, # 0100
            ascii, # 0101
            ascii, # 0110
            ascii, # 0111
            continuation, # 1000
            continuation, # 1001
            continuation, # 1010
            continuation, # 1011
            _2_bytes, # 1100
            _2_bytes, # 1101
            _3_bytes, # 1110
            _4_bytes, # 1111
        ]
    */
    const __m512i max_shifts = _mm512_setr_epi64(
        0x0707070707070707,
        0x20100b0b80808080,
        0x0707070707070707,
        0x20100b0b80808080,
        0x0707070707070707,
        0x20100b0b80808080,
        0x0707070707070707,
        0x20100b0b80808080
    );

    {
        const __m512i shift = _mm512_shuffle_epi8(max_shifts, char_class);
        const __m512i shifted = _mm512_sllv_epi32(max, shift);
        max = _mm512_andnot_si512(shifted, max);
    }

    // 4. check range min-max
    __mmask16 r1;
    r1 = _mm512_cmpge_epu32_mask(decoded, min);
    r1 = _mm512_mask_cmple_epu32_mask(r1, decoded, max);

    // 5. check surrogate pairs range
    __mmask16 r2;
    r2 = _mm512_cmpge_epu32_mask(decoded, _mm512_set1_epi32(0xd800));
    r2 = _mm512_mask_cmple_epu32_mask(r2, decoded, _mm512_set1_epi32(0xdfff));

    return _kandn_mask16(r2, r1);
}

size_t avx512_validating_utf8_to_utf32(const char* str, size_t len, char16_t* dwords) {
    const char* ptr = str;
    const char* end = ptr + len;

    char16_t* output = dwords;

    if (len > 64) {

        // load 16 bytes: [abcdefghijklmnop]
        __m128i prev128 = _mm_loadu_si128((const __m128i*)ptr);
        __m512i prev = _mm512_broadcast_i64x2(prev128);
        ptr += 16;

        while (ptr + 16 < end) {
            // load next 16 bytes: [qrstu....]
            const __m128i curr128 = _mm_loadu_si128((const __m128i*)ptr);

            // validate if bytes a..u form valid UTF-8 string
            if (false and !avx512_utf8_structure_validate_16_bytes(prev128, curr128))
                return 0;

            const __m512i curr = _mm512_broadcast_i64x2(curr128);


            /* 1. Load all possible 4-byte substring into an AVX512 register
                  from bytes a..p (16) and q..t (4). For these bytes
                  we create following 32-bit lanes

                  [abcd|bcde|cdef|defg|efgh|...]
                   ^                          ^
                   byte 0 of reg              byte 63 of reg */

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
            prev128 = curr128;

            /*
                2. Classify which words contain valid UTF-8 characters.
                   We test if the 0th byte is not a continuation byte (0b10xxxxxx)

                   Note: can reause bitmask build inside avx512_utf8_structure_validate_16_bytes.
            */
            __mmask16 valid;
            {
                const __m512i t0 = _mm512_and_si512(input, _mm512_set1_epi32(0xc0));
                valid = _mm512_cmpneq_epu32_mask(t0, _mm512_set1_epi32(0x80));
            }
            const int valid_count = __builtin_popcount(valid);

            // 4. Char class
            __m512i char_class;
            char_class = _mm512_srli_epi32(input, 4);
            char_class = _mm512_and_si512(char_class, _mm512_set1_epi32(0x0000000f));
            char_class = _mm512_or_si512(char_class, _mm512_set1_epi32(0x80808000));

            // 3. Convert words into UTF-32
            const __m512i utf32 = avx512_utf8_to_utf32__aux__version3(input, char_class);

            // 4. Validate if UTF-32 chars are in valid ranges
            if (1) {
                const __mmask16 in_range = avx512_utf8_validate_ranges(char_class, utf32);
                if ((valid & in_range) != valid) {
                    return 0;
                }
            }

            // 4. Pack only valid words
            const __m512i out = _mm512_mask_compress_epi32(_mm512_setzero_si512(), valid, utf32);

            // 5. Store them
            _mm256_storeu_si256((__m256i*)output, _mm512_cvtepi32_epi16(out));

            output += valid_count;
            ptr += 16;
        }
    }

    // TODO: process the tail

    return output - dwords;
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
  return avx512_validating_utf8_to_utf32(buf, len, utf16_output);
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
