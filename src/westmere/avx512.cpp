namespace {
    const __m512i v_3f3f_3f7f = _mm512_set1_epi32(0x3f3f3f7f);
    const __m512i v_3f3f_3f00 = _mm512_set1_epi32(0x3f3f3f00);
    const __m512i v_0140_0140 = _mm512_set1_epi32(0x01400140);
    const __m512i v_0001_1000 = _mm512_set1_epi32(0x00011000);
    const __m512i v_0001_0000 = _mm512_set1_epi32(0x00010000);
    const __m512i v_0010_ffff = _mm512_set1_epi32(0x0010ffff);
    const __m512i v_ffff_f800 = _mm512_set1_epi32(0xfffff800);
    const __m512i v_0000_d800 = _mm512_set1_epi32(0xd800);
    const __m512i v_0000_000f = _mm512_set1_epi32(0x0f);
    const __m512i v_8080_8000 = _mm512_set1_epi32(0x80808000);
    const __m512i v_0000_00c0 = _mm512_set1_epi32(0xc0);
    const __m512i v_0000_0080 = _mm512_set1_epi32(0x80);
}


// included directly

// This is slightly modified verison3.

/*
    32-bit lanes in `char_class` have form 0x8080800N, where N is 4 higest
    bits from the leading byte; 0x80 resets corresponding bytes during pshufb.
*/
__m512i avx512_utf8_to_utf32__aux__version5(__m512i char_class, __m512i utf8) {
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

    /* 1. Reset higher bits in the leading bytes and two MSB in
          continuation bytes

        |00dd.dddd|00cc.cccc|00bb.bbbb|0000.0aaa| 4-byte char
         ^^        ^^        ^^        ^^^^
        |00??.????|00cc.cccc|00bb.bbbb|0000.aaaa| 3-byte char
         ^^        ^^        ^^        ^^^^
        |00??.????|00??.????|00bb.bbbb|000a.aaaa| 2-byte char
         ^^        ^^        ^^        ^^^
        |00??.????|00??.????|00??.????|0aaa.aaaa| ASCII char
         ^^        ^^        ^^        ^
    */
    /** pshufb

        continuation = 0
        ascii    = 0x7f
        _2_bytes = 0x1f
        _3_bytes = 0x0f
        _4_bytes = 0x0f # keep 4th byte (it's 0 for valid UTF-8,
                        # if 1 then we detect invalid values)

        mask_leading_bytes = 4 * [ # mask for byte 0
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
        ] */
    __m512i values;

    const __m512i mask_leading_bytes = _mm512_setr_epi64(
        0x7f7f7f7f7f7f7f7f,
        0x0f0f1f1f00000000,
        0x7f7f7f7f7f7f7f7f,
        0x0f0f1f1f00000000,
        0x7f7f7f7f7f7f7f7f,
        0x0f0f1f1f00000000,
        0x7f7f7f7f7f7f7f7f,
        0x0f0f1f1f00000000
    );

    {
        __m512i mask;
        mask = _mm512_shuffle_epi8(mask_leading_bytes, char_class);

        // values = utf8 & (mask | v_3f3f_3f00)
        values = _mm512_ternarylogic_epi32(utf8, mask, v_3f3f_3f00, 0xe0);
    }

    /* 2. Swap and join fields A-B and C-D

        |0000.cccc|ccdd.dddd|0000.000a|aabb.bbbb| 4-byte char
        |0000.cccc|cc??.????|0000.00aa|aabb.bbbb| 3-byte char
        |0000.????|????.????|0000.0aaa|aabb.bbbb| 2-byte char
        |0000.????|????.????|000a.aaaa|aa??.????| ASCII char */
    values = _mm512_maddubs_epi16(values, v_0140_0140);

    /* 3. Swap and join field AB & CD

        |0000.0000|000a.aabb|bbbb.cccc|ccdd.dddd| 4-byte char
        |0000.0000|00aa.aabb|bbbb.cccc|cc??.????| 3-byte char
        |0000.0000|0aaa.aabb|bbbb.????|????.????| 2-byte char
        |0000.000a|aaaa.aa??|????.????|????.????| ASCII char */
    values = _mm512_madd_epi16(values, v_0001_1000);

    /* 4. Shift right the values by variable amounts to reset lowest bits

        |0000.0000|000a.aabb|bbbb.cccc|ccdd.dddd| 4-byte char -- no change
        |0000.0000|0000.0000|aaaa.bbbb|bbcc.cccc| 3-byte char -- shift by 6
        |0000.0000|0000.0000|0000.0aaa|aabb.bbbb| 2-byte char -- shift by 12
        |0000.0000|0000.0000|0000.0000|0aaa.aaaa| ASCII char  -- shift by 18
    */
    {
    /** pshufb

        continuation = 0
        ascii    = 18
        _2_bytes = 12
        _3_bytes = 6
        _4_bytes = 0

        shift_right = 4 * [
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
        ] */
        const __m512i shift_right = _mm512_setr_epi64(
            0x1212121212121212,
            0x00060c0c00000000,
            0x1212121212121212,
            0x00060c0c00000000,
            0x1212121212121212,
            0x00060c0c00000000,
            0x1212121212121212,
            0x00060c0c00000000
        );

        const __m512i shift = _mm512_shuffle_epi8(shift_right, char_class);
        values = _mm512_srlv_epi32(values, shift);
    }

    return values;
}

__m512i avx512_utf8_to_utf32__aux__version5(__m512i utf8) {
    /* 1. Classify leading bytes */
    __m512i char_class;
    char_class = _mm512_srli_epi32(utf8, 4);
    char_class = _mm512_and_si512(char_class, v_0000_000f);
    char_class = _mm512_or_si512(char_class, v_8080_8000);

    return avx512_utf8_to_utf32__aux__version5(char_class, utf8);
}


int utf8_decode(const char* bytes, uint32_t& val) {
    const uint8_t b0 = bytes[0];
    if ((b0 & 0xf8) == 0xf0) {
        val = (b0 & 0x07);
        val = (val << 6) & (bytes[1] & 0x3f);
        val = (val << 6) & (bytes[2] & 0x3f);
        val = (val << 6) & (bytes[3] & 0x3f);
        return 4;
    }

    if ((b0 & 0xf0) == 0xe0) {
        val = (b0 & 0x0f);
        val = (val << 6) & (bytes[1] & 0x3f);
        val = (val << 6) & (bytes[2] & 0x3f);
        return 3;
    }

    if ((b0 & 0xe0) == 0xc0) {
        val = (b0 & 0x3f);
        val = (val << 6) & (bytes[1] & 0x3f);
        return 2;
    }

    val = b0;
    return 1;
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
    values = _mm512_and_si512(utf8, v_3f3f_3f7f);

    /* 2. Swap and join fields A-B and C-D

        |0000.cccc|ccdd.dddd|0001.110a|aabb.bbbb| 4-byte char
        |0000.cccc|cc??.????|0001.10aa|aabb.bbbb| 3-byte char
        |0000.????|????.????|0001.0aaa|aabb.bbbb| 2-byte char
        |0000.????|????.????|000a.aaaa|aa??.????| ASCII char */
    values = _mm512_maddubs_epi16(values, v_0140_0140);

    /* 3. Swap and join field AB & CD

        |0000.0001|110a.aabb|bbbb.cccc|ccdd.dddd| 4-byte char
        |0000.0001|10aa.aabb|bbbb.cccc|cc??.????| 3-byte char
        |0000.0001|0aaa.aabb|bbbb.????|????.????| 2-byte char
        |0000.000a|aaaa.aa??|????.????|????.????| ASCII char */
    values = _mm512_madd_epi16(values, v_0001_1000);

    /* 4. Get the 4 most significant bits from byte #0 -- these values
          will be used by to distinguish character classes

        |0000.0000|0000.0000|0000.0000|0000.1111|
        |0000.0000|0000.0000|0000.0000|0000.1110|
        |0000.0000|0000.0000|0000.0000|0000.110a|
        |0000.0000|0000.0000|0000.0000|0000.0aaa| */
    __m512i char_class;
    char_class = _mm512_srli_epi32(utf8, 4);
    char_class = _mm512_and_si512(char_class, v_0000_000f);
    char_class = _mm512_or_si512(char_class, v_8080_8000);

    /* 5. Shift left the values by variable amounts to reset highest UTF-8 bits
        |aaab.bbbb|bccc.cccd|dddd.d000|0000.0000| 4-byte char -- by 11
        |aaaa.bbbb|bbcc.cccc|????.??00|0000.0000| 3-byte char -- by 10
        |aaaa.abbb|bbb?.????|????.???0|0000.0000| 2-byte char -- by 9
        |aaaa.aaa?|????.????|????.????|?000.0000| ASCII char -- by 7 */
    {
        /** pshufb

        continuation = 0
        ascii    = 7
        _2_bytes = 9
        _3_bytes = 10
        _4_bytes = 11

        shift_left_v3 = 4 * [
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
        ] */
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

        const __m512i shift = _mm512_shuffle_epi8(shift_left_v3, char_class);
        values = _mm512_sllv_epi32(values, shift);
    }

    /* 5. Shift right the values by variable amounts to reset lowest bits
        |aaab.bbbb|bccc.cccd|dddd.d000|0000.0000| 4-byte char -- by 11
        |aaaa.bbbb|bbcc.cccc|????.??00|0000.0000| 3-byte char -- by 16
        |aaaa.abbb|bbb?.????|????.???0|0000.0000| 2-byte char -- by 21
        |aaaa.aaa?|????.????|????.????|?000.0000| ASCII char -- by 25 */
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

        const __m512i shift = _mm512_shuffle_epi8(shift_right, char_class);
        values = _mm512_srlv_epi32(values, shift);
    }

    return values;
}


__m512i avx512_utf8_to_utf32__aux__version3(__m512i char_class, __m512i utf8) {
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
    values = _mm512_and_si512(utf8, v_3f3f_3f7f);

    /* 2. Swap and join fields A-B and C-D

        |0000.cccc|ccdd.dddd|0001.110a|aabb.bbbb| 4-byte char
        |0000.cccc|cc??.????|0001.10aa|aabb.bbbb| 3-byte char
        |0000.????|????.????|0001.0aaa|aabb.bbbb| 2-byte char
        |0000.????|????.????|000a.aaaa|aa??.????| ASCII char */
    values = _mm512_maddubs_epi16(values, v_0140_0140);

    /* 3. Swap and join field AB & CD

        |0000.0001|110a.aabb|bbbb.cccc|ccdd.dddd| 4-byte char
        |0000.0001|10aa.aabb|bbbb.cccc|cc??.????| 3-byte char
        |0000.0001|0aaa.aabb|bbbb.????|????.????| 2-byte char
        |0000.000a|aaaa.aa??|????.????|????.????| ASCII char */
    values = _mm512_madd_epi16(values, v_0001_1000);

    /* 4. Shift left the values by variable amounts to reset highest UTF-8 bits
        |aaab.bbbb|bccc.cccd|dddd.d000|0000.0000| 4-byte char -- by 11
        |aaaa.bbbb|bbcc.cccc|????.??00|0000.0000| 3-byte char -- by 10
        |aaaa.abbb|bbb?.????|????.???0|0000.0000| 2-byte char -- by 9
        |aaaa.aaa?|????.????|????.????|?000.0000| ASCII char -- by 7 */
    {
        /** pshufb

        continuation = 0
        ascii    = 7
        _2_bytes = 9
        _3_bytes = 10
        _4_bytes = 11

        shift_left_v3 = 4 * [
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
        ] */
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

        const __m512i shift = _mm512_shuffle_epi8(shift_left_v3, char_class);
        values = _mm512_sllv_epi32(values, shift);
    }

    /* 5. Shift right the values by variable amounts to reset lowest bits
        |aaab.bbbb|bccc.cccd|dddd.d000|0000.0000| 4-byte char -- by 11
        |aaaa.bbbb|bbcc.cccc|????.??00|0000.0000| 3-byte char -- by 16
        |aaaa.abbb|bbb?.????|????.???0|0000.0000| 2-byte char -- by 21
        |aaaa.aaa?|????.????|????.????|?000.0000| ASCII char -- by 25 */
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

        const __m512i shift = _mm512_shuffle_epi8(shift_right, char_class);
        values = _mm512_srlv_epi32(values, shift);
    }

    return values;
}


uint32_t continuation_bytes(__m128i utf8bytes) {
    const __m128i t0 = _mm_and_si128(_mm_set1_epi8(char(0xc0)), utf8bytes);
    const __m128i t1 = _mm_cmpeq_epi8(_mm_set1_epi8(char(0x80)), t0);
    return _mm_movemask_epi8(t1);
}


__mmask16 avx512_utf8_validate_ranges_masked(__m512i char_class, __m512i utf32) {
    __m512i min = v_0001_0000;

    const __m512i min_shifts = _mm512_setr_epi64(
        0x2020202020202020,
        0x0005090980808080,
        0x2020202020202020,
        0x0005090980808080,
        0x2020202020202020,
        0x0005090980808080,
        0x2020202020202020,
        0x0005090980808080
    );

    {
        const __m512i shift = _mm512_shuffle_epi8(min_shifts, char_class);
        min = _mm512_srlv_epi32(min, shift);
    }

    __m512i max = v_0010_ffff;

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

    __mmask16 not_surrogate;
    {
        const __m512i t0 = _mm512_and_si512(utf32, v_ffff_f800);
        not_surrogate = _mm512_cmpneq_epu32_mask(t0, v_0000_d800);
    }

#if 0
    __mmask16 in_range;
    in_range = _mm512_mask_cmpge_epu32_mask(not_surrogate, utf32, min);
    in_range = _mm512_mask_cmple_epu32_mask(in_range, utf32, max);

    return in_range;
#else
    const __m512i d = _mm512_sub_epi32(max, min);
    const __m512i v = _mm512_sub_epi32(utf32, min);

    return _mm512_mask_cmple_epu32_mask(not_surrogate, v, d);
#endif
}


template <typename OUTPUT>
size_t avx512_validating_utf8_to_fixed_length(const char* str, size_t len, OUTPUT* dwords) {
    constexpr bool UTF32 = std::is_same<OUTPUT, uint32_t>::value;
    constexpr bool UTF16 = std::is_same<OUTPUT, char16_t>::value;
    static_assert(UTF32 or UTF16, "output type has to be uint32_t (for UTF-32) or char16_t (for UTF-16)");

    const char* ptr = str;
    const char* end = ptr + len;

    OUTPUT* output = dwords;

    if (len > 64) {

        // load 16 bytes: [abcdefghijklmnop]
        __m128i   prev128   = _mm_loadu_si128((const __m128i*)ptr);
        uint32_t  prev_cont = continuation_bytes(prev128);
        __m512i   prev      = _mm512_broadcast_i64x2(prev128);
        __mmask16 ascii     = _mm_movemask_epi8(prev128);
        ptr += 16;

        while (ptr + 16 < end) {
            // load next 16 bytes: [qrstu....]
            const __m128i curr128 = _mm_loadu_si128((const __m128i*)ptr);
            uint32_t curr_cont = continuation_bytes(curr128);
            const __m512i curr = _mm512_broadcast_i64x2(curr128);
            if (ascii == 0) {
                if (UTF32)
                    _mm512_storeu_si512((__m512i*)output, _mm512_cvtepu8_epi32(prev128));
                else
                    _mm256_storeu_si256((__m256i*)output, _mm256_cvtepu8_epi16(prev128));

                output += 16;
                ptr += 16;
                prev = curr;
                prev128 = curr128;
                prev_cont = curr_cont;
                ascii = _mm_movemask_epi8(curr128);
                continue;
            }

            ascii = _mm_movemask_epi8(curr128);

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

            const uint32_t continuation = prev_cont | (curr_cont << 16);

            const __m512i input = _mm512_shuffle_epi8(merged, expand_ver2);
            prev = curr;
            prev128 = curr128;
            prev_cont = curr_cont;

            /* 1. Classify leading bytes */
            __m512i char_class;
            char_class = _mm512_srli_epi32(input, 4);
            char_class = _mm512_and_si512(char_class, v_0000_000f);
            char_class = _mm512_or_si512(char_class, v_8080_8000);

            /* 2. Get positions of leading bytes */
            __mmask16 leading_bytes;
            {
                const __m512i t0 = _mm512_and_si512(input, v_0000_00c0);
                leading_bytes = _mm512_cmpneq_epu32_mask(t0, v_0000_0080);
            }

            /* 3. Validate UTF-8 structure */
            {
                /** pshufb
                continuation = 0
                ascii    = 0x01
                _2_bytes = 0x03
                _3_bytes = 0x07
                _4_bytes = 0x0f

                mask_lookup = 4 * [
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
                ] */
                const __m512i mask_lookup = _mm512_setr_epi64(
                    0x0101010101010101,
                    0x0f07030300000000,
                    0x0101010101010101,
                    0x0f07030300000000,
                    0x0101010101010101,
                    0x0f07030300000000,
                    0x0101010101010101,
                    0x0f07030300000000
                );
                const __m512i mask = _mm512_shuffle_epi8(mask_lookup, char_class);
                const __m512i expected = _mm512_srli_epi32(mask, 1);

                __m512i v_continuation;
                v_continuation = _mm512_set1_epi32(continuation);
                v_continuation = _mm512_srlv_epi32(v_continuation, _mm512_setr_epi32(
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
                ));

                const __m512i masked = _mm512_and_si512(v_continuation, mask);

                const __mmask16 matched = _mm512_mask_cmpeq_epu32_mask(leading_bytes, masked, expected);
                if (matched != leading_bytes)
                    return 0;
            }

            const int valid_count = __builtin_popcount(leading_bytes);

            // 3. Convert words into UTF-32
            const __m512i utf32 = avx512_utf8_to_utf32__aux__version3(char_class, input);

            // 4. Validate if UTF-32 chars are in valid ranges
            {
                const __mmask16 in_range = avx512_utf8_validate_ranges_masked(char_class, utf32);
                if ((leading_bytes & in_range) != leading_bytes) {
                    return 0;
                }
            }

            // 4. Pack only valid words
            const __m512i out = _mm512_mask_compress_epi32(_mm512_setzero_si512(), leading_bytes, utf32);

            // 5. Store them
            if (UTF32)
                _mm512_storeu_si512((__m512i*)output, out);
            else
                _mm256_storeu_si256((__m256i*)output, _mm512_cvtepi32_epi16(out));

            output += valid_count;
            ptr += 16;
        }
    }

    // slow path
    if (len > 64) {
        ptr -= 16;
    }

    while (ptr < end) {
        uint32_t val;
        ptr += utf8_decode(ptr, val);
        *output++ = val;
    }

    return output - dwords;
}

size_t avx512_validating_utf8_to_utf32(const char* str, size_t len, uint32_t* dwords) {
    return avx512_validating_utf8_to_fixed_length<uint32_t>(str, len, dwords);
}

size_t avx512_validating_utf8_to_utf16(const char* str, size_t len, char16_t* chars) {
    return avx512_validating_utf8_to_fixed_length<char16_t>(str, len, chars);
}


namespace {
    const __m512i v_0f = _mm512_set1_epi8(0x0f);
    const __m512i v_10 = _mm512_set1_epi8(0x10);
    const __m512i v_40 = _mm512_set1_epi8(0x40);
    const __m512i v_80 = _mm512_set1_epi8(char(0x80));
    const __m512i v_1f = _mm512_set1_epi8(0x1f);
    const __m512i v_3f = _mm512_set1_epi8(0x3f);
    const __m512i v_c0 = _mm512_set1_epi8(char(0xc0));
    const __m512i v_c2 = _mm512_set1_epi8(char(0xc2));
    const __m512i v_e0 = _mm512_set1_epi8(char(0xe0));
    const __m512i v_f0 = _mm512_set1_epi8(char(0xf0));
    const __m512i v_f8 = _mm512_set1_epi8(char(0xf8));
}


__mmask64 avx512_validate_leading_bytes(__m512i leading_bytes, __m512i continuation1, __mmask64 tested_chars) {

    const __m512i nibble0 = _mm512_and_si512(leading_bytes, v_0f);

    // 1. Assume all non-continuation bytes are valid leading bytes
    //    We unset all continuation bytes (0b10xx_xxxx) and ASCII chars
    //    (0b0xxx_xxxx) --- so we looking for 0b11xx_xxxx chars.
    //
    //    (We may mask-out some leading bytes via `tested_chars`).
    __mmask64 valid;
    {
        const __m512i t0 = _mm512_and_si512(leading_bytes, v_c0);
        valid = _mm512_cmpeq_epi8_mask(t0, v_c0);
    }

    __mmask64 _2bytes;
    {
        const __m512i t0 = _mm512_and_si512(leading_bytes, v_e0);
        _2bytes = _mm512_cmpeq_epi8_mask(t0, v_c0);
    }

    __mmask64 _3bytes;
    {
        const __m512i t0 = _mm512_and_si512(leading_bytes, v_f0);
        _3bytes = _mm512_cmpeq_epi8_mask(t0, v_e0);
    }

    __mmask64 _4bytes;
    {
        const __m512i t0 = _mm512_and_si512(leading_bytes, v_f8);
        _4bytes = _mm512_cmpeq_epi8_mask(t0, v_f0);
    }

    // 1. Handle 2-byte chars
    //    Valid if leading byte is greater than 0xc2
    __mmask64 valid_2bytes = _mm512_mask_cmpge_epu8_mask(_2bytes, leading_bytes, v_c2);

    // 4. Handle 3-byte chars
    //    let M = (continuation1 & 03f) > 0x1f
    continuation1 = _mm512_and_si512(continuation1, v_3f);
    __mmask64 valid_3bytes;
    {
        /** pshufb 
            M     = 0b0000_0000 # we test 5th bit
            notM  = 0b0010_0000
            true  = 0b1100_0000

            fixup_lookup = 4 * [
                M,      # 0b0000
                true,   # 0b0001
                true,   # 0b0010
                true,   # 0b0011
                true,   # 0b0100
                true,   # 0b0101
                true,   # 0b0110
                true,   # 0b0111
                true,   # 0b1000
                true,   # 0b1001
                true,   # 0b1010
                true,   # 0b1011
                true,   # 0b1100
                notM,   # 0b1101
                true,   # 0b1110
                true,   # 0b1111
        ] */
        const __m512i fixup_lookup = _mm512_setr_epi64(
            0xc0c0c0c0c0c0c000,
            0xc0c020c0c0c0c0c0,
            0xc0c0c0c0c0c0c000,
            0xc0c020c0c0c0c0c0,
            0xc0c0c0c0c0c0c000,
            0xc0c020c0c0c0c0c0,
            0xc0c0c0c0c0c0c000,
            0xc0c020c0c0c0c0c0
        );

        const __m512i fixup = _mm512_shuffle_epi8(fixup_lookup, nibble0);
        const __m512i t0 = _mm512_xor_si512(fixup, continuation1);
        valid_3bytes = _mm512_mask_cmpgt_epu8_mask(_3bytes, t0, v_1f);
    }

    // 5. Handle 4-byte chars
    __mmask64 valid_4bytes;
    {
        // continuation1 in range [0..63] (0b0000_0000 .. 0b0011_1111)
        //
        // case 1: c1 <= 0xf:  0b0000_xxxx - 0x10 = 0b1111_yyyy
        // case 2: c1  > 0xf:  0b00xx_xxxx - 0x10 = 0b00yy_yyyy
        __m512i t0;
        t0 = _mm512_sub_epi8(continuation1, v_10);
        t0 = _mm512_xor_si512(t0, v_40);
        // bit 7th of c: continuation1 <= 0x0f
        // bit 6th of c: continuation1 >  0x0f

        /** pshufb 
            le_0f   = 0x80  # c1[7] = continuation1 <= 0xf0
            gt_0f   = 0x40  # c1[6] = not c1[7]
            true    = gt_0f | le_0f
            false   = 0x00

            mask_lookup = 4 * [
                gt_0f,  # 0b0000
                true,   # 0b0001
                true,   # 0b0010
                true,   # 0b0011
                le_0f,  # 0b0100
                false,  # 0b0101
                false,  # 0b0110
                false,  # 0b0111
                false,  # 0b1000
                false,  # 0b1001
                false,  # 0b1010
                false,  # 0b1011
                false,  # 0b1100
                false,  # 0b1101
                false,  # 0b1110
                false,  # 0b1111
        ] */
        const __m512i mask_lookup = _mm512_setr_epi64(
            0x00000080c0c0c040,
            0x0000000000000000,
            0x00000080c0c0c040,
            0x0000000000000000,
            0x00000080c0c0c040,
            0x0000000000000000,
            0x00000080c0c0c040,
            0x0000000000000000
        );
        const __m512i mask = _mm512_shuffle_epi8(mask_lookup, nibble0);

        valid_4bytes = _mm512_mask_test_epi8_mask(_4bytes, mask, t0);
    }

    {
        // all: marks all valid leading bytes
        __mmask64 all = _kor_mask64(valid_2bytes, _kor_mask64(valid_3bytes, valid_4bytes));

        // reset leading byte marks from input
        valid = _kxor_mask64(valid, all);
        valid = _kand_mask64(valid, tested_chars);
    }

    return valid == 0;
}


bool avx512_validate_utf8_structure(__m512i input) {
    /* 1. bitmask for various character byte classes.

        leading: 111010011100011010001
                 abccdddefgggghiijjjjk   a-k -- 11 UTF8- chars characters

        ascii:   110000011000010000001
                 ab     ef    h      k

        2 bytes: 001000000000001000000
                   cc          ii

        3 bytes: 000010000000000000000
                     ddd              

        4 bytes: 000000000100010010000
                          gggg   jjjj
    */
    uint64_t leading;
    uint64_t ascii;
    uint64_t _2bytes;
    uint64_t _3bytes;
    uint64_t _4bytes;

    // we can valide 60 - 4 leading bytes
    constexpr uint64_t mask  = 0x0ffffffffffffffflu;

    {
        const __m512i t0 = _mm512_and_si512(input, v_c0);
        leading = _mm512_cmpneq_epu8_mask(t0, v_80);
    }
    {
        ascii = _mm512_testn_epi8_mask(input, v_80) & mask;
    }
    {
        const __m512i t0 = _mm512_and_si512(input, v_e0);
        _2bytes = _mm512_cmpeq_epi8_mask(t0, v_c0) & mask;
    }
    {
        const __m512i t0 = _mm512_and_si512(input, v_f0);
        _3bytes = _mm512_cmpeq_epi8_mask(t0, v_e0) & mask;
    }
    {
        const __m512i t0 = _mm512_and_si512(input, v_f8);
        _4bytes = _mm512_cmpeq_epi8_mask(t0, v_f0) & mask;
    }

    const uint64_t next1 = leading >> 1;
    const uint64_t next2 = leading >> 2;
    const uint64_t next3 = leading >> 3;
    const uint64_t next4 = leading >> 4;

    /* 1. validate ascii
        ascii =             110000011000010000001
                            ^^^^^^^^^^^^^^^^^    
                            60 consdered bytes
        next1 =             110100111000110100010
        next & ascii =      100000011000010000000
    */ 
    const uint64_t valid_ascii = ascii & next1;


    /* 2. validate 2-byte chars
    
        2-byte   001000000000001000000 
        ~next1   001011000111001011101 -- expect a contination byte
        next2    101001110001101000100 -- and a leading byte
        ------------------------------
        and-all  001000000000001000000
    */
    const uint64_t valid_2bytes = (_2bytes & next2) & ~next1;

    /* 3. validate 3-byte chars

        3-byte   000010000000000000000
        ~next1   001011000111001011100 -- expect a contination byte
        ~next2   010110001110010111000 -- another contination byte
        next3    010011100011010001000 -- and a leading byte
        ------------------------------
        and-all  000010000000000000000

    */
    const uint64_t valid_3bytes = (_3bytes & next3) & ~(next1 | next2);

    /* 4. validate 4-byte chars

        4-byte   000000000100010010000
        ~next1   001011000111001011100 -- expect a contination byte
        ~next2   010110001110010111000 -- another contination
        ~next3   101100011100101110000 -- another contination
        next4    100111000110100010000 -- and a leading byte
        ------------------------------
        and-all  000000000100000010000

    */
    const uint64_t valid_4bytes = (_4bytes & next4) & ~(next1 | next2 | next3);
    return (valid_ascii == ascii)
       and (valid_2bytes == _2bytes)
       and (valid_3bytes == _3bytes)
       and (valid_4bytes == _4bytes);
}


namespace {
    __m512i _mm512_rotate_by1_epi8(const __m512i input) {

        // lanes order: 1, 2, 3, 0 => 0b00_11_10_01
        const __m512i permuted = _mm512_shuffle_i32x4(input, input, 0x39);

        return _mm512_alignr_epi8(permuted, input, 1);
    }
}


template <unsigned idx0, unsigned idx1, unsigned idx2, unsigned idx3>
struct shuffle_const {
    static_assert((idx0 >= 0 and idx0 <= 3), "idx0 must be in range 0..3");
    static_assert((idx1 >= 0 and idx1 <= 3), "idx1 must be in range 0..3");
    static_assert((idx2 >= 0 and idx2 <= 3), "idx2 must be in range 0..3");
    static_assert((idx3 >= 0 and idx3 <= 3), "idx3 must be in range 0..3");

    static const unsigned value = idx0 | (idx1 << 2) | (idx2 << 4) | (idx3 << 6);
};

template <unsigned idx>
constexpr __m512i broadcast_epi128(__m512i v) {
    return _mm512_shuffle_i32x4(v, v, shuffle_const<idx, idx, idx, idx>::value);
}


template <typename OUTPUT>
size_t avx512bw_validating_utf8_to_fixed_length(const char* str, size_t len, OUTPUT* dwords) {
    constexpr bool UTF32 = std::is_same<OUTPUT, uint32_t>::value;
    constexpr bool UTF16 = std::is_same<OUTPUT, char16_t>::value;
    static_assert(UTF32 or UTF16, "output type has to be uint32_t (for UTF-32) or char16_t (for UTF-16)");

    const char* ptr = str;
    const char* end = ptr + len;

    OUTPUT* output = dwords;

    while (ptr + 64 < end) {
        const __m512i input = _mm512_loadu_si512((const __m512i*)ptr);
        const __mmask64 ascii = _mm512_test_epi8_mask(input, v_80);
        if (ascii == 0) {
            const __m256i h0 = _mm512_castsi512_si256(input);
            const __m256i h1 = _mm512_extracti32x8_epi32(input, 1);

            const __m128i t0 = _mm256_castsi256_si128(h0);
            const __m128i t1 = _mm256_extracti32x4_epi32(h0, 1);
            const __m128i t2 = _mm256_castsi256_si128(h1);
            const __m128i t3 = _mm256_extracti32x4_epi32(h1, 1);

            if (UTF32) {
                _mm512_storeu_si512((__m512i*)(output + 0*16), _mm512_cvtepu8_epi32(t0));
                _mm512_storeu_si512((__m512i*)(output + 1*16), _mm512_cvtepu8_epi32(t1));
                _mm512_storeu_si512((__m512i*)(output + 2*16), _mm512_cvtepu8_epi32(t2));
                _mm512_storeu_si512((__m512i*)(output + 3*16), _mm512_cvtepu8_epi32(t3));
            }
            else {
                _mm256_storeu_si256((__m256i*)(output + 0*16), _mm256_cvtepu8_epi16(t0));
                _mm256_storeu_si256((__m256i*)(output + 1*16), _mm256_cvtepu8_epi16(t1));
                _mm256_storeu_si256((__m256i*)(output + 2*16), _mm256_cvtepu8_epi16(t2));
                _mm256_storeu_si256((__m256i*)(output + 3*16), _mm256_cvtepu8_epi16(t3));
            }

            output += 64;
            ptr += 64;
            continue;
        }

        // 1. Validate the structure of UTF-8 sequence.
        //    Note: procedure validates chars that starts in range [0..60]
        //    of input.
        if (not avx512_validate_utf8_structure(input)) {
            return false;
        }

        // 2. Precise validate: once we know that the bytes structure is correct,
        //    we have to check for some forbidden input values.
        //    Note: this procedure validates chars in range [0..63], due to
        //    method of obtaining continuation1 vector. (compare with
        //    validation/avx512-validate-utf8.cpp)
        const __m512i continuation1 = _mm512_rotate_by1_epi8(input);
        if (not avx512_validate_leading_bytes(input, continuation1, 0x0ffffffffffffffflu)) {
            return false;
        }

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

        // 3. Convert 3*16 input bytes
        // We waste the last 16 bytes, which are not fully validated...
        // this is another topic to check in the future.
#define TRANSCODE16(LANE0, LANE1)                                                                            \
        {                                                                                                    \
            const __m512i merged = _mm512_mask_mov_epi32(LANE0, 0x1000, LANE1);                              \
            const __m512i input = _mm512_shuffle_epi8(merged, expand_ver2);                                  \
                                                                                                             \
            __mmask16 leading_bytes;                                                                         \
            const __m512i t0 = _mm512_and_si512(input, v_0000_00c0);                                         \
            leading_bytes = _mm512_cmpneq_epu32_mask(t0, v_0000_0080);                                       \
                                                                                                             \
            __m512i char_class;                                                                              \
            char_class = _mm512_srli_epi32(input, 4);                                                        \
            char_class = _mm512_and_si512(char_class, v_0000_000f);                                          \
            char_class = _mm512_or_si512(char_class, v_8080_8000);                                           \
                                                                                                             \
            const int valid_count = __builtin_popcount(leading_bytes);                                       \
            const __m512i utf32 = avx512_utf8_to_utf32__aux__version3(char_class, input);                    \
                                                                                                             \
            const __m512i out = _mm512_mask_compress_epi32(_mm512_setzero_si512(), leading_bytes, utf32);    \
                                                                                                             \
            if (UTF32)                                                                                       \
                _mm512_storeu_si512((__m512i*)output, out);                                                  \
            else                                                                                             \
                _mm256_storeu_si256((__m256i*)output, _mm512_cvtepi32_epi16(out));                           \
                                                                                                             \
            output += valid_count;                                                                           \
        }

        const __m512i lane0 = broadcast_epi128<0>(input);
        const __m512i lane1 = broadcast_epi128<1>(input);
        TRANSCODE16(lane0, lane1)

        const __m512i lane2 = broadcast_epi128<2>(input);
        TRANSCODE16(lane1, lane2)

        const __m512i lane3 = broadcast_epi128<3>(input);
        TRANSCODE16(lane2, lane3)

        ptr += 3*16;
    }

    while (ptr < end) {
        uint32_t val;
        ptr += utf8_decode(ptr, val);
        *output++ = val;
    }

    return output - dwords;
}


size_t avx512bw_validating_utf8_to_utf32(const char* str, size_t len, uint32_t* dwords) {
    return avx512bw_validating_utf8_to_fixed_length<uint32_t>(str, len, dwords);
}

size_t avx512bw_validating_utf8_to_utf16(const char* str, size_t len, char16_t* chars) {
    return avx512bw_validating_utf8_to_fixed_length<char16_t>(str, len, chars);
}
