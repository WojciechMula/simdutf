template <typename T> struct simd16;

template <typename T, typename Mask = simd16<bool>> struct base_u16 {
  __m128i value;
  static const int SIZE = sizeof(value);

  // Conversion from/to SIMD register
  simdutf_really_inline base_u16() = default;
  simdutf_really_inline base_u16(const __m128i _value) : value(_value) {}
  // Bit operations
  simdutf_really_inline simd16<T> operator|(const simd16<T> other) const {
    return __lsx_vor_v(this->value, other.value);
  }
  simdutf_really_inline simd16<T> operator&(const simd16<T> other) const {
    return __lsx_vand_v(this->value, other.value);
  }
  simdutf_really_inline simd16<T> operator~() const { return *this ^ 0xFFu; }

  friend simdutf_really_inline Mask operator==(const simd16<T> lhs,
                                               const simd16<T> rhs) {
    return __lsx_vseq_h(lhs.value, rhs.value);
  }

  template <int N = 1>
  simdutf_really_inline simd16<T> prev(const simd16<T> prev_chunk) const {
    return __lsx_vor_v(__lsx_vbsll_v(*this, N * 2),
                       __lsx_vbsrl_v(prev_chunk, 16 - N * 2));
  }
};

template <typename T, typename Mask = simd16<bool>>
struct base16 : base_u16<T> {
  simdutf_really_inline base16() : base_u16<T>() {}
  simdutf_really_inline base16(const __m128i _value) : base_u16<T>(_value) {}
  template <typename Pointer>
  simdutf_really_inline base16(const Pointer *ptr)
      : base16(__lsx_vld(ptr, 0)) {}

  static const int SIZE = sizeof(base_u16<T>::value);

  template <int N = 1>
  simdutf_really_inline simd16<T> prev(const simd16<T> prev_chunk) const {
    return __lsx_vor_v(__lsx_vbsll_v(*this, N * 2),
                       __lsx_vbsrl_v(prev_chunk, 16 - N * 2));
  }
};

// SIMD byte mask type (returned by things like eq and gt)
template <> struct simd16<bool> : base16<bool> {
  static simdutf_really_inline simd16<bool> splat(bool _value) {
    return __lsx_vreplgr2vr_h(uint16_t(-(!!_value)));
  }

  simdutf_really_inline simd16() : base16() {}
  simdutf_really_inline simd16(const __m128i _value) : base16<bool>(_value) {}
};

template <typename T> struct base16_numeric : base16<T> {
  static simdutf_really_inline simd16<T> splat(T _value) {
    return __lsx_vreplgr2vr_h(_value);
  }
  static simdutf_really_inline simd16<T> zero() { return __lsx_vldi(0); }
  static simdutf_really_inline simd16<T> load(const T values[8]) {
    return __lsx_vld(reinterpret_cast<const uint16_t *>(values), 0);
  }

  simdutf_really_inline base16_numeric(const __m128i _value)
      : base16<T>(_value) {}

  // Store to array
  simdutf_really_inline void store(T dst[8]) const {
    return __lsx_vst(this->value, dst, 0);
  }

  // Override to distinguish from bool version
  simdutf_really_inline simd16<T> operator~() const { return *this ^ 0xFFu; }

  // Addition/subtraction are the same for signed and unsigned
  simdutf_really_inline simd16<T> operator+(const simd16<T> other) const {
    return __lsx_vadd_b(*this, other);
  }
  simdutf_really_inline simd16<T> operator-(const simd16<T> other) const {
    return __lsx_vsub_b(*this, other);
  }
  simdutf_really_inline simd16<T> &operator+=(const simd16<T> other) {
    *this = *this + other;
    return *static_cast<simd16<T> *>(this);
  }
  simdutf_really_inline simd16<T> &operator-=(const simd16<T> other) {
    *this = *this - other;
    return *static_cast<simd16<T> *>(this);
  }
};

// Unsigned code unitstemplate<>
template <> struct simd16<uint16_t> : base16_numeric<uint16_t> {
  simdutf_really_inline simd16(const __m128i _value)
      : base16_numeric<uint16_t>((__m128i)_value) {}

  // Splat constructor
  simdutf_really_inline simd16(uint16_t _value) : simd16(splat(_value)) {}

  // Array constructor
  simdutf_really_inline simd16(const uint16_t *values) : simd16(load(values)) {}
  simdutf_really_inline simd16(const char16_t *values)
      : simd16(load(reinterpret_cast<const uint16_t *>(values))) {}

  // Order-specific operations
  simdutf_really_inline simd16<bool>
  operator<=(const simd16<uint16_t> other) const {
    return __lsx_vsle_hu(this->value, other.value);
  }
  simdutf_really_inline simd16<bool>
  operator>(const simd16<uint16_t> other) const {
    return __lsx_vslt_hu(other.value, this->value);
  }
  simdutf_really_inline simd16<bool>
  operator<(const simd16<uint16_t> other) const {
    return __lsx_vslt_hu(this->value, other.value);
  }

  template <unsigned N>
  static simdutf_really_inline simd8<uint8_t>
  pack_shifted_right(const simd16<uint16_t> &v0, const simd16<uint16_t> &v1) {
    return __lsx_vssrlni_bu_h(v1.value, v0.value, N);
  }

  // Pack with the unsigned saturation of two uint16_t code units into single
  // uint8_t vector
  static simdutf_really_inline simd8<uint8_t> pack(const simd16<uint16_t> &v0,
                                                   const simd16<uint16_t> &v1) {
    return pack_shifted_right<0>(v0, v1);
  }

  // Change the endianness
  simdutf_really_inline simd16<uint16_t> swap_bytes() const {
    return __lsx_vshuf4i_b(this->value, 0b10110001);
  }
};

template <typename T> struct simd16x32 {
  static constexpr int NUM_CHUNKS = 64 / sizeof(simd16<T>);
  static_assert(
      NUM_CHUNKS == 4,
      "LOONGARCH kernel should use four registers per 64-byte block.");
  simd16<T> chunks[NUM_CHUNKS];

  simd16x32(const simd16x32<T> &o) = delete; // no copy allowed
  simd16x32<T> &
  operator=(const simd16<T> other) = delete; // no assignment allowed
  simd16x32() = delete;                      // no default constructor allowed

  simdutf_really_inline
  simd16x32(const simd16<T> chunk0, const simd16<T> chunk1,
            const simd16<T> chunk2, const simd16<T> chunk3)
      : chunks{chunk0, chunk1, chunk2, chunk3} {}
  simdutf_really_inline simd16x32(const T *ptr)
      : chunks{simd16<T>::load(ptr),
               simd16<T>::load(ptr + sizeof(simd16<T>) / sizeof(T)),
               simd16<T>::load(ptr + 2 * sizeof(simd16<T>) / sizeof(T)),
               simd16<T>::load(ptr + 3 * sizeof(simd16<T>) / sizeof(T))} {}

  simdutf_really_inline void store(T *ptr) const {
    this->chunks[0].store(ptr + sizeof(simd16<T>) * 0 / sizeof(T));
    this->chunks[1].store(ptr + sizeof(simd16<T>) * 1 / sizeof(T));
    this->chunks[2].store(ptr + sizeof(simd16<T>) * 2 / sizeof(T));
    this->chunks[3].store(ptr + sizeof(simd16<T>) * 3 / sizeof(T));
  }

  simdutf_really_inline uint64_t to_bitmask() const {
    __m128i mask = __lsx_vbsll_v(__lsx_vmsknz_b((this->chunks[3]).value), 6);
    mask = __lsx_vor_v(
        mask, __lsx_vbsll_v(__lsx_vmsknz_b((this->chunks[2]).value), 4));
    mask = __lsx_vor_v(
        mask, __lsx_vbsll_v(__lsx_vmsknz_b((this->chunks[1]).value), 2));
    mask = __lsx_vor_v(mask, __lsx_vmsknz_b((this->chunks[0]).value));
    return __lsx_vpickve2gr_du(mask, 0);
  }

  simdutf_really_inline void swap_bytes() {
    this->chunks[0] = this->chunks[0].swap_bytes();
    this->chunks[1] = this->chunks[1].swap_bytes();
    this->chunks[2] = this->chunks[2].swap_bytes();
    this->chunks[3] = this->chunks[3].swap_bytes();
  }

  simdutf_really_inline uint64_t lteq(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] <= mask, this->chunks[1] <= mask,
                           this->chunks[2] <= mask, this->chunks[3] <= mask)
        .to_bitmask();
  }

  simdutf_really_inline uint64_t not_in_range(const T low, const T high) const {
    const simd16<T> mask_low = simd16<T>::splat(low);
    const simd16<T> mask_high = simd16<T>::splat(high);
    return simd16x32<bool>(
               (this->chunks[0] > mask_high) | (this->chunks[0] < mask_low),
               (this->chunks[1] > mask_high) | (this->chunks[1] < mask_low),
               (this->chunks[2] > mask_high) | (this->chunks[2] < mask_low),
               (this->chunks[3] > mask_high) | (this->chunks[3] < mask_low))
        .to_bitmask();
  }
}; // struct simd16x32<T>
