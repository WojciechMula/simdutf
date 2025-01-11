#include "simdutf.h"

#if SIMDUTF_FEATURE_BASE64
// We include base64_tables once.
#include "tables/base64_tables.h"
#endif // SIMDUTF_FEATURE_BASE64

#include "implementation.cpp"
#include "encoding_types.cpp"
#include "error.cpp"
#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16
// The large tables should be included once and they
// should not depend on a kernel.
#include "tables/utf8_to_utf16_tables.h"
#include "tables/utf16_to_utf8_tables.h"
// End of tables.
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16

// The scalar routines should be included once.
#if SIMDUTF_FEATURE_ASCII
#include "scalar/ascii.h"
#endif // SIMDUTF_FEATURE_ASCII
#if SIMDUTF_FEATURE_UTF8
#include "scalar/utf8.h"
#endif // SIMDUTF_FEATURE_UTF8
#if SIMDUTF_FEATURE_UTF16
#include "scalar/utf16.h"
#endif // SIMDUTF_FEATURE_UTF16
#if SIMDUTF_FEATURE_UTF32
#include "scalar/utf32.h"
#endif // SIMDUTF_FEATURE_UTF32
#if SIMDUTF_FEATURE_LATIN1
#include "scalar/latin1.h"
#endif // SIMDUTF_FEATURE_LATIN1
#if SIMDUTF_FEATURE_BASE64
#include "scalar/base64.h"
#endif // SIMDUTF_FEATURE_BASE64

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32
#include "scalar/utf32_to_utf8/valid_utf32_to_utf8.h"
#include "scalar/utf32_to_utf8/utf32_to_utf8.h"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32

#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32
#include "scalar/utf32_to_utf16/valid_utf32_to_utf16.h"
#include "scalar/utf32_to_utf16/utf32_to_utf16.h"
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16
#include "scalar/utf16_to_utf8/valid_utf16_to_utf8.h"
#include "scalar/utf16_to_utf8/utf16_to_utf8.h"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16

#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32
#include "scalar/utf16_to_utf32/valid_utf16_to_utf32.h"
#include "scalar/utf16_to_utf32/utf16_to_utf32.h"
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_UTF32

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16
#include "scalar/utf8_to_utf16/valid_utf8_to_utf16.h"
#include "scalar/utf8_to_utf16/utf8_to_utf16.h"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF16

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32
#include "scalar/utf8_to_utf32/valid_utf8_to_utf32.h"
#include "scalar/utf8_to_utf32/utf8_to_utf32.h"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_UTF32

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
#include "scalar/latin1_to_utf8/latin1_to_utf8.h"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1
#include "scalar/latin1_to_utf16/latin1_to_utf16.h"
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1
#if SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1
#include "scalar/latin1_to_utf32/latin1_to_utf32.h"
#endif // SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
#include "scalar/utf8_to_latin1/utf8_to_latin1.h"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1
#include "scalar/utf16_to_latin1/utf16_to_latin1.h"
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1
#if SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1
#include "scalar/utf32_to_latin1/utf32_to_latin1.h"
#endif // SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1

#if SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
#include "scalar/utf8_to_latin1/valid_utf8_to_latin1.h"
#endif // SIMDUTF_FEATURE_UTF8 && SIMDUTF_FEATURE_LATIN1
#if SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1
#include "scalar/utf16_to_latin1/valid_utf16_to_latin1.h"
#endif // SIMDUTF_FEATURE_UTF16 && SIMDUTF_FEATURE_LATIN1
#if SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1
#include "scalar/utf32_to_latin1/valid_utf32_to_latin1.h"
#endif // SIMDUTF_FEATURE_UTF32 && SIMDUTF_FEATURE_LATIN1

SIMDUTF_PUSH_DISABLE_WARNINGS
SIMDUTF_DISABLE_UNDESIRED_WARNINGS

#if SIMDUTF_IMPLEMENTATION_ARM64
  //#include "arm64/implementation.cpp"
#endif
#if SIMDUTF_IMPLEMENTATION_FALLBACK
  #include "fallback/implementation.cpp"
#endif
#if SIMDUTF_IMPLEMENTATION_ICELAKE
  #include "icelake/implementation.cpp"
#endif
#if SIMDUTF_IMPLEMENTATION_HASWELL
  #include "haswell/implementation.cpp"
#endif
#if SIMDUTF_IMPLEMENTATION_PPC64
  //#include "ppc64/implementation.cpp"
#endif
#if SIMDUTF_IMPLEMENTATION_RVV
  //#include "rvv/implementation.cpp"
#endif
#if SIMDUTF_IMPLEMENTATION_WESTMERE
  //#include "westmere/implementation.cpp"
#endif
#if SIMDUTF_IMPLEMENTATION_LSX
  //#include "lsx/implementation.cpp"
#endif
#if SIMDUTF_IMPLEMENTATION_LASX
  //#include "lasx/implementation.cpp"
#endif

SIMDUTF_POP_DISABLE_WARNINGS
