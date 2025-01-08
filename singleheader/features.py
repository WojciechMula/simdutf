from pathlib import Path
from io import StringIO
import os


TMP = Path("/dev/shm/")
if not TMP.exists():
    TMP = Path("/tmp")


def main():
    compilers = {
        'default': 'c++',
    }

    crosscompilers = find_crosscompilers()
    if crosscompilers:
        print("Found the following crosscompilers in $PATH:")
        for arch, compiler in crosscompilers.items():
            print('%-12s: %s' % (arch, compiler))
            compilers[arch] = compiler

    make = create_make(compilers)

    file = Path('Makefile')
    update_file(make, file)


def create_make(compilers):
    f = StringIO()

    def writeln(s):
        f.write(s + '\n')

    writeln("SRC=../src")
    writeln("INC=../include")

    archs = []
    for arch, compiler in compilers.items():
        targets = []
        for features in feature_combinations:
            parts = [arch] + name_for_features(features)

            target_dir = TMP / '-'.join(parts)
            target = target_dir / 'simdutf.o'

            targets.append((target, features))

        archs.append((arch, compiler, targets))
        
        writeln('')
        writeln(f"{arch.upper()}=\\")
        for (target, _) in targets:
            writeln(f'\t{target} \\')

    # add convienent targets for architectures ('default' is first)
    for arch in compilers:
        writeln('')
        writeln(f'.PHONY: {arch}')
        writeln(f'{arch}: $({arch.upper()})')

    # add 'all' target
    writeln('')
    writeln('.PHONY: all')
    writeln('all: %s' % (' '.join(compilers)))
    writeln('')
    writeln('.PHONY: all')
    writeln('clean:')
    writeln('\t$(RM) %s' % ' '.join(f"$({arch.upper()})" for arch in compilers))

    # add individual targets
    for arch, compiler, targets in archs:
        for target, features in targets:
            opts = ' '.join(feature2option[feat] for feat in features)

            target_dir = target.parent
            
            writeln('')
            writeln(f"{target}: amalgamate.py")
            writeln(f"\tmkdir -p {target_dir}")
            writeln(f"\tpython3 amalgamate.py --no-zip --no-readme --source-dir=$(SRC) --include-dir=$(INC) --output {target_dir} {opts}")
            writeln(f"\tcd {target_dir} && {compiler} -c simdutf.cpp")


    return f.getvalue()


def update_file(contents, path):
    if path.exists():
        if path.read_text() == contents:
            return

        print(f"updating {path}")
    else:
        print(f"creating {path}")

    path.write_text(contents)
    

def find_crosscompilers():
    found = {}
    for path in os.environ['PATH'].split(':'):
        path = Path(path)
        gxx = glob_many(path, ['*-g++*', '*-c++*', '*clang++'])
        for arch in crosscompilers:
            if arch in found:
                continue

            for name in gxx:
                if is_compiler(arch, ['g++', 'c++', 'clang++'], name):
                    found[arch] = name
                    break

    return found


def glob_many(rootdir, patterns):
    tmp = []
    for pat in patterns:
        tmp.extend([file.name for file in rootdir.glob(pat)])

    tmp.sort()
    return tmp


def is_compiler(arch, compilers, name):
    # we're looking for "arch-foo-bar-g++" or "arch-foo-bar-g++-version"
    tmp = name.split('-')
    if tmp[0] != arch:
        return False

    if tmp[-1] in compilers:
        return True

    if len(tmp) >= 3 and tmp[-2] in compilers and is_number(tmp[-1]):
        return True


def is_number(s):
    try:
        _ = int(s)
        return True
    except ValueError:
        return False


def name_for_features(features):
    return [feature2stem[feat] for feat in features]


SIMDUTF_FEATURE_DETECT_ENCODING = 'SIMDUTF_FEATURE_DETECT_ENCODING'
SIMDUTF_FEATURE_LATIN1          = 'SIMDUTF_FEATURE_LATIN1'
SIMDUTF_FEATURE_ASCII           = 'SIMDUTF_FEATURE_ASCII'
SIMDUTF_FEATURE_BASE64          = 'SIMDUTF_FEATURE_BASE64'
SIMDUTF_FEATURE_UTF8            = 'SIMDUTF_FEATURE_UTF8'
SIMDUTF_FEATURE_UTF16           = 'SIMDUTF_FEATURE_UTF16'
SIMDUTF_FEATURE_UTF32           = 'SIMDUTF_FEATURE_UTF32'


feature2stem = {
    SIMDUTF_FEATURE_DETECT_ENCODING : 'de',
    SIMDUTF_FEATURE_LATIN1          : 'lat1',
    SIMDUTF_FEATURE_ASCII           : 'ascii',
    SIMDUTF_FEATURE_BASE64          : 'base64',
    SIMDUTF_FEATURE_UTF8            : 'utf8',
    SIMDUTF_FEATURE_UTF16           : 'utf16',
    SIMDUTF_FEATURE_UTF32           : 'utf32',
}

feature2option = {
    SIMDUTF_FEATURE_DETECT_ENCODING : '--with-detect-enc',
    SIMDUTF_FEATURE_LATIN1          : '--with-latin1',
    SIMDUTF_FEATURE_ASCII           : '--with-ascii',
    SIMDUTF_FEATURE_BASE64          : '--with-base64',
    SIMDUTF_FEATURE_UTF8            : '--with-utf8',
    SIMDUTF_FEATURE_UTF16           : '--with-utf16',
    SIMDUTF_FEATURE_UTF32           : '--with-utf32',
}

feature_combinations = [
    [SIMDUTF_FEATURE_DETECT_ENCODING],
    [SIMDUTF_FEATURE_ASCII],
    [SIMDUTF_FEATURE_UTF8],
    [SIMDUTF_FEATURE_UTF16],
    [SIMDUTF_FEATURE_UTF32],
    [SIMDUTF_FEATURE_UTF8, SIMDUTF_FEATURE_LATIN1],
    [SIMDUTF_FEATURE_UTF16, SIMDUTF_FEATURE_LATIN1],
    [SIMDUTF_FEATURE_UTF32, SIMDUTF_FEATURE_LATIN1],
    [SIMDUTF_FEATURE_UTF8, SIMDUTF_FEATURE_LATIN1, SIMDUTF_FEATURE_DETECT_ENCODING],
    [SIMDUTF_FEATURE_UTF16, SIMDUTF_FEATURE_LATIN1, SIMDUTF_FEATURE_DETECT_ENCODING],
    [SIMDUTF_FEATURE_UTF32, SIMDUTF_FEATURE_LATIN1, SIMDUTF_FEATURE_DETECT_ENCODING],
    [SIMDUTF_FEATURE_UTF8, SIMDUTF_FEATURE_LATIN1, SIMDUTF_FEATURE_DETECT_ENCODING, SIMDUTF_FEATURE_ASCII],
    [SIMDUTF_FEATURE_UTF16, SIMDUTF_FEATURE_LATIN1, SIMDUTF_FEATURE_DETECT_ENCODING, SIMDUTF_FEATURE_ASCII],
    [SIMDUTF_FEATURE_UTF32, SIMDUTF_FEATURE_LATIN1, SIMDUTF_FEATURE_DETECT_ENCODING, SIMDUTF_FEATURE_ASCII],
    [SIMDUTF_FEATURE_UTF8, SIMDUTF_FEATURE_UTF16],
    [SIMDUTF_FEATURE_UTF16, SIMDUTF_FEATURE_UTF32],
    [SIMDUTF_FEATURE_UTF32, SIMDUTF_FEATURE_UTF8],
    [SIMDUTF_FEATURE_UTF32, SIMDUTF_FEATURE_UTF16],
    [SIMDUTF_FEATURE_UTF32, SIMDUTF_FEATURE_UTF16, SIMDUTF_FEATURE_UTF8],
    [SIMDUTF_FEATURE_UTF8, SIMDUTF_FEATURE_UTF16, SIMDUTF_FEATURE_LATIN1],
    [SIMDUTF_FEATURE_UTF16, SIMDUTF_FEATURE_UTF32, SIMDUTF_FEATURE_LATIN1],
    [SIMDUTF_FEATURE_UTF32, SIMDUTF_FEATURE_UTF8, SIMDUTF_FEATURE_LATIN1],
    [SIMDUTF_FEATURE_UTF32, SIMDUTF_FEATURE_UTF16, SIMDUTF_FEATURE_LATIN1],
    [SIMDUTF_FEATURE_UTF32, SIMDUTF_FEATURE_UTF16, SIMDUTF_FEATURE_UTF8, SIMDUTF_FEATURE_LATIN1],
    #{SIMDUTF_FEATURE_UTF8},
]

crosscompilers = [
    'aarch64',
    'loongarch64',
    'powerpc64',
    'riscv64',
]

if __name__ == '__main__':
    main()
