#!/usr/bin/env python3
#
# Creates the amalgamated source files.
#
import argparse
import sys
import os.path
import subprocess
import os
import re
import shutil
import datetime
if sys.version_info[0] < 3:
    sys.stdout.write("Sorry, requires Python 3.x or better\n")
    sys.exit(1)

SCRIPTPATH = os.path.dirname(os.path.abspath(sys.argv[0]))
PROJECTPATH = os.path.dirname(SCRIPTPATH)
print(f"SCRIPTPATH={SCRIPTPATH} PROJECTPATH={PROJECTPATH}")

known_features = {
    'SIMDUTF_FEATURE_DETECT_ENCODING',
    'SIMDUTF_FEATURE_LATIN1',
    'SIMDUTF_FEATURE_ASCII',
    'SIMDUTF_FEATURE_BASE64',
    'SIMDUTF_FEATURE_UTF8',
    'SIMDUTF_FEATURE_UTF16',
    'SIMDUTF_FEATURE_UTF32',
}

def parse_args():
    p = argparse.ArgumentParser("SIMDUTF tool for amalgmation")
    p.add_argument("--source-dir",
                   metavar="SRC",
                   help="Source dir")
    p.add_argument("--include-dir",
                   metavar="INC",
                   help="Include dir")
    p.add_argument("--output-dir",
                   default='simdutf',
                   metavar="DIR",
                   help="Output directory")
    p.add_argument("--no-zip",
                   default=True,
                   action='store_false',
                   dest='zip',
                   help="Do not create .zip file")
    p.add_argument("--no-readme",
                   default=True,
                   action='store_false',
                   dest='readme',
                   help="Do not show readme after creating files")
    p.add_argument("--with-utf8",
                   default=None,
                   action='store_true',
                   help="Include UTF-8 support")
    p.add_argument("--with-utf16",
                   default=None,
                   action='store_true',
                   help="Include UTF-16 support")
    p.add_argument("--with-utf32",
                   default=None,
                   action='store_true',
                   help="Include UTF-32 support")
    p.add_argument("--with-base64",
                   default=None,
                   action='store_true',
                   help="Include Base64 support")
    p.add_argument("--with-detect-enc",
                   default=None,
                   action='store_true',
                   help="Include encoding detection support")
    p.add_argument("--with-ascii",
                   default=None,
                   action='store_true',
                   help="Include ASCII support")
    p.add_argument("--with-latin1",
                   default=None,
                   action='store_true',
                   help="Include Latin1 support")
    
    args = p.parse_args()

    enabled_features = set()
    if args.with_utf8:
        enabled_features.add('SIMDUTF_FEATURE_UTF8')
    if args.with_utf16:
        enabled_features.add('SIMDUTF_FEATURE_UTF16')
    if args.with_utf32:
        enabled_features.add('SIMDUTF_FEATURE_UTF32')
    if args.with_base64:
        enabled_features.add('SIMDUTF_FEATURE_BASE64')
    if args.with_detect_enc:
        enabled_features.add('SIMDUTF_FEATURE_DETECT_ENCODING')
    if args.with_ascii:
        enabled_features.add('SIMDUTF_FEATURE_ASCII')
    if args.with_latin1:
        enabled_features.add('SIMDUTF_FEATURE_LATIN1')

    if not enabled_features:
        enabled_features = set(known_features)

    return (args, enabled_features)


(args, enabled_features) = parse_args()

print("We are about to amalgamate all simdutf files into one source file.")
print("See https://www.sqlite.org/amalgamation.html and https://en.wikipedia.org/wiki/Single_Compilation_Unit for rationale.")
if "AMALGAMATE_SOURCE_PATH" not in os.environ:
    if args.source_dir is not None:
        AMALGAMATE_SOURCE_PATH = args.source_dir
    else:
        AMALGAMATE_SOURCE_PATH = os.path.join(PROJECTPATH, "src")
else:
    AMALGAMATE_SOURCE_PATH = os.environ["AMALGAMATE_SOURCE_PATH"]

if "AMALGAMATE_INCLUDE_PATH" not in os.environ:
    if args.include_dir is not None:
        AMALGAMATE_INCLUDE_PATH = args.include_dir
    else:
        AMALGAMATE_INCLUDE_PATH = os.path.join(PROJECTPATH, "include")
else:
    AMALGAMATE_INCLUDE_PATH = os.environ["AMALGAMATE_INCLUDE_PATH"]

if "AMALGAMATE_OUTPUT_PATH" not in os.environ:
    if args.output_dir is not None:
        AMALGAMATE_OUTPUT_PATH = args.output_dir
    else:
        AMALGAMATE_OUTPUT_PATH = os.path.join(SCRIPTPATH)
else:
    AMALGAMATE_OUTPUT_PATH = os.environ["AMALGAMATE_OUTPUT_PATH"]

# this list excludes the "src/generic headers"
ALLCFILES = ["simdutf.cpp"]

# order matters
ALLCHEADERS = ["simdutf.h"]

found_includes = []

current_implementation=''

def doinclude(fid, file, line):
    p = os.path.join(AMALGAMATE_INCLUDE_PATH, file)
    pi = os.path.join(AMALGAMATE_SOURCE_PATH, file)

    if os.path.exists(p):
        # generic includes are included multiple times
        if re.match('.*generic/.*.h', file):
            dofile(fid, AMALGAMATE_INCLUDE_PATH, file)
        # begin/end_implementation are also included multiple times
        elif re.match('.*/begin.h', file):
            dofile(fid, AMALGAMATE_INCLUDE_PATH, file)
        elif re.match('.*/end.h', file):
            dofile(fid, AMALGAMATE_INCLUDE_PATH, file)
        elif file not in found_includes:
            found_includes.append(file)
            dofile(fid, AMALGAMATE_INCLUDE_PATH, file)
        else:
            pass
    elif os.path.exists(pi):
        # generic includes are included multiple times
        if re.match('.*generic/.*.h', file):
            dofile(fid, AMALGAMATE_SOURCE_PATH, file)
        # begin/end_implementation are also included multiple times
        elif re.match('.*/begin.h', file):
            dofile(fid, AMALGAMATE_SOURCE_PATH, file)
        elif re.match('.*/end.h', file):
            dofile(fid, AMALGAMATE_SOURCE_PATH, file)
        elif file not in found_includes:
            found_includes.append(file)
            dofile(fid, AMALGAMATE_SOURCE_PATH, file)
        else:
            pass
    else:
        # If we don't recognize it, just emit the #include
        print(line, file=fid)

def dofile(fid, prepath, filename):
    global current_implementation
    #print(f"// dofile: invoked with prepath={prepath}, filename={filename}",file=fid)
    file = os.path.join(prepath, filename)
    RELFILE = os.path.relpath(file, PROJECTPATH)
    # Last lines are always ignored. Files should end by an empty lines.
    #print(f"/* begin file {RELFILE} */")
    print(f"/* begin file {RELFILE} */", file=fid)
    includepattern = re.compile(r'^\s*#\s*include "(.*)"')
    redefines_simdutf_implementation = re.compile(r'^#define\s+SIMDUTF_IMPLEMENTATION\s+(.*)')
    undefines_simdutf_implementation = re.compile(r'^#undef\s+SIMDUTF_IMPLEMENTATION\s*$')
    uses_simdutf_implementation = re.compile('SIMDUTF_IMPLEMENTATION([^_a-zA-Z0-9]|$)')
    for line in filter_features(file):
            line = line.rstrip('\n')
            s = includepattern.search(line)
            if s:
                includedfile = s.group(1)
                # include all from simdutf.cpp except simdutf.h
                if includedfile == "simdutf.h" and filename == "simdutf.cpp":
                    print(line, file=fid)
                    continue

                if includedfile.startswith('../'):
                    includedfile = includedfile[2:]
                # we explicitly include simdutf headers, one time each (unless they are generic, in which case multiple times is fine)
                doinclude(fid, includedfile, line)
            else:
                # does it contain a redefinition of SIMDUTF_IMPLEMENTATION ?
                s=redefines_simdutf_implementation.search(line)
                if s:
                    current_implementation=s.group(1)
                    print(f"// redefining SIMDUTF_IMPLEMENTATION to \"{current_implementation}\"\n// {line}", file=fid)
                elif undefines_simdutf_implementation.search(line):
                    # Don't include #undef SIMDUTF_IMPLEMENTATION since we're handling it ourselves
                    # print(f"// {line}")
                    pass
                else:
                    # copy the line, with SIMDUTF_IMPLEMENTATION replace to what it is currently defined to
                    print(uses_simdutf_implementation.sub(current_implementation+"\\1",line), file=fid)
    print(f"/* end file {RELFILE} */", file=fid)


def filter_features(file):
    """
    Design:

    * Feature macros SIMDUTF_FEATURE_foo must not be nested.
    * All #endifs must contain a comment with the repeated condition.
    """
    current_features = None
    start_if_line = None
    enabled = True
    prev_line = ''

    root_header = file.endswith("/implementation.h")

    with open(file, 'r') as f:
        for (lineno, line) in enumerate(f, 1):
            line = line.rstrip()
            if root_header and line.startswith('#define SIMDUTF_FEATURE'):
                # '#define SIMDUTF_FEATURE_FOO 1'
                tmp = line.split()
                assert len(tmp) == 3, line
                assert tmp[2] == '1'
                flag = tmp[1]

                if flag in enabled_features:
                    yield line
                else:
                    yield f'#define {flag} 0'

            elif line.startswith('#if SIMDUTF_FEATURE'):
                if start_if_line is not None:
                    raise ValueError(f"{file}:{lineno}: feature block already opened at line {start_if_line}")

                current_features = get_features(file, lineno, line[len('#if '):])
                start_if_line = lineno
                enabled = current_features.evaluate(enabled_features)
            elif line.startswith('#endif // SIMDUTF_FEATURE'):
                if start_if_line is None:
                    raise ValueError(f"{file}:{lineno}: feature block not opened, orphan #endif found")

                features = get_features(line, lineno, line[len('#endif // '):])
                if str(features) != str(current_features):
                    raise ValueError(f"{file}:{lineno}: feature #endif condition different than opening #if")

                enabled = True
                start_if_line = None
                current_features = None
            elif enabled:
                if not prev_line.endswith('\\'):
                    yield f"// {file}:{lineno}"

                if line or (not line and prev_line):
                    yield line

                prev_line = line


def get_features(file, lineno, line):
    try:
        return parse_condition(line)
    except e:
        raise ValueError(f"{file}:{lineno}: {e}")
        

class Token:
    def __init__(self, name):
        if name not in known_features:
            raise ValueError("unknown feature name '{name}'")

        self.name = name

    def evaluate(self, enabled_features):
        return self.name in enabled_features

    def __str__(self):
        return self.name


class And:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def evaluate(self, enabled_features):
        a = self.a.evaluate(enabled_features)
        b = self.b.evaluate(enabled_features)

        return a and b

    def __str__(self):
        return '(%s && %s)' % (self.a, self.b)


class Or:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def evaluate(self, enabled_features):
        a = self.a.evaluate(enabled_features)
        b = self.b.evaluate(enabled_features)

        return a or b

    def __str__(self):
        return '(%s || %s)' % (self.a, self.b)


def parse_condition(s):
    tokens = [t for t in re.split('( |\\(|\\)|&&|\\|\\|)', s) if t not in ('', ' ')]

    # Note: this is plain pattern matching, nothing generic
    if len(tokens) == 1:
        return Token(tokens[0])

    if len(tokens) == 3:
        if tokens[1] == '&&':
            a = Token(tokens[0])
            b = Token(tokens[2])
            return And(a, b)

        if tokens[1] == '||':
            a = Token(tokens[0])
            b = Token(tokens[2])
            return Or(a, b)

    if len(tokens) == 7:
        if tokens[1] == '&&' and tokens[2] == '(' and tokens[4] == '||' and tokens[6] == ')':
            a = Token(tokens[0])
            b = Token(tokens[3])
            c = Token(tokens[5])

            return And(a, Or(b, c))

    raise ValueError("cannot parse '{line}'")


# Get the generation date from git, so the output is reproducible.
# The %ci specifier gives the unambiguous ISO 8601 format, and
# does not change with locale and timezone at time of generation.
# Forcing it to be UTC is difficult, because it needs to be portable
# between gnu date and busybox date.
try:
    timestamp = subprocess.run(['git', 'show', '-s', '--format=%ci', 'HEAD'],
                           stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
except:
    print("git not found, timestamp based on current time")
    timestamp = str(datetime.datetime.now())
print(f"timestamp is {timestamp}")

os.makedirs(AMALGAMATE_OUTPUT_PATH, exist_ok=True)
AMAL_H = os.path.join(AMALGAMATE_OUTPUT_PATH, "simdutf.h")
AMAL_C = os.path.join(AMALGAMATE_OUTPUT_PATH, "simdutf.cpp")
DEMOCPP = os.path.join(AMALGAMATE_OUTPUT_PATH, "amalgamation_demo.cpp")
README = os.path.join(AMALGAMATE_OUTPUT_PATH, "README.md")

print(f"Creating {AMAL_H}")
amal_h = open(AMAL_H, 'w')
print(f"/* auto-generated on {timestamp}. Do not edit! */", file=amal_h)
for h in ALLCHEADERS:
    doinclude(amal_h, h, f"ERROR {h} not found")

amal_h.close()
print()
print()
print(f"Creating {AMAL_C}")
amal_c = open(AMAL_C, 'w')
print(f"/* auto-generated on {timestamp}. Do not edit! */", file=amal_c)
for c in ALLCFILES:
    doinclude(amal_c, c, f"ERROR {c} not found")

amal_c.close()

# copy the README and DEMOCPP
if SCRIPTPATH != AMALGAMATE_OUTPUT_PATH:
  shutil.copy2(os.path.join(SCRIPTPATH,"amalgamation_demo.cpp"),AMALGAMATE_OUTPUT_PATH)
  shutil.copy2(os.path.join(SCRIPTPATH,"README.md"),AMALGAMATE_OUTPUT_PATH)

if args.zip:
    import zipfile
    zf = zipfile.ZipFile(os.path.join(AMALGAMATE_OUTPUT_PATH,'singleheader.zip'), 'w', zipfile.ZIP_DEFLATED)
    zf.write(AMAL_C,  "simdutf.cpp")
    zf.write(AMAL_H,  "simdutf.h")
    zf.write(DEMOCPP, "amalgamation_demo.cpp")


print("Done with all files generation.")

print(f"Files have been written to directory: {AMALGAMATE_OUTPUT_PATH}/")
print("Done with all files generation.")

if args.readme:
    print("\nGiving final instructions:")
    with open(README) as r:
        for line in r:
            print(line)
