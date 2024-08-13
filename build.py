import ninja_syntax as n
import os
import sys
from subprocess import run

join = os.path.join

def list_files_in_dir(d):
    for f in os.listdir(d):
        p = os.path.join(d, f)
        if os.path.isfile(p):
            yield f

def create_build_ninja():
    fout = open("build.ninja", "w")
    out = n.Writer(fout)

    def outd(p):
        return join("$outdir", p)

    out._line('builddir = .ninja')

    out.variable(
        key   = 'msvc_deps_prefix',
        value = 'Note: including file:',
        )

    out.variable(
        key   = 'outdir',
        value = 'out',
        )

    out.rule(
        name    = 'compile_cuda',
        depfile = '$out.d',
        command = 'nvcc -MD -MF $out.d -c -O2 $extraflags $in -o $out',
        )

    out.rule(
        name    = "compile_debug",
        depfile = '$out.d',
        command = 'nvcc -MD -MF $out.d $extraflags -g -c $in -o $out',
        )

    out.rule(
        name    = 'build_binary',
        command = 'nvcc -o $out $extraflags $in',
        )

    out.rule(
        name    = "compile_cl_debug",
        deps    = 'msvc',
        command = 'clang-cl -nologo -MTd /showIncludes /EHsc $extraflags -Zi -c $in -o $out',
        )

    out.rule(
        name    = 'build_cl_exe',
        command = 'cl -nologo /Fe$out $extraflags $in $linkflags',
        )

    # ------------------------------------------------------------------------------------------- #
    # Linux build, optimized for RTX4090
    # ------------------------------------------------------------------------------------------- #

    out.build(
        outputs = outd('hash_search.o'),
        rule    = 'compile_cuda',
        inputs  = 'src/sha1_hash_search.cu',
        variables = {
            'extraflags': '--gpu-architecture=sm_89 -lto',
        },
        )

    out.build(
        outputs = outd('main.o'),
        rule    = 'compile_cuda',
        inputs  = 'src/main.cu',
        variables = {
            'extraflags': '--gpu-architecture=sm_89 -lto',
        }
        )

    out.build(
        outputs = outd('hash_search_rtx4090'),
        rule    = 'build_binary',
        inputs  = [outd(p) for p in [
            'hash_search.o',
            'main.o',
        ]],
        variables = {
            'extraflags': '--gpu-architecture=sm_89',
        },
        )

    # ------------------------------------------------------------------------------------------- #
    # Local Windows build
    # ------------------------------------------------------------------------------------------- #

    out.build(
        outputs = outd('hash_search.obj'),
        rule    = 'compile_cuda',
        inputs  = 'src/sha1_hash_search.cu',
        variables = {
            'extraflags': '-std=c++20',
        },
        )

    out.build(
        outputs = outd('main.obj'),
        rule    = 'compile_cuda',
        inputs  = 'src/main.cu',
        variables = {
            'extraflags': '-std=c++20',
        },
        )

    out.build(
        outputs = outd('hash_search.exe'),
        rule    = 'build_binary',
        inputs  = [outd(p) for p in [
            'hash_search.obj',
            'main.obj',
        ]],
        )

    # ------------------------------------------------------------------------------------------- #
    # Tests
    # ------------------------------------------------------------------------------------------- #

    out.build(
        outputs = outd('sha256.obj'),
        rule    = 'compile_cl_debug',
        inputs  = 'src/sha256.cpp',
        )

    for f in list_files_in_dir('test'):
        base, _ = os.path.splitext(f);
        if not base.endswith('.test'):
            continue
        out.build(
            outputs   = outd(base) + '.obj',
            rule      = 'compile_cl_debug',
            inputs    = join('test', f),
            variables = {
                'extraflags': '-I src -I ext',
            },
            )

    out.build(
        outputs = outd('hash_search.test.exe'),
        rule    = 'build_cl_exe',
        inputs  = [outd(p) for p in [
            'jobgenerator.test.obj',
            'sha256.test.obj',
            'sha256.obj',
        ]] + [
            'ext/gtest.lib',
            'ext/gtest_main.lib',
        ],
        variables = {
            'extraflags': '-Zi',
            'linkflags': '-link -subsystem:console',
        },
        )

    #out.build(
    #    outputs  = 'test',
    #    rule     = 'phony',
    #    implicit = outd('sha1_hash_search.test.exe'),
    #    )

if __name__ == '__main__':
    create_build_ninja()
    command = ["ninja"] + sys.argv[1:]
    print("Running command " + str(command))
    run(command)
