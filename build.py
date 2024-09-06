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
        name    = "compile_cuda_debug",
        depfile = '$out.d',
        command = 'nvcc -Xcompiler /FS -Xcompiler -MTd -MD -MF $out.d $extraflags -g -c $in -o $out',
        )

    out.rule(
        name    = 'build_cuda_binary',
        command = 'nvcc -o $out $extraflags $in $linkflags',
        )

    # ------------------------------------------------------------------------------------------- #
    # Linux build, optimized for RTX4090
    # ------------------------------------------------------------------------------------------- #

    out.build(
        outputs = outd('jobgenerator.o'),
        rule    = 'compile_cuda',
        inputs  = 'src/jobgenerator.cpp',
        variables = {
            'extraflags': '--gpu-architecture=sm_89 -lto',
        },
        )

    out.build(
        outputs = outd('hash_search.o'),
        rule    = 'compile_cuda',
        inputs  = 'src/hash_search.cu',
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
        rule    = 'build_cuda_binary',
        inputs  = [outd(p) for p in [
            'jobgenerator.o',
            'hash_search.o',
            'main.o',
        ]],
        variables = {
            'extraflags': '--gpu-architecture=sm_89 -lto',
        },
        )

    # ------------------------------------------------------------------------------------------- #
    # Local Windows build
    # ------------------------------------------------------------------------------------------- #

    out.build(
        outputs = outd('jobgenerator.obj'),
        rule    = 'compile_cuda',
        inputs  = 'src/jobgenerator.cpp',
        )

    out.build(
        outputs = outd('hash_search.obj'),
        rule    = 'compile_cuda',
        inputs  = 'src/hash_search.cu',
        variables = {
            'extraflags': '-std=c++20 -lineinfo',
        },
        )

    out.build(
        outputs = outd('main.obj'),
        rule    = 'compile_cuda',
        inputs  = 'src/main.cu',
        variables = {
            'extraflags': '-std=c++20 -lineinfo',
        },
        )

    out.build(
        outputs = outd('hash_search.exe'),
        rule    = 'build_cuda_binary',
        inputs  = [outd(p) for p in [
            'jobgenerator.obj',
            'hash_search.obj',
            'main.obj',
        ]],
        )

    # ------------------------------------------------------------------------------------------- #
    # Local Windows test
    # ------------------------------------------------------------------------------------------- #

    out.build(
        outputs = outd('jobgenerator_debug.obj'),
        rule    = 'compile_cuda_debug',
        inputs  = 'src/jobgenerator.cpp',
        )

    out.build(
        outputs = outd('hash_search_debug.obj'),
        rule    = 'compile_cuda_debug',
        inputs  = 'src/hash_search.cu',
    )

    for f in list_files_in_dir('test'):
        base, ext = os.path.splitext(f);
        if not base.endswith('.test'):
            continue

        rule = 'compile_cuda_debug'

        out.build(
            outputs   = outd(base) + '.obj',
            rule      = rule,
            inputs    = join('test', f),
            variables = {
                'extraflags': '-I src -I ext',
            },
            )

    out.build(
        outputs = outd('hash_search.test.exe'),
        rule    = 'build_cuda_binary',
        inputs  = [outd(p) for p in [
            'jobgenerator.test.obj',
            'sha256.test.obj',
            'sha256_l55.test.obj',
            'hash_search_debug.obj',
            'jobgenerator_debug.obj',
        ]] + [
            'ext/gtest.lib',
            'ext/gtest_main.lib',
        ],
        variables = {
            'extraflags': '-Xcompiler -Zi',
            'linkflags':  'libcmtd.lib -Xlinker -subsystem:console',
        },
        )

    out.build(
        outputs  = 'test',
        rule     = 'phony',
        implicit = outd('hash_search.test.exe'),
        )

if __name__ == '__main__':
    create_build_ninja()
    command = ["ninja"] + sys.argv[1:]
    run(command)
