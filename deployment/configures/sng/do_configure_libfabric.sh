#!/bin/bash

if [[ "$CPU" != $(cat current_configured_arch) ]]; then
    echo "archs are different! will clean"

    make clean
    make distclean
    sh autogen.sh
fi;


PREFIX="$HOME/build/libfabric_monitor_build"

echo "$CPU" > current_configured_arch
./configure \
    CFLAGS="-I/dss/lrzsys/sys/spack/release/22.2.1/opt/skylake_avx512/numactl/2.0.14-intel-vbsz45d/include" \
    CXXFLAGS="-I/dss/lrzsys/sys/spack/release/22.2.1/opt/skylake_avx512/numactl/2.0.14-intel-vbsz45d/include" \
    LDFLAGS="-L/dss/lrzsys/sys/spack/release/22.2.1/opt/skylake_avx512/numactl/2.0.14-intel-vbsz45d/lib"\
        --prefix="$PREFIX" \
        --enable-opx=yes \
        --enable-only \
        --with-json="$SPACK_PATH" \
        --with-curl="$SPACK_PATH" \
        --enable-restricted-dl \
        --enable-tcp \
        --enable-udp \
        --enable-rxm \
        --enable-rxd \
        --enable-hook_debug \
        --enable-hook_hmem \
        --enable-dmabuf_peer_mem \
        --enable-gdrcopy-dlopen \
        --enable-monitor=dl