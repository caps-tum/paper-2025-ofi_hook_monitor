if [[ "$CPU" != $(cat current_configured_arch) ]]; then
    echo "archs are different! will clean"

    make clean
    make distclean
fi;

echo "$CPU" > current_configured_arch

LIBFABRIC_PATH="$HOME/build/libfabric_monitor_build"

PREFIX="$HOME/build/ompi505_monitor_build"

./configure CFLAGS="-Wall" LDFLAGS="-Wl,-rpath,$LIBFABRIC_PATH/lib"\
     --prefix="$PREFIX" \
     --with-ofi="$LIBFABRIC_PATH" \
     --with-libevent=internal \
     --with-pmix=internal \
     --with-hwloc=internal \
     --with-hwloc-pci