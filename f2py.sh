#!/bin/bash

#FILES='create_nodes.f90'
FILES="compute_adaptahop.f90"
PYTHON="uv run python"
#F2PY=f2py
F2PY="$PYTHON -m numpy.f2py"
FORT=gfortran # or "ifort"
MACHINE="gc" # or "tardis"
BASEDIR=$(dirname "$0")

for f in $FILES
do
    bn=$(basename "$f" .f90)
    echo -e "\n\n\nCompiling $bn\n\n\n"
    if [ $FORT == "gfortran" ]; then
        # If not OMP, remove `-fopenmp` and `-lgomp`
        $FORT -x f95-cpp-input -c -fopenmp $f 
        if [ $MACHINE == "tardis" ]; then
            export CFLAGS="-fPIC -O2 -std=c99"
        fi
        $F2PY -lgomp --f90exec=$FORT --f77exec=$FORT --f90flags='-fopenmp -O3 -x f95-cpp-input'  -c $f -m $bn

    elif [ $FORT == "ifort" ]; then
        # If not OMP, remove `-fopenmp` and `-liomp5`
        $FORT -O3 -foptimize-sibling-calls -c $f
        if [ $MACHINE == "tardis" ]; then
            $F2PY -c $f -m $bn --compiler=intelem --fcompiler=intelem --opt='-O3 -heap-arrays -foptimize-sibling-calls -fpp -m64 -free -fopenmp -std=c99' -liomp5
        elif [ $MACHINE == "gc" ]; then
            $F2PY -c $f -m $bn --fcompiler=intelem --opt='-O3 -heap-arrays -foptimize-sibling-calls -fpp -m64 -free -fopenmp -D_GNU_SOURCE' -liomp5
        else
            echo "NotImplementedError: unknown machine '$MACHINE'"
            exit 2
        fi
    fi
done
rm *.o *.mod
