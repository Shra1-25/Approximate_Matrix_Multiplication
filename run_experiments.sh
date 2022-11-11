#!/bin/bash
((n = 2000))
((dmin = 256))
((dmax = 4096))
((rmin = 16))
((rmax = 128))
((d = dmin))
((r = rmin))

nvcc -std=c++11 -Xcompiler -fopenmp driver.cu -o gpu_experiment

# while [[ ${t} -le 64 ]] ; do
#     export OMP_NUM_THREADS=${t}
#     echo "Running with ${t} threads"
#     ((d = dmin))
while [[ ${d} -le ${dmax} ]] ; do
    ((r = rmin))
    while [[ ${r} -le ${rmax} ]] ; do
        echo "Testing with D ${d} and R ${r}"
        echo "./gpu_experiment -n${n} -d${d} -r${r} > gpu_results/n${n}_d${d}_r${r}.txt"
        ./gpu_experiment -n${n} -d${d} -r${r} > gpu_results/n${n}_d${d}_r${r}.txt
        ((r *= 2))
    done
    ((d *= 2))
done
    # (( t *= 2))
# done