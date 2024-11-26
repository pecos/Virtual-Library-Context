# kmeans + Hotspot3D
# time ./parallel 512 8 1500 benchmarks/hotspot3D/data/power_512x8 benchmarks/hotspot3D/data/temp_512x8 output.out -n 24 -i benchmarks/kmeans/data/819200.txt -t 0.0000001 -l 10
# time ./parallel_vlc 512 8 1500 benchmarks/hotspot3D/data/power_512x8 benchmarks/hotspot3D/data/temp_512x8 output.out -n 24 -i benchmarks/kmeans/data/819200.txt -t 0.0000001 -l 10

# kmeans + cfd
# time ./parallel -n 24 -i benchmarks/kmeans/data/819200.txt -t 0.0000001 -l 10
# time ./parallel_vlc -n 24 -i benchmarks/kmeans/data/819200.txt -t 0.0000001 -l 10

# Hotspot3D + cfd
# time ./parallel 512 8 1500 benchmarks/hotspot3D/data/power_512x8 benchmarks/hotspot3D/data/temp_512x8 output.out
time ./parallel_vlc 512 8 1500 benchmarks/hotspot3D/data/power_512x8 benchmarks/hotspot3D/data/temp_512x8 output.out