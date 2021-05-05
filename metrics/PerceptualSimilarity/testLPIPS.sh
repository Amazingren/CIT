CUDA_VISIBLE_DECIVES=1 python lpips_2dirs.py \
-d0 PATH_of_your_project/XingVTON/result/TOM/test/try-on \
-d1 PATH_of_your_project/XingVTON/data/test/image \
-o ./example_dists.txt \
--use_gpu

