# Super-Resolution Benchmark Aggregated Results

## Dataset: BSD100

| experiment               | model    | dataset   |   scale | degradation                        |   noise_sigma |   n_images |   param_count |   mean_inference_time_s |   mean_psnr |   mean_ssim |
|:-------------------------|:---------|:----------|--------:|:-----------------------------------|--------------:|-----------:|--------------:|------------------------:|------------:|------------:|
| baselines_x2_clean       | lanczos  | BSD100    |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.00533134  |     29.9249 |    0.855024 |
| baselines_x2_clean       | bicubic  | BSD100    |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.00156843  |     29.7456 |    0.849729 |
| baselines_x2_clean       | bilinear | BSD100    |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.000552473 |     28.687  |    0.810618 |
| baselines_x2_clean       | nearest  | BSD100    |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.000214694 |     28.4236 |    0.825264 |
| our_nn_x4_clean          | our_nn   | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |        202259 |             0.00622724  |     26.9004 |    0.713671 |
| our_nn_x4_clean_a0_plain | our_nn   | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |        195971 |             0.00469788  |     26.4956 |    0.698709 |
| baselines_x4_clean       | lanczos  | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.0044555   |     26.127  |    0.675843 |
| baselines_x4_clean       | bicubic  | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.00171579  |     26.0522 |    0.672175 |
| baselines_x4_clean       | bilinear | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.000543737 |     25.5454 |    0.646913 |
| baselines_x4_clean       | nearest  | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.000309973 |     25.0378 |    0.629964 |
| baselines_x4_noise15     | bilinear | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.000591114 |     24.6814 |    0.582621 |
| baselines_x4_noise15     | bicubic  | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.00181463  |     24.5896 |    0.56605  |
| baselines_x4_noise15     | lanczos  | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.00457048  |     24.5443 |    0.560946 |
| baselines_x4_noise15     | nearest  | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.0002426   |     23.5164 |    0.483999 |


## Dataset: Set14

| experiment               | model    | dataset   |   scale | degradation                        |   noise_sigma |   n_images |   param_count |   mean_inference_time_s |   mean_psnr |   mean_ssim |
|:-------------------------|:---------|:----------|--------:|:-----------------------------------|--------------:|-----------:|--------------:|------------------------:|------------:|------------:|
| baselines_x2_clean       | lanczos  | Set14     |       2 | bicubic_down_x2                    |             0 |         14 |             0 |             0.00775567  |     30.7821 |    0.880048 |
| baselines_x2_clean       | bicubic  | Set14     |       2 | bicubic_down_x2                    |             0 |         14 |             0 |             0.002294    |     30.5577 |    0.875201 |
| baselines_x2_clean       | bilinear | Set14     |       2 | bicubic_down_x2                    |             0 |         14 |             0 |             0.000808614 |     29.249  |    0.84071  |
| baselines_x2_clean       | nearest  | Set14     |       2 | bicubic_down_x2                    |             0 |         14 |             0 |             0.000383667 |     28.582  |    0.846429 |
| our_nn_x4_clean          | our_nn   | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |        202259 |             0.00824761  |     27.5565 |    0.755879 |
| our_nn_x4_clean_a0_plain | our_nn   | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |        195971 |             0.00670828  |     26.8166 |    0.735034 |
| baselines_x4_clean       | lanczos  | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |             0 |             0.00650406  |     26.3192 |    0.713005 |
| baselines_x4_clean       | bicubic  | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |             0 |             0.00254777  |     26.2209 |    0.708865 |
| baselines_x4_clean       | bilinear | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |             0 |             0.000842793 |     25.5157 |    0.682055 |
| baselines_x4_clean       | nearest  | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |             0 |             0.000347853 |     24.7411 |    0.655327 |
| baselines_x4_noise15     | bicubic  | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |             0 |             0.00258085  |     24.7386 |    0.604726 |
| baselines_x4_noise15     | lanczos  | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |             0 |             0.00664038  |     24.7079 |    0.600178 |
| baselines_x4_noise15     | bilinear | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |             0 |             0.000853624 |     24.668  |    0.618935 |
| baselines_x4_noise15     | nearest  | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |             0 |             0.000405754 |     23.3594 |    0.513824 |


## Dataset: Set5

| experiment               | model    | dataset   |   scale | degradation                        |   noise_sigma |   n_images |   param_count |   mean_inference_time_s |   mean_psnr |   mean_ssim |
|:-------------------------|:---------|:----------|--------:|:-----------------------------------|--------------:|-----------:|--------------:|------------------------:|------------:|------------:|
| baselines_x2_clean       | lanczos  | Set5      |       2 | bicubic_down_x2                    |             0 |          5 |             0 |             0.00458813  |     34.3172 |    0.936457 |
| baselines_x2_clean       | bicubic  | Set5      |       2 | bicubic_down_x2                    |             0 |          5 |             0 |             0.00193005  |     33.9798 |    0.933158 |
| baselines_x2_clean       | bilinear | Set5      |       2 | bicubic_down_x2                    |             0 |          5 |             0 |             0.000673485 |     32.2548 |    0.911932 |
| baselines_x2_clean       | nearest  | Set5      |       2 | bicubic_down_x2                    |             0 |          5 |             0 |             0.000930166 |     30.8584 |    0.900162 |
| our_nn_x4_clean          | our_nn   | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |        202259 |             0.0109611   |     30.5586 |    0.865506 |
| our_nn_x4_clean_a0_plain | our_nn   | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |        195971 |             0.00775466  |     29.4973 |    0.835271 |
| baselines_x4_clean       | lanczos  | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |             0 |             0.0033402   |     28.8146 |    0.818202 |
| baselines_x4_clean       | bicubic  | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |             0 |             0.00162411  |     28.6344 |    0.813831 |
| baselines_x4_clean       | bilinear | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |             0 |             0.000591278 |     27.5604 |    0.789574 |
| baselines_x4_noise15     | bicubic  | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |             0 |             0.00164256  |     26.379  |    0.697069 |
| baselines_x4_noise15     | lanczos  | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |             0 |             0.00344696  |     26.3475 |    0.69167  |
| baselines_x4_noise15     | bilinear | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |             0 |             0.000576305 |     26.2807 |    0.717854 |
| baselines_x4_clean       | nearest  | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |             0 |             0.000208616 |     26.2582 |    0.738007 |
| baselines_x4_noise15     | nearest  | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |             0 |             0.000271797 |     24.3779 |    0.57999  |


## Dataset: Urban100

| experiment               | model    | dataset   |   scale | degradation                        |   noise_sigma |   n_images |   param_count |   mean_inference_time_s |   mean_psnr |   mean_ssim |
|:-------------------------|:---------|:----------|--------:|:-----------------------------------|--------------:|-----------:|--------------:|------------------------:|------------:|------------:|
| baselines_x2_clean       | lanczos  | Urban100  |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.0271893   |     27.2632 |    0.850534 |
| baselines_x2_clean       | bicubic  | Urban100  |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.00813493  |     27.0749 |    0.845851 |
| baselines_x2_clean       | bilinear | Urban100  |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.00239351  |     25.9681 |    0.809047 |
| baselines_x2_clean       | nearest  | Urban100  |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.000910926 |     25.6276 |    0.821119 |
| our_nn_x4_clean          | our_nn   | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |        202259 |             0.0248653   |     24.4069 |    0.721104 |
| our_nn_x4_clean_a0_plain | our_nn   | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |        195971 |             0.0210049   |     23.729  |    0.688265 |
| baselines_x4_clean       | lanczos  | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.022418    |     23.3253 |    0.665357 |
| baselines_x4_clean       | bicubic  | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.00860064  |     23.2421 |    0.661603 |
| baselines_x4_clean       | bilinear | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.00269364  |     22.6883 |    0.635398 |
| baselines_x4_noise15     | lanczos  | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.0231045   |     22.4018 |    0.559994 |
| baselines_x4_noise15     | bicubic  | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.00845326  |     22.3975 |    0.563912 |
| baselines_x4_noise15     | bilinear | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.00271069  |     22.2051 |    0.574918 |
| baselines_x4_clean       | nearest  | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.00102543  |     22.1747 |    0.616031 |
| baselines_x4_noise15     | nearest  | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.00123266  |     21.3196 |    0.481177 |


## Dataset: test_dummy

| experiment   | model   | dataset    |   scale | degradation     |   noise_sigma |   n_images |   param_count |   mean_inference_time_s |   mean_psnr |   mean_ssim |
|:-------------|:--------|:-----------|--------:|:----------------|--------------:|-----------:|--------------:|------------------------:|------------:|------------:|
| smoke_test   | srcnn   | test_dummy |       2 | bicubic_down_x2 |             0 |          5 |         20099 |               0.0119119 |     10.5603 |    0.538006 |

