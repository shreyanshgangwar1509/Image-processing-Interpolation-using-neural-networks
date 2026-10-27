# Super-Resolution Benchmark Aggregated Results

## Dataset: BSD100

| experiment                 | model    | dataset   |   scale | degradation                        |   noise_sigma |   n_images |   param_count |   mean_inference_time_s |   mean_psnr |   mean_ssim |
|:---------------------------|:---------|:----------|--------:|:-----------------------------------|--------------:|-----------:|--------------:|------------------------:|------------:|------------:|
| our_nn_x2_clean            | our_nn   | BSD100    |       2 | bicubic_down_x2                    |             0 |        100 |        165267 |             0.00950284  |     31.489  |    0.891047 |
| baselines_x2_clean         | lanczos  | BSD100    |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.00533134  |     29.9249 |    0.855024 |
| srcnn_x2_clean             | srcnn    | BSD100    |       2 | bicubic_down_x2                    |             0 |        100 |         20099 |             0.00717984  |     29.8903 |    0.853792 |
| baselines_x2_clean         | bicubic  | BSD100    |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.00156843  |     29.7456 |    0.849729 |
| baselines_x2_clean         | bilinear | BSD100    |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.000552473 |     28.687  |    0.810618 |
| baselines_x2_clean         | nearest  | BSD100    |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.000214694 |     28.4236 |    0.825264 |
| our_nn_x4_clean            | our_nn   | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |        202259 |             0.00622724  |     26.9004 |    0.713671 |
| our_nn_x4_clean_a1_skip    | our_nn   | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |        195971 |             0.00493125  |     26.8579 |    0.711872 |
| our_nn_x4_clean_a2_attn    | our_nn   | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |        199283 |             0.00594174  |     26.8475 |    0.711188 |
| our_nn_x4_clean_a0_plain   | our_nn   | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |        195971 |             0.00469788  |     26.4956 |    0.698709 |
| srcnn_x4_clean             | srcnn    | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |         20099 |             0.00733933  |     26.3447 |    0.687711 |
| baselines_x4_clean         | lanczos  | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.0044555   |     26.127  |    0.675843 |
| baselines_x4_clean         | bicubic  | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.00171579  |     26.0522 |    0.672175 |
| our_nn_x4_noise15          | our_nn   | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |        202259 |             0.00631217  |     25.5524 |    0.636835 |
| baselines_x4_clean         | bilinear | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.000543737 |     25.5454 |    0.646913 |
| our_nn_x4_noise15_a1_skip  | our_nn   | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |        195971 |             0.00500838  |     25.5357 |    0.635509 |
| our_nn_x4_noise15_a2_attn  | our_nn   | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |        199283 |             0.00579811  |     25.5163 |    0.635367 |
| our_nn_x4_noise15_a0_plain | our_nn   | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |        195971 |             0.00464125  |     25.41   |    0.630823 |
| srcnn_x4_noise15           | srcnn    | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |         20099 |             0.00733373  |     25.3017 |    0.623481 |
| baselines_x4_clean         | nearest  | BSD100    |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.000309973 |     25.0378 |    0.629964 |
| baselines_x4_noise15       | bilinear | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.000591114 |     24.6814 |    0.582621 |
| baselines_x4_noise15       | bicubic  | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.00181463  |     24.5896 |    0.56605  |
| baselines_x4_noise15       | lanczos  | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.00457048  |     24.5443 |    0.560946 |
| baselines_x4_noise15       | nearest  | BSD100    |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.0002426   |     23.5164 |    0.483999 |


## Dataset: Set14

| experiment                 | model    | dataset   |   scale | degradation                        |   noise_sigma |   n_images |   param_count |   mean_inference_time_s |   mean_psnr |   mean_ssim |
|:---------------------------|:---------|:----------|--------:|:-----------------------------------|--------------:|-----------:|--------------:|------------------------:|------------:|------------:|
| our_nn_x2_clean            | our_nn   | Set14     |       2 | bicubic_down_x2                    |             0 |         14 |        165267 |             0.0133637   |     32.6882 |    0.909708 |
| baselines_x2_clean         | lanczos  | Set14     |       2 | bicubic_down_x2                    |             0 |         14 |             0 |             0.00775567  |     30.7821 |    0.880048 |
| srcnn_x2_clean             | srcnn    | Set14     |       2 | bicubic_down_x2                    |             0 |         14 |         20099 |             0.009973    |     30.6958 |    0.877103 |
| baselines_x2_clean         | bicubic  | Set14     |       2 | bicubic_down_x2                    |             0 |         14 |             0 |             0.002294    |     30.5577 |    0.875201 |
| baselines_x2_clean         | bilinear | Set14     |       2 | bicubic_down_x2                    |             0 |         14 |             0 |             0.000808614 |     29.249  |    0.84071  |
| baselines_x2_clean         | nearest  | Set14     |       2 | bicubic_down_x2                    |             0 |         14 |             0 |             0.000383667 |     28.582  |    0.846429 |
| our_nn_x4_clean            | our_nn   | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |        202259 |             0.00824761  |     27.5565 |    0.755879 |
| our_nn_x4_clean_a1_skip    | our_nn   | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |        195971 |             0.00718987  |     27.457  |    0.7532   |
| our_nn_x4_clean_a2_attn    | our_nn   | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |        199283 |             0.00789147  |     27.4522 |    0.752811 |
| our_nn_x4_clean_a0_plain   | our_nn   | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |        195971 |             0.00670828  |     26.8166 |    0.735034 |
| srcnn_x4_clean             | srcnn    | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |         20099 |             0.0101823   |     26.5984 |    0.724421 |
| baselines_x4_clean         | lanczos  | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |             0 |             0.00650406  |     26.3192 |    0.713005 |
| baselines_x4_clean         | bicubic  | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |             0 |             0.00254777  |     26.2209 |    0.708865 |
| our_nn_x4_noise15          | our_nn   | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |        202259 |             0.00820962  |     25.8245 |    0.677859 |
| our_nn_x4_noise15_a1_skip  | our_nn   | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |        195971 |             0.00714879  |     25.7676 |    0.675395 |
| our_nn_x4_noise15_a2_attn  | our_nn   | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |        199283 |             0.00778387  |     25.7318 |    0.675039 |
| our_nn_x4_noise15_a0_plain | our_nn   | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |        195971 |             0.00673207  |     25.5216 |    0.668398 |
| baselines_x4_clean         | bilinear | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |             0 |             0.000842793 |     25.5157 |    0.682055 |
| srcnn_x4_noise15           | srcnn    | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |         20099 |             0.0101902   |     25.4599 |    0.661287 |
| baselines_x4_clean         | nearest  | Set14     |       4 | bicubic_down_x4                    |             0 |         14 |             0 |             0.000347853 |     24.7411 |    0.655327 |
| baselines_x4_noise15       | bicubic  | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |             0 |             0.00258085  |     24.7386 |    0.604726 |
| baselines_x4_noise15       | lanczos  | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |             0 |             0.00664038  |     24.7079 |    0.600178 |
| baselines_x4_noise15       | bilinear | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |             0 |             0.000853624 |     24.668  |    0.618935 |
| baselines_x4_noise15       | nearest  | Set14     |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |         14 |             0 |             0.000405754 |     23.3594 |    0.513824 |


## Dataset: Set5

| experiment                 | model    | dataset   |   scale | degradation                        |   noise_sigma |   n_images |   param_count |   mean_inference_time_s |   mean_psnr |   mean_ssim |
|:---------------------------|:---------|:----------|--------:|:-----------------------------------|--------------:|-----------:|--------------:|------------------------:|------------:|------------:|
| our_nn_x2_clean            | our_nn   | Set5      |       2 | bicubic_down_x2                    |             0 |          5 |        165267 |             0.0104639   |     37.0971 |    0.957152 |
| baselines_x2_clean         | lanczos  | Set5      |       2 | bicubic_down_x2                    |             0 |          5 |             0 |             0.00458813  |     34.3172 |    0.936457 |
| srcnn_x2_clean             | srcnn    | Set5      |       2 | bicubic_down_x2                    |             0 |          5 |         20099 |             0.00718598  |     34.146  |    0.931795 |
| baselines_x2_clean         | bicubic  | Set5      |       2 | bicubic_down_x2                    |             0 |          5 |             0 |             0.00193005  |     33.9798 |    0.933158 |
| baselines_x2_clean         | bilinear | Set5      |       2 | bicubic_down_x2                    |             0 |          5 |             0 |             0.000673485 |     32.2548 |    0.911932 |
| baselines_x2_clean         | nearest  | Set5      |       2 | bicubic_down_x2                    |             0 |          5 |             0 |             0.000930166 |     30.8584 |    0.900162 |
| our_nn_x4_clean            | our_nn   | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |        202259 |             0.0109611   |     30.5586 |    0.865506 |
| our_nn_x4_clean_a1_skip    | our_nn   | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |        195971 |             0.00734272  |     30.4164 |    0.862123 |
| our_nn_x4_clean_a2_attn    | our_nn   | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |        199283 |             0.00808301  |     30.3984 |    0.861612 |
| our_nn_x4_clean_a0_plain   | our_nn   | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |        195971 |             0.00775466  |     29.4973 |    0.835271 |
| srcnn_x4_clean             | srcnn    | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |         20099 |             0.00730038  |     29.1926 |    0.829055 |
| baselines_x4_clean         | lanczos  | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |             0 |             0.0033402   |     28.8146 |    0.818202 |
| baselines_x4_clean         | bicubic  | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |             0 |             0.00162411  |     28.6344 |    0.813831 |
| our_nn_x4_noise15          | our_nn   | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |        202259 |             0.01196     |     27.8864 |    0.78731  |
| our_nn_x4_noise15_a1_skip  | our_nn   | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |        195971 |             0.00756702  |     27.8355 |    0.785379 |
| our_nn_x4_noise15_a2_attn  | our_nn   | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |        199283 |             0.00838227  |     27.751  |    0.780856 |
| baselines_x4_clean         | bilinear | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |             0 |             0.000591278 |     27.5604 |    0.789574 |
| our_nn_x4_noise15_a0_plain | our_nn   | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |        195971 |             0.00679002  |     27.4993 |    0.773766 |
| srcnn_x4_noise15           | srcnn    | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |         20099 |             0.00721469  |     27.3875 |    0.772435 |
| baselines_x4_noise15       | bicubic  | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |             0 |             0.00164256  |     26.379  |    0.697069 |
| baselines_x4_noise15       | lanczos  | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |             0 |             0.00344696  |     26.3475 |    0.69167  |
| baselines_x4_noise15       | bilinear | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |             0 |             0.000576305 |     26.2807 |    0.717854 |
| baselines_x4_clean         | nearest  | Set5      |       4 | bicubic_down_x4                    |             0 |          5 |             0 |             0.000208616 |     26.2582 |    0.738007 |
| baselines_x4_noise15       | nearest  | Set5      |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |          5 |             0 |             0.000271797 |     24.3779 |    0.57999  |


## Dataset: Urban100

| experiment                 | model    | dataset   |   scale | degradation                        |   noise_sigma |   n_images |   param_count |   mean_inference_time_s |   mean_psnr |   mean_ssim |
|:---------------------------|:---------|:----------|--------:|:-----------------------------------|--------------:|-----------:|--------------:|------------------------:|------------:|------------:|
| our_nn_x2_clean            | our_nn   | Urban100  |       2 | bicubic_down_x2                    |             0 |        100 |        165267 |             0.0431256   |     29.6804 |    0.899674 |
| baselines_x2_clean         | lanczos  | Urban100  |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.0271893   |     27.2632 |    0.850534 |
| srcnn_x2_clean             | srcnn    | Urban100  |       2 | bicubic_down_x2                    |             0 |        100 |         20099 |             0.031179    |     27.2408 |    0.847033 |
| baselines_x2_clean         | bicubic  | Urban100  |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.00813493  |     27.0749 |    0.845851 |
| baselines_x2_clean         | bilinear | Urban100  |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.00239351  |     25.9681 |    0.809047 |
| baselines_x2_clean         | nearest  | Urban100  |       2 | bicubic_down_x2                    |             0 |        100 |             0 |             0.000910926 |     25.6276 |    0.821119 |
| our_nn_x4_clean            | our_nn   | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |        202259 |             0.0248653   |     24.4069 |    0.721104 |
| our_nn_x4_clean_a1_skip    | our_nn   | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |        195971 |             0.022787    |     24.3059 |    0.716362 |
| our_nn_x4_clean_a2_attn    | our_nn   | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |        199283 |             0.0234383   |     24.301  |    0.715994 |
| our_nn_x4_clean_a0_plain   | our_nn   | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |        195971 |             0.0210049   |     23.729  |    0.688265 |
| srcnn_x4_clean             | srcnn    | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |         20099 |             0.0318361   |     23.5846 |    0.680532 |
| baselines_x4_clean         | lanczos  | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.022418    |     23.3253 |    0.665357 |
| our_nn_x4_noise15          | our_nn   | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |        202259 |             0.0247652   |     23.2643 |    0.653676 |
| baselines_x4_clean         | bicubic  | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.00860064  |     23.2421 |    0.661603 |
| our_nn_x4_noise15_a1_skip  | our_nn   | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |        195971 |             0.0223991   |     23.213  |    0.651274 |
| our_nn_x4_noise15_a2_attn  | our_nn   | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |        199283 |             0.0235044   |     23.1882 |    0.649883 |
| our_nn_x4_noise15_a0_plain | our_nn   | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |        195971 |             0.021312    |     23.0132 |    0.641477 |
| srcnn_x4_noise15           | srcnn    | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |         20099 |             0.0318377   |     22.8781 |    0.632627 |
| baselines_x4_clean         | bilinear | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.00269364  |     22.6883 |    0.635398 |
| baselines_x4_noise15       | lanczos  | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.0231045   |     22.4018 |    0.559994 |
| baselines_x4_noise15       | bicubic  | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.00845326  |     22.3975 |    0.563912 |
| baselines_x4_noise15       | bilinear | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.00271069  |     22.2051 |    0.574918 |
| baselines_x4_clean         | nearest  | Urban100  |       4 | bicubic_down_x4                    |             0 |        100 |             0 |             0.00102543  |     22.1747 |    0.616031 |
| baselines_x4_noise15       | nearest  | Urban100  |       4 | bicubic_down_x4+gaussian_noise_s15 |            15 |        100 |             0 |             0.00123266  |     21.3196 |    0.481177 |


## Dataset: test_dummy

| experiment   | model   | dataset    |   scale | degradation     |   noise_sigma |   n_images |   param_count |   mean_inference_time_s |   mean_psnr |   mean_ssim |
|:-------------|:--------|:-----------|--------:|:----------------|--------------:|-----------:|--------------:|------------------------:|------------:|------------:|
| smoke_test   | srcnn   | test_dummy |       2 | bicubic_down_x2 |             0 |          5 |         20099 |               0.0119119 |     10.5603 |    0.538006 |

