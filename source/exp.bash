bsub -W 47:00 -R "rusage[mem=10000]" "python main.py --exname d_n_a_glove_crop_deta_non_normalize_alpha_noise_freq_10 --setting multiple_games_glove_d_n_a_crop_delta_non_normalize_alpha_noise_std_10  &> /dev/null"

bsub -W 47:00 -R "rusage[mem=10000]" "python main.py --exname d_n_a_glove_crop_deta_normalize_alpha_noise_freq_10 --setting multiple_games_glove_d_n_a_crop_delta_normalize_alpha_noise_std_10  &> /dev/null"

bsub -W 47:00 -R "rusage[mem=10000]" "python main.py --exname d_n_a_crop_deta_normalize_alpha_noise_freq_10 --setting multiple_games_d_n_a_crop_delta_normalize_alpha_noise_std_10  &> /dev/null"

bsub -W 47:00 -R "rusage[mem=10000]" "python main.py --exname d_n_a_crop_deta_non_normalize_alpha_noise_freq_10 --setting multiple_games_d_n_a_crop_delta_non_normalize_alpha_noise_std_10  &> /dev/null"
