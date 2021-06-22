commonvoicedir=$1
wavs_dir=$2
mkdir extraction_dir
mkdir extraction_dir/compare
mkdir extraction_dir/gemaps
python prepare/common_voice_prepare.py $commonvoicedir
python prepare/common_voice_parallel_infile_extraction.py $wavs_dir $commonvoicedir/train.csv extraction_dir/compare 
python prepare/common_voice_parallel_infile_extraction.py $wavs_dir $commonvoicedir/dev.csv extraction_dir/compare 
python prepare/common_voice_parallel_infile_extraction.py $wavs_dir $commonvoicedir/test.csv extraction_dir/compare 
python prepare/common_voice_parallel_infile_extraction_gemaps.py $wavs_dir $commonvoicedir/train.csv extractin_dir/gemaps
python prepare/common_voice_parallel_infile_extraction_gemaps.py $wavs_dir $commonvoicedir/dev.csv extractin_dir/gemaps
python prepare/common_voice_parallel_infile_extraction_gemaps.py $wavs_dir $commonvoicedir/test.csv extractin_dir/gemaps
python prepare/add_LLD_workers.py prepare/preparation extraction_dir/gemaps training_csvs
python prepare/add_LLD_workers_compare.py training_csvs/ extraction_dir/compare final_training_csvs
python prepare/normalizing_inputs.py final_training_csvs normalized_training_csvs
