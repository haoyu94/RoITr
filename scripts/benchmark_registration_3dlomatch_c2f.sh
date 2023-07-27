for N_POINTS in 250 500 1000 2500 5000
do
python registration/evaluate_registration_c2f.py --source_path ./snapshot/tdmatch_ripoint_transformer_test/3DLoMatch --benchmark 3DLoMatch --n_points $N_POINTS
done
