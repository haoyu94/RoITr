for N_POINTS in 250 500 1000 2500 5000
do
python registration/evaluate_registration_c2f.py --source_path ./snapshot/tdmatch_ripoint_transformer_test/3DMatch --benchmark 3DMatch --n_points $N_POINTS
done
