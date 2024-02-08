# Classification

# Main Result
python eval_cls.py --task cls --class_num 0 --exp_name main_result --num_points 10000
python eval_cls.py --task cls --class_num 1 --exp_name main_result --num_points 10000
python eval_cls.py --task cls --class_num 2 --exp_name main_result --num_points 10000

# Effect Of Rotations
python eval_cls.py --task cls --class_num 0 --exp_name low_rotate_points --num_points 10000 --rotate 1 --x 15 --y 15 --z 15
python eval_cls.py --task cls --class_num 0 --exp_name med_rotate_points --num_points 10000 --rotate 1 --x 45 --y 45 --z 45 
python eval_cls.py --task cls --class_num 0 --exp_name high_rotate_points --num_points 10000 --rotate 1 --x 90 --y 60 --z 0

python eval_cls.py --task cls --class_num 1 --exp_name low_rotate_points --num_points 10000 --rotate 1 --x 15 --y 15 --z 15
python eval_cls.py --task cls --class_num 1 --exp_name med_rotate_points --num_points 10000 --rotate 1 --x 45 --y 45 --z 45 
python eval_cls.py --task cls --class_num 1 --exp_name high_rotate_points --num_points 10000 --rotate 1 --x 90 --y 60 --z 0


python eval_cls.py --task cls --class_num 2 --exp_name low_rotate_points --num_points 10000 --rotate 1 --x 15 --y 15 --z 15
python eval_cls.py --task cls --class_num 2 --exp_name med_rotate_points --num_points 10000 --rotate 1 --x 45 --y 45 --z 45 
python eval_cls.py --task cls --class_num 2 --exp_name high_rotate_points --num_points 10000 --rotate 1 --x 90 --y 60 --z 0

# Effect Of Number of Points

python eval_cls.py --task cls --class_num 0 --exp_name xtr_low_number_points --num_points 50 
python eval_cls.py --task cls --class_num 0 --exp_name very_low_number_points --num_points 100 
python eval_cls.py --task cls --class_num 0 --exp_name low_number_points --num_points 1000 
python eval_cls.py --task cls --class_num 0 --exp_name med_number_points --num_points 5000  
python eval_cls.py --task cls --class_num 0 --exp_name high_number_points --num_points 10000

python eval_cls.py --task cls  --class_num 1 --exp_name xtr_low_number_points --num_points 50 
python eval_cls.py --task cls  --class_num 1 --exp_name very_low_number_points --num_points 100 
python eval_cls.py --task cls  --class_num 1 --exp_name low_number_points --num_points 1000 
python eval_cls.py --task cls  --class_num 1 --exp_name med_number_points --num_points 5000  
python eval_cls.py --task cls  --class_num 1 --exp_name high_number_points --num_points 10000

python eval_cls.py --task cls --class_num 2 --exp_name xtr_low_number_points --num_points 50 
python eval_cls.py --task cls --class_num 2 --exp_name very_low_number_points --num_points 100 
python eval_cls.py --task cls --class_num 2 --exp_name low_number_points --num_points 1000 
python eval_cls.py --task cls --class_num 2 --exp_name med_number_points --num_points 5000  
python eval_cls.py --task cls --class_num 2 --exp_name high_number_points --num_points 10000

# Segmentation

# Main Result
python eval_seg.py --task seg --exp_name main_result --num_points 10000

# Effect Of Rotations
python eval_seg.py --task seg --exp_name low_rotate_points --num_points 10000 --rotate 1 --x 15 --y 15 --z 15
python eval_seg.py --task seg --exp_name med_rotate_points --num_points 10000 --rotate 1 --x 45 --y 45 --z 45 
python eval_seg.py --task seg --exp_name high_rotate_points --num_points 10000 --rotate 1 --x 90 --y 60 --z 0

# Effect Of Number of Points

python eval_seg.py --task seg --exp_name xtr_low_number_points --num_points 50 
python eval_seg.py --task seg --exp_name very_low_number_points --num_points 100 
python eval_seg.py --task seg --exp_name low_number_points --num_points 1000 
python eval_seg.py --task seg --exp_name med_number_points --num_points 5000  
python eval_seg.py --task seg --exp_name high_number_points --num_points 10000