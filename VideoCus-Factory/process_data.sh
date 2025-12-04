source /sensei-fs/users/xic/opt/sam2/bin/activate


python process_data_v1_local_valid.py -G 0 -k 1 -N 8 &
python process_data_v1_local_valid.py -G 1 -k 2 -N 8 &
python process_data_v1_local_valid.py -G 2 -k 3 -N 8 &
python process_data_v1_local_valid.py -G 3 -k 4 -N 8 &
python process_data_v1_local_valid.py -G 4 -k 5 -N 8 &
python process_data_v1_local_valid.py -G 5 -k 6 -N 8 &
python process_data_v1_local_valid.py -G 6 -k 7 -N 8 &
python process_data_v1_local_valid.py -G 7 -k 8 -N 8 &
wait
