sudo chmod -R 777 /sensei-fs/users/xic/
sudo chmod -R 777 /home/
sudo chmod -R 777 /sensei-fs/users/xic/project/codebase/segment-anything-2/

ln -s /sensei-fs/users/xic /home/xic
source /sensei-fs/users/xic/opt/sam2/bin/activate
cd /sensei-fs/users/xic/project/codebase/segment-anything-2
rm -rf /home/xic/.cache/

python process_data_v1.py -G 0 -k 5 -N 8
