while true; do
    # 同步命令
    aws s3 sync /mnt/localssd/yuanhaoc/video2seg/ s3://cis-intern-2024/yuanhao_cai/VideoEdit/label/video2seg/
    
    # 等待 4 小时 (4 小时 = 14400 秒)
    sleep 3600
done