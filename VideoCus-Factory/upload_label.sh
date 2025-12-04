while true; do
    # 同步命令
    aws s3 sync /mnt/localssd/yuanhaoc/ s3://cis-intern-2024/yuanhao_cai/VideoEdit/label/
    
    # 等待 4 小时 (4 小时 = 14400 秒)
    sleep 14400
done