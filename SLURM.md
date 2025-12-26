&nbsp;

<div align="center">

<h2> Regular Commands for SLURM</h2> 

</div>


install tmux:
```
  sudo apt install tmux
```


&nbsp;

### 1. Interactive
#### (a) Launch an interactive node:
```py
  salloc -A genai_interns -p a100 --nodes=1 --gpus-per-node=8 --cpus-per-task=80 --time=72:00:00 --qos=a100_genai_interns_high
  salloc -A genai_interns -p a100 --nodes=1 --gpus-per-node=8 --cpus-per-task=80 --time=72:00:00 --qos=a100_genai_shared
```
After launching, the GPUs will be allocated to the AWS server. You can directly run `nvidia-smi`

先在提交节点上边开一个 tmux, 然后使用 salloc 之后就不会因为关闭终端而导致掉线了
```py
  (genai-cloud1) yuanhaoc@submit-2:/data/home/yuanhaoc/DiffSynth-Studio$ salloc --nodes=1 --gpus-per-node=8 --time=168:00:00 --cpus-per-task=80
  salloc: Pending job allocation 2585849
  salloc: job 2585849 queued and waiting for resources
  salloc: job 2585849 has been allocated resources
  salloc: Granted job allocation 2585849
  salloc: Waiting for resource configuration
  salloc: Nodes a100-st-p4de24xlarge-11 are ready for job
```

然后新建一个终端, 用 srun 接入计算结点, 然后就可以复制 GPU 环境来开 tmux 了，有一点绕
```sh
srun --jobid="${SLURM_JOB_ID}" --pty /bin/bash -l
srun --jobid=2588028 --pty /bin/bash -l
srun --jobid=2588029 --pty /bin/bash -l
srun --jobid=2588030 --pty /bin/bash -l
srun --jobid=2588031 --pty /bin/bash -l
srun --jobid=2588112 --pty /bin/bash -l
srun --jobid=2588113 --pty /bin/bash -l
srun --jobid=2588194 --pty /bin/bash -l
srun --jobid=2588195 --pty /bin/bash -l
srun --jobid=2588200 --pty /bin/bash -l
srun --jobid=2588204 --pty /bin/bash -l
srun --jobid=2588350 --pty /bin/bash -l
srun --jobid=2589062 --pty /bin/bash -l
srun --jobid=2598588 --pty /bin/bash -l
srun --jobid=2602788 --pty /bin/bash -l
srun --jobid=2603652 --pty /bin/bash -l
srun --jobid=2603657 --pty /bin/bash -l
srun --jobid=2622112 --pty /bin/bash -l
srun --jobid=2623285 --pty /bin/bash -l
srun --jobid=113 --pty /bin/bash -l
srun --jobid=142 --pty /bin/bash -l
srun --jobid=143 --pty /bin/bash -l
```

#### (b) `Ctrl + D` 即退出服务器, 尽量不要用这个, 即便在 tmux 里面也别用

#### (c) 由于 SLURM allocates GPU 计算资源以后, 新建 terminal 并不会有 GPU 的上下文, 所以需要使用 tmux 分屏来保持这个上下文, 一些常用的分屏命令:
```sh
  水平分屏： Ctrl-b "
  垂直分屏： Ctrl-b %
  在各个窗格之间切换	Ctrl-b 方向键（←/→/↑/↓）
  关闭当前窗格	exit 或 Ctrl-d
```
新建几个 tmux 也行

#### (d) tmux 好像是会自动结束的, 后边可以多用 sbatch

#### (e) 重新接入 SLURM 分配的计算节点, 先在计算节点内部 bash, 然后再在 split 的窗口处 bash
```sh
  echo $SLURM_JOB_ID 
  srun --jobid=$SLURM_JOB_ID --pty /bin/bash -l
  # 也可以 bash 一下提前写好的 tmux file
  # tmux source-file ~/.tmux.conf
```

#### (f) 查看当前的 SLURM 使用情况
```sh
  squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %.10q"
```

#### (g) 任务取消
```sh
  scancel -u yuanhaoc
```