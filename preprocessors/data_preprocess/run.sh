#! /bin/bash
set -ex
step=1

n_groups_sad=4
group_sad=1
export CUDA_VISIBLE_DEVICES=0

input_dir=$1
output_dir=$2
prefix=$3
n_groups=$4

mkdir -p "$output_dir/$prefix"

if [ $step -le 0 ];then
    source /mnt/nas1/kaixuan/anaconda3/etc/profile.d/conda.sh && conda activate py38
    start=$(date +%s)
    # 获取文件总数
    total_files=$(find "$input_dir" -type f -size +1k | wc -l)
    # 如果没有文件，则退出
    if [ "$total_files" -eq 0 ]; then
        echo "No files found in $input_dir"
        exit 0
    fi

    # 计算每组文件数
    files_per_group=$(( (total_files + n_groups - 1) / n_groups )) # 更精确的向上取整

    # 清理旧的分组文件
    rm -f "$output_dir/$prefix/file_group_"*.txt

    # 使用 while read 循环进行分组
    group=1
    count=0
    find "$input_dir" -type f -size +1k | while IFS= read -r file; do
        # 将文件名追加到对应的分组文件中
        echo "$file" >> "$output_dir/$prefix/file_group_$group.txt"

        count=$((count + 1))
        if [ $count -ge $files_per_group ] && [ $group -lt $n_groups ]; then
            count=0
            group=$((group + 1))
        fi
    done

    # 并行运行Python脚本
    # conversion: 文件转换成 16bit/24khz
    for group_file in "$output_dir/$prefix"/file_group_*.txt; do
        # 检查文件是否存在且不为空
        if [ -s "$group_file" ]; then
            echo "Start converting group: $group_file"
            python conversion.py -i "$input_dir" -o "$output_dir" -pre "$prefix" -f "$group_file" &
        fi
    done
    wait


    
    # normalization:

    for group_file in "$output_dir/$prefix"/file_group_*.txt; do
        # 检查文件是否存在且不为空
        if [ -s "$group_file" ]; then
            echo "Start normalize group: $group_file"
            python normalization.py -i $input_dir -o $output_dir -pre $prefix -f "$group_file" &
            #python normalization.py -i $input_dir -o $output_dir -pre $prefix -f $output_dir/$prefix/file_group_$i.txt &
        fi
    done
    wait
    

    end=$(date +%s)
    take=$(( end - start ))
    echo Time taken to execute commands is ${take} seconds.
fi


# SAD and force split

if [ $step -le 1 ];then
    step=1
    . /mnt/nas1/zhangying/anaconda3/bin/activate pipeline_env
    source /mnt/nas1/kaixuan/anaconda3/etc/profile.d/conda.sh && conda activate py38
    python pipeline.py -i $input_dir -o $output_dir -pre $prefix -s $step -n $n_groups_sad -g $group_sad
fi
<< EOF
# SNR
if [ $step -le 2 ];then
    source /mnt/nas1/kaixuan/anaconda3/etc/profile.d/conda.sh && conda activate cuda10
    python pipeline2.py -o $output_dir -pre $prefix  -n $n_groups_sad -g $group_sad
fi


# ASR and alignment
if [ $step -le 3 ];then
    step=3
    source /mnt/nas1/kaixuan/anaconda3/etc/profile.d/conda.sh && conda activate py38
    python pipeline.py -i $input_dir -o $output_dir -pre $prefix -s $step -n $n_groups_sad -g $group_sad
fi
EOF

