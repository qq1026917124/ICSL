#! /bin/bash
#
if [[ $# -lt 2 ]]; then
	echo "Usage: $0 output_directory task_source_file"
exit 1
fi
echo "Output to $1"
python3 nohup python data/gen_maze_record.py \
	--output_path $1  	 \
	--task_source NEW  	 \
    --max_steps 10000    \   
    --n_range 15,16      \
    --start_index 0      \
    --epochs 256     	 \
    --workers 256     	 \

