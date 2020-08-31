#python eval_DAVIS_graph_memory.py -g '4' -s val -y 17 -D /raid/DAVIS/DAVIS-2017/DAVIS-train-val
#python eval_DAVIS.py -g '0' -s val -y 16 -D /media/xiankai/Data/segmentation/DAVIS-2016
python runfiles/eval_DAVIS_graph_memory.py -c './workspace_STM_alpha/main_runfile_graph_memory.pth.tar'
