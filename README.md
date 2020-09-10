## GraphMemVOS
Code for ECCV 2020 spotlight paper: Video Object Segmentation with Episodic Graph Memory Networks
#
![](../master/eccv-framework.png)

## Testing
1. Install python (3.6.5), pytorch (version:1.0.1) and requirements in the requirements.txt files. Download the DAVIS-2017 dataset.

2. Download the pretrained model from [googledrive](https://drive.google.com/file/d/1HO6wlhAYSuBDx4Cnb8efQyLs357ycDz2/view?usp=sharing) and put it into the workspace_STM_alpha files. 

3. Run 'run_graph_memory_test.sh' and change the davis dataset path, pretrainde model path and result path and the paths in local_config.py.

The segmentation results can be download from [googledrive](https://drive.google.com/file/d/1CYDtlQNlq2ZEKI29LLOb8TZq4eSpiRPJ/view?usp=sharing).

## Results
1. **DAVIS** ( Val 2017):

In the inference stage, we ran using the default size of DAVIS (480p).

**Mean J&F** |  **J score** | **F score** | 
---------|  :---------: | :---------: 
 **82.8** | **80.0** | **85.2** |
 
 2. **YouTube-VOS** (Val 2018):
  **J Seen** | **F Seen** | **J Unseen** | **F Unseen** | **Mean** |
---------|  :---------: | :---------: |  :---------: | :---------: 
 **80.7** | **85.1** | **74.0** | **80.9** | **80.2** |
 
 3. **DAVIS-2016**:
  **J score** | **F score** | **Mean T** |
---------|  :---------: | :---------: 
 **82.5** | **81.2** | **19.8** |
 
 4. **Youtube-Objects**:
  **Airplane** | **Bird** | **Boat** |  **Car** | **Cat** | **Cow** |  **Dog** | **Horse** | **Motorbike** |**Train** |**Mean** |
---------|  :---------: | :---------: |:---------: | :---------: |:---------: | :---------: |:---------: | :---------: | :---------: | :---------: 
 **86.1** | **75.7** | **68.6** |**82.4** | **65.9** | **70.5** |**77.1** | **72.2** | **63.8** |**47.8** | **71.4** |

## Citation

If you find the code and dataset useful in your research, please consider citing:
```
@inproceedings{lu2020video,  
 title={Video Object Segmentation with Episodic Graph Memory Networks},  
 author={Lu, Xiankai and Wang, Wenguan and Martin, Danelljan and Zhou, Tianfei and Shen, Jianbing and Luc, Van Gool},  
 booktitle={ECCV},  
 year={2020}  
}
```
## Other related projects/papers:

1. Zero-shot Video Object Segmentation via Attentive Graph Neural Networks, ICCV 2019 (https://github.com/carrierlxk/AGNN) 

## Acknowledge

1. Video object segmentation using space-time memory networks, ICCV 2019 (https://github.com/seoungwugoh/STM)
2. A Generative Appearance Model for End-to-End Video Object Segmentation, CVPR2019 (https://github.com/joakimjohnander/agame-vos)
3. https://github.com/lyxok1/STM-Training

Any comments, please email: carrierlxk@gmail.com



