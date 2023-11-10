# 2023 Spring NTUEE CV - Final Project (Group 13)
### code structure
In this submission, there are one `.txt` file, four `.py` files, two default sub directories, which cantain all `.pkl` and `.png` files we used in our algorithm respectivelly, and three potential directories (based on what kind of experiment that user will implement). And, of course, current README `.md` file. Below is the code structure.
<pre>
├──submission
    ├── main.py
    ├── pred.py
    ├── eval.py
    ├── utils.py
    ├── requirements.txt
    ├── README.md
    ├── pkl
        ├── camera_dict.pkl
        ├── camera_dict_test.pkl
        ├── new_test_init.pkl
        ├── new_train_init.pkl
        ├── sub_map.pkl
        └── test_sub_map.pkl
    ├── mask
        ├── b_below_mask.png
        ├── b_dilated_mask.png
        ├── f_below_mask.png
        ├── f_dilated_mask.png
        ├── fr_below_mask.png
        ├── fr_dilated_mask.png
        ├── fl_below_mask.png
        └── fl_dilated_mask.png
    ├── ITRI_dataset
        └── ...
    ├── ITRI_DLC
        └── ...
    └── ITRI_DLC2
        └── ...
</pre>


```
main.py --> the main body of proposed algorithm, generating point cloud.
pred.py --> based on point cloud, calculating the displacement of two direction.
eval.py --> evaluate the performance of result
utils.py --> library
```

### how to use
 * First, use below command to install needed library
```sh
$ pip install -r requirements.txt
```
* Second, copy all dataset directory into current directory. (For testing and reproduction, <b>ITRI_DLC</b> and <b>ITRI_DLC2</b> are needed).
* Third, run below commands orderly.
```sh
$ pyhton3 main.py # point cloud result saved in ./result/exp_name/
$ python3 pred.py # pred_pose.txt saved in ./result/exp_name/solution/test1 or test2/
$ python3 eval.py # score will printed on cmd
```

### Appendix
If there are need of checking result of training set (seq1 to seq3), please copy ITRI_dataset into current directory. And run below command orderly. The result will be very similar to testing, except for the number of sequence.
```sh
$ pyhton3 main.py --mode train
$ python3 pred.py --mode train --result_name TRAIN_ST_0
$ python3 eval.py --mode train --result_name TRAIN_ST_0
```