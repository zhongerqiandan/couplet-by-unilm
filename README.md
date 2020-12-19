# couplet-by-unilm
我是一个无情的对对子机器。基于seq2seq模型，输入一个句子，输出对应的对子，基于topk采样，因此每次生成的对子都不相同。本项目仅供娱乐，数据集来自于https://github.com/wb14123/couplet-dataset
## 训练自己的对对子模型
1. 环境准备：python3.6, tensorflow-gpu 1.14.0, bert4keras
2. 下载[chinese_wwm_ext_bert模型](https://github.com/ymcui/Chinese-BERT-wwm)
3. 将run_unilm_couplet.py中的config_path, checkpoint_path, dict_path, model_save_path以及数据路径更改为自己的路径
4. 如果没有gpu就令os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
5. 
```
python run_unilm_couplet.py
```

## 效果
等模型训练结束后，执行test.py文件可以简单查看模型效果
![pic/WechatIMG5.png](https://github.com/zhongerqiandan/couplet-by-unilm/blob/master/pics/WechatIMG5.png)
![pic/WechatIMG6.png](https://github.com/zhongerqiandan/couplet-by-unilm/blob/master/pics/WechatIMG6.png)
![pic/WechatIMG7.png](https://github.com/zhongerqiandan/couplet-by-unilm/blob/master/pics/WechatIMG7.png)
![pic/WechatIMG8.png](https://github.com/zhongerqiandan/couplet-by-unilm/blob/master/pics/WechatIMG8.png)
![pic/WechatIMG9.png](https://github.com/zhongerqiandan/couplet-by-unilm/blob/master/pics/WechatIMG9.png)

