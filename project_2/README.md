# Dependency Parser on StanfordNLP



## Requirements
All follows requirements in `Pipfile`. <br>
Note that the Python version is required >= 3.6 by stanfordnlp. And other dependencies would be installed while installing `stanfordnlp` package.


<br><br>
## Prepare Word Embedding
Use Fasttext Latest `Word Vector for 157 languages`.
[Download Here](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz).
After downloading and extracting, move the file cc.zh.300.vec to `./stanfordnlp/wordvecdir/ChineseT/` and rename it to `zh.vectors.xz`.


<br><br>
## Prepare Training and Development Data
Training data and development data is already prepared in `data/depparse/`.There's no need to prepare data manully if no other requirements are requested. Yet, if need, please follow the instruments below. <br>
Put training data and development data into `data/depparse/`. <br>
Rename them following
* training data -> `zh_chinese.train.in.conllu` <br>
* development data -> `zh_chinese.dev.in.conllu` and `zh_chinese.dev.gold.conllu`


<br><br>
## Training
Enter into the working directory `./stanfordnlp`. <br>
To start training, run <br>
```console
$ ./scripts/run_depparse.sh Chinese gold --batch_size 1000 --tag_emb_dim 200 --word_emb_dim 150 --no_char --lr 0.001 --max_steps 15000
```
When training, predicted file `data/depparse/zh_chinese.dev.pred.conllu` would be produced. <br>
Overall, the `UAS` score on development data must be >= 0.76, if not, there may be something wrong.



<br><br>
## Prediction
Enter into the working directory `./stanfordnlp`. <br>
To get predicted file `data/depparse/zh_chinese.dev.pred.conllu` on your own, run <br>
```console
$ ./scripts/run_depparse.pred.sh Chinese gold
```
