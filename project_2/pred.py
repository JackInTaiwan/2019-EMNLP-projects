import stanfordnlp
# stanfordnlp.download('zh')   # This downloads the English models for the neural pipeline
nlp = stanfordnlp.Pipeline(lang='zh', models_dir='./', depparse_vec_filename='./stanfordnlp/wordvecdir/ChineseT/zh.vectors.xz', tokenize_pretokenized=True)
doc = nlp("目前 全 市 上亿 元 的 三资 工业 企业 已 达 五十一 家 ， 而 在 一九九四年 仅 有 二十三 家 。")
print(doc.sentences)
doc.sentences[0].print_dependencies()