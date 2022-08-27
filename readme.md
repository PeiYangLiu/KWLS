#KWLS

This code is based on the open-source code of BERT.

Download and unzip [BERT](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip)

Example data can be found in ```./data```

Replace vocab_file, bert_config_file, init_checkpoint in train_kwls.sh with BERT dir you unzip.

Run this code by ```sh train_kwls.sh```

Soft Label will be generated in ```./data/soft_label.npy```

The prediction will be found in ```./data/output```