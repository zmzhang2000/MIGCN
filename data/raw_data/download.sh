# The data download script has not been fully tested, and may exist some problems.
# It aims to provide the data source.


# GloVe and StanfordCoreNLP
cd data/raw_data

# GloVe embedding from https://nlp.stanford.edu/projects/glove/
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
# StanfordCoreNLP https://github.com/Lynten/stanford-corenlp
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip

cd ../../


# ActivityNet
mkdir data/raw_data/activitynet
cd data/raw_data/activitynet

# ActivityNet annotations from https://cs.stanford.edu/people/ranjaykrishna/densevid/
wget https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
unzip captions.zip -d captions
# Activitynet C3D features from https://cs.stanford.edu/people/ranjaykrishna/densevid/
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-00
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-01
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-02
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-03
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-04
wget http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-05
cat activitynet_v1-3.part-* > temp.zip && unzip temp.zip

cd ../../../


# Charades-STA
mkdir data/raw_data/charades
cd data/raw_data/charades

# Charades-STA movie length information from https://github.com/WuJie1010/Temporally-language-grounding
wget https://drive.google.com/open?id=16rFGu9rnhnH-WQeUmN7VtMgljrhGspll
tar -xf ref_info.tar
# Charades-STA annotations from https://github.com/jiyanggao/TALL
wget https://drive.google.com/file/d/1ZjG7wJpPSMIBYnW7BAG2u9VVEoNvFm5c/view?usp=sharing
wget https://drive.google.com/file/d/1QG4MXFkoj6JFU0YK5olTY75xTARKSW5e/view?usp=sharing
# Charades-STA C3D and Two-Stream feature from https://github.com/WuJie1010/TSP-PRL
wget https://drive.google.com/file/d/16nWy7rL8z3uvBDWU6OMSD1wmytZSbnrx/view?usp=sharing
wget https://drive.google.com/file/d/1u7y4Z-fIRA_jqBP4fK-8zLznWJs3PBg0/view?usp=sharing
wget https://drive.google.com/file/d/1k4tamDpFsPoSiBSpOqgxpiD7iLE5shcW/view?usp=sharing
wget https://drive.google.com/file/d/1OizKfbLSl_ezdgkN_sKdhQnlgUMn3ay_/view?usp=sharing
cat features.tar.gz* > features.tar.gz && tar -xzf features.tar.gz
# Charades-STA I3D feature extracted by https://github.com/piergiaj/pytorch-i3d
wget https://drive.google.com/file/d/1jLMgck27U7pIkcEDO6mVIlepti9SW9nd/view?usp=sharing

cd ../../../