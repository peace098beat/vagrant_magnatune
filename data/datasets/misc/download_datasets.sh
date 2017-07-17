#MP3データの入手
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003

# 分割zipファイルを統合、解凍する
cat mp3.zip* > ./music.zip
unzip music.zip

# タグデータの入手
wget http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv