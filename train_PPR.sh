g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result line.cpp -o line -lgsl -lm -lgslcblas
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result reconstruct.cpp -o reconstruct
python3 gen_graph.py --input_path dataset/toyset/ --epsilon 0.5 --theta 24.0
python3 preprocess_edges.py dataset/toyset/G_edges.txt dataset/toyset/net_edges.txt

./reconstruct -train dataset/toyset/net_edges.txt -output dataset/toyset/net_dense.txt -depth 2 -threshold -400
./line -train dataset/toyset/net_dense.txt -output dataset/toyset/vec_2nd_wo_norm.txt -binary 0 -size 16 -order 2 -negative 5 -samples 1000 -threads 15
python3 train.py
