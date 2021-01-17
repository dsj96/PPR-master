# PPR-master
The implementation of the PPR model(Personalized POI Recommendation).
```Personalized POI Recommendation:Spatio-Temporal Representation Learning withSocial Tie``` was accepted by DASFAA 2021.
You can train and evaluate the model: ```./train_PPR.sh```

# Usage:
## Install dependencies 
```pip install -r requirements.txt```
## Function
1. ```gen_graph.py``` file is used for heterogeneous graph construction. Parameter ```theta``` is $\theta$ in Equ.2, and ```epsilon``` is $\varepsilon$ in Equ.6.
2. ```reconstruct.cpp``` file is used for densifying graph. Parameter ```-threshold``` is $\rho$.
3. ```line.cpp``` file is used for learning latent representations. Parameter ```-size``` is embedding dim $d$.
4. ```train.py``` file is used for training and evaluating the spatio-temporal neural network. Parameter ```DELT_T``` is the time constraint $\tau$, and ```INPUT_SIZE/2``` is the embedding dim $d$. You could also change the ```HIDDEN_SIZE, EPOCH, LR, LAYERS OR TEST_SAMPLE_NUM```. 

# Data
In our experiments, the Foursquare datasets are from https://sites.google.com/site/dbhongzhi/. And the Gowalla and Brightkite dataset are from https://snap.stanford.edu/data/loc-gowalla.html and http://snap.stanford.edu/data/loc-Gowalla.html.
## Data Format
We utilize the first 80% chronological check-ins of each useras the training set, the remaining 20% as the test data.

train_checkin_file.txt and test_checkin_file.txt :
```<USER ID> \t <CHECKIN TIME> \t <POI ID> \t <LONGITUDE> \t <LATITUDE>```

friendship_file.txt : ```<USER ID>,<USER ID>```
