


min_la=40.500
max_la=40.900
min_lo=-74.396
max_lo=-73.746
step_dis=0.005 #500m [81,131]
seq_len=500
input_size=[round((max_la-min_la)/step_dis)+1,round((max_lo-min_lo)/step_dis)+1]
read_nworker=8
epochs=100
Qquan=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
#percentiles=[0.15,0.5,0.85]
percentiles=[0,0.5,1]
cost_rate=[0.452]
#cost_rate=[0.2376,0.3564]
alo=3