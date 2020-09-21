#!/usr/bin/env bash

#stylizedTrainDir=/f_data1/TrainingSets/coco_stylized/images/train2017
#orgTrainDir=/f_data1/TrainingSets/coco/images/train2017/
#stylizedTestDir=/f_data1/TrainingSets/coco_stylized/images/val2017
#orgTestDir=/f_data1/TrainingSets/coco/images/val2017/
#trainSet=/f_data1/TrainingSets/coco_stylized/images/train2017.txt
#valSet=/f_data1/TrainingSets/coco_stylized/images/val2017.txt
#outputDir=models_aleatoric2
#batchSize=16
#nEpochs=20
#gpu=0

stylizedTrainDir=/f_data1/TrainingSets/mycam_flumen_v2.3/train
orgTrainDir=/f_data1/TrainingSets/mycam_flumen_v2.3/train
stylizedTestDir=/f_data1/TrainingSets/mycam_flumen_v2.3/test
orgTestDir=/f_data1/TrainingSets/mycam_flumen_v2.3/test
trainSet=/f_data1/TrainingSets/mycam_flumen_v2.3/train_files.txt
valSet=/f_data1/TrainingSets/mycam_flumen_v2.3/test_files.txt
# outputDir=models/unetmod_test_2
outputDir=models/transformernet6_1
batchSize=16
nEpochs=12
gpu=0

python main_inv_function.py --lr=0.002 --uncertainty=default --print-freq=2000 --gpu=$gpu --batchSize=$batchSize --outputDir=$outputDir --stylizedTrainDir=$stylizedTrainDir --orgTrainDir=$orgTrainDir --stylizedTestDir=$stylizedTestDir --orgTestDir=$orgTestDir --trainSet=$trainSet --valSet=$valSet --nEpochs=$nEpochs



#outputDir=models/transformernet3_2_aleatoric
#python main_inv_function.py --lr=0.001 --uncertainty=aleatoric --print-freq=2000 --gpu=$gpu --batchSize=$batchSize --outputDir=$outputDir --stylizedTrainDir=$stylizedTrainDir --orgTrainDir=$orgTrainDir --stylizedTestDir=$stylizedTestDir --orgTestDir=$orgTestDir --trainSet=$trainSet --valSet=$valSet --nEpochs=$nEpochs


