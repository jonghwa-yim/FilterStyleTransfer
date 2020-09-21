#!/usr/bin/env bash

modelPath='models/models_default/model_epoch_8.pth'
# modelPath='models/flumen22_model_epoch_8.pth'
# modelPath='models/models_aleatoric2_cont/model_epoch_7.pth'
imageSize=400
targetGTDir='my_images/sample_dataset/test/style'


# ==============================

styleImageDir='/f_data1/DataCollections/MyCam/real_filters_w_real_pictures/normal_users_style'
targetImageDir='/f_data1/DataCollections/MyCam/real_filters_w_real_pictures/normal_users_original'
stylesOrgDir='/f_data1/DataCollections/MyCam/real_filters_w_real_pictures/normal_users_original'
targetGTDir='/f_data1/DataCollections/MyCam/real_filters_w_real_pictures/normal_users_style'
outputDir='/f_data1/DataCollections/MyCam/real_filters_w_real_pictures/output/200416_oldmodel_tmp'

python main_stylizer_unseen_testing.py --uncertainty=aleatoric_combined -S $styleImageDir -T $targetImageDir -m $modelPath -s $imageSize --outputDir=$outputDir --styleImageOrgDir=$stylesOrgDir --targetGTDir=$targetGTDir --reg_weight=0.1 --outputMode=0


# ==============================

#styleImageDir='my_images/for_paper/styles7_3_pilgram_selected'
#targetImageDir='my_images/for_paper/targets7_2'
#stylesOrgDir='my_images/for_paper/styles7'
#targetGTDir='my_images/for_paper/targets7_2_piland3rd_gt'
#outputDir='my_images/output/200519_7_3_oldmodel_default'
#
#python main_stylizer.py -O -S $styleImageDir -T $targetImageDir -m $modelPath -s $imageSize --outputDir=$outputDir --styleImageOrgDir=$stylesOrgDir --targetGTDir=$targetGTDir --reg_weight=0.1 --outputMode=0
#
#styleImageDir='my_images/for_paper/styles10'
#targetImageDir='my_images/for_paper/targets10'
#outputDir='my_images/output/200519_10_oldmodel_default'
#
#python main_stylizer.py -O -S $styleImageDir -T $targetImageDir -m $modelPath -s $imageSize --outputDir=$outputDir --targetGTDir=$targetGTDir --reg_weight=0.1 --outputMode=0 # --variance
#
#styleImageDir='my_images/for_paper/styles8_coco'
#targetImageDir='my_images/for_paper/targets7_2'
#outputDir='my_images/output/200519_8_to72_oldmodel_default'
#
#python main_stylizer.py -O -S $styleImageDir -T $targetImageDir -m $modelPath -s $imageSize --outputDir=$outputDir --targetGTDir=$targetGTDir --reg_weight=0.1 --outputMode=0 # --variance
#
#styleImageDir='my_images/for_paper/styles4_wj'
#targetImageDir='my_images/for_paper/targets4_wj'
#outputDir='my_images/output/200519_4_oldmodel_default'
#
#python main_stylizer.py -O -S $styleImageDir -T $targetImageDir -m $modelPath -s $imageSize --outputDir=$outputDir --targetGTDir=$targetGTDir --reg_weight=0.1 --outputMode=0 # --variance
#
#styleImageDir='my_images/for_paper/styles5_ux'
#targetImageDir='my_images/for_paper/targets5_ux'
#outputDir='my_images/output/200519_5_oldmodel_default'
#
#python main_stylizer.py -O -S $styleImageDir -T $targetImageDir -m $modelPath -s $imageSize --outputDir=$outputDir --targetGTDir=$targetGTDir --reg_weight=0.1 --outputMode=0 # --variance
#
#styleImageDir='my_images/for_paper/styles6_recol'
#targetImageDir='my_images/for_paper/targets6_recol'
#outputDir='my_images/output/200519_6_oldmodel_default'
#
#python main_stylizer.py -O -S $styleImageDir -T $targetImageDir -m $modelPath -s $imageSize --outputDir=$outputDir --targetGTDir=$targetGTDir --reg_weight=0.1 --outputMode=0 #--variance
