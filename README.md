# AI_Project - Breast Cancer Segmentation – Semi Supervised AI System

This project focuses on the development of a semi-supervised artificial intelligence system for 
automatic tumor region generation in breast MRI images. The project is based on a medical imaging 
dataset containing more than 620 breast MRI patient cases, where only a small subset of 
approximately 99 patients includes manually annotated ROI (Region of Interest) tumor masks. 

The problem we want to solve is the limited availability of annotated medical imaging data. In medical 
AI applications, manual segmentation of tumors is a time-consuming and expensive task that requires 
expert radiologists. As a result, many medical datasets contain only a small number of labeled images, 
while the majority of the available data remains unlabeled and cannot be directly used for supervised 
segmentation tasks. 

The final goal of this project is to develop a semi-supervised segmentation pipeline capable of learning 
tumor patterns from the available labeled MRI scans and automatically generating pseudo-ROI masks 
for unlabeled patient data. The generated masks will then be analyzed and evaluated using 
segmentation metrics and visual comparison methods. In addition to the AI component, the project 
also includes data acquisition, preprocessing, exploratory data analysis, visualization, and model 
evaluation to create a complete advanced data analytics and artificial intelligence workflow.
