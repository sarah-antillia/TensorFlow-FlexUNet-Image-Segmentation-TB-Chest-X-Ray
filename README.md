<h2>TensorFlow-FlexUNet-Image-Segmentation-TB-Chest-X-Ray (2026/11/15)</h2>
<h3>TB-Chest-X-Ray: AI Generated Pseudo Masks Segmentation Challenge</h3>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the second experiment of Image Segmentation for <b>TB (Tuberculosis)-Chest-X-Ray </b>
 based on
our <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
TensorFlowFlexUNet Model</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>) 
 and a 512x512-pixel PNG <a href="https://drive.google.com/file/d/1ARgKwVFHnkw8nqnVNYDEUCeYXugBlLJT/view?usp=sharing">
 <b>
TB-Chest-X-Ray-ImageMask-Dataset.zip
</b></a> 
with colored masks (<a href="https://creativecommons.org/licenses/by/4.0/">
Attribution 4.0 International (CC BY 4.0)</a>)
  which was derived by us from 
 the Kaggle  
<br><br>
<a href="https://www.kaggle.com/datasets/sairakiran1111/dataset-of-tuberculosis-chest-x-rays-images">
<b>
Dataset of Tuberculosis Chest X-rays Images
</b>
</a> by saira kiran1111 .
<br><br>
<hr>
<b>Actual Image Segmentation for TB-Chest-X-Ray Images of 512x512 pixels</b><br>
As shown below, the inferred masks resemble the ground-truth masks. <br>
<br>
<b>class_color_map = {Lung: cyan, Tuberculosis: yellow}</b>
<br>
<br>
<table>
<tr>
<th width="320" height="auto">Input: image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/images/TB.220.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/masks/TB.220.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test_output/TB.220.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/images/TB.660.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/masks/TB.660.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test_output/TB.660.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/images/TB.1497.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/masks/TB.1497.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test_output/TB.1497.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from the following Kaggle:<br><br>

<a href="https://www.kaggle.com/datasets/sairakiran1111/dataset-of-tuberculosis-chest-x-rays-images">
<b>
Dataset of Tuberculosis Chest X-rays Images
</b>
</a><br>
<b>Chest X-ray dataset of TB and normal cases for deep learning-based tuberculosis</b>
<br>
 by saira kiran1111
<br><br>
The following explanation (except)  was taken the website above.
<br><br>
<b>About Dataset</b><br>
This dataset contains 2,494 chest X-ray images from tuberculosis (TB) patients and 514 normal 
chest X-ray images, collected from a local hospital in Pakistan. 
The dataset includes both TB-positive cases and healthy control images to support 
the development and evaluation of artificial intelligence-based approaches for 
tuberculosis detection and chest X-ray analysis.
<br>
<br>
All images have been processed and resized to a consistent dimension to improve data uniformity 
and facilitate machine learning and deep learning applications. The dataset has been anonymized 
to protect patient privacy, and no personally identifiable information is included.
<br><br>
<b>Dataset Details</b><br>
Disease category: Tuberculosis (TB)
Imaging modality: Chest X-ray radiographs<br>
Total images: 3,008<br>
TB-positive images: 2,494<br>
Normal images: 514<br>
Image preprocessing: Resized to uniform dimensions<br>
Source: Local hospital, Pakistan<br>
Data format: Chest X-ray image files<br>
<br>
<b>Citation</b><br>
Kiran, Saira; Jabeen, Dr Ishrat (2024), “Dataset of Tuberculosis Chest X-rays Images”, <br>
Mendeley Data, V2, doi: 10.17632/8j2g3csprk.2
<br><br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/">
Attribution 4.0 International (CC BY 4.0)</a>
<br>
<br>
<h3>
2. Tuberculosis-Chest-X-Ray ImageMask Dataset
</h3>
<h3>2.1 Download ImageMask Dataset</h3>
 If you would like to train this Tuberculosis-Chest-X-Ray Segmentation model,
 please download the dataset from the Google drive:
 <a href="https://drive.google.com/file/d/1ARgKwVFHnkw8nqnVNYDEUCeYXugBlLJT/view?usp=sharing">
TB-Chest-X-Ray-ImageMask-Dataset.zip</a>.
Expand the downloaded, and put it under the <b>./dataset</b> folder.
<br>
<pre>
./dataset
└─Tuberculosis-Chest-X-Ray
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Tuberculosis-Chest-X-Ray Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/TB-Chest-X-Ray_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images in the training and valid datasets is large enough to use for the
 training set of our segmentation model.
<br>
<h3>2.2 Derivation of ImageMask Dataset</h3>
The folder structure of the original dataset  is as follows, but it contains no annotation (mask) files.
<pre>
./Dataset of Tuberculosis Chest X-rays Images
    ├─Normal Chest X-rays ( 514 files)
    │   ├─others (1).jpg
...
    │   └─others (514).jpg
    └─TB Chest X-rays     ( 2,494 files)
        ├─TB.1.jpg
...
        └─TB.2500.jpg
</pre>
<b>Step 1</b><br>
we generated a 512x512-pixel upscaled PNG master images dataset from the 256x256-pixel JPG files in <b>TB Chest X-rays</b> folder.
<br><br>
<b>Step 2</b><br>
We generated the first pseudo masks corresponding to the master images in <b> TB Chest X-rays</b> folder 
by applying a segmentation (inference) method of the pretrained model 
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Tuberculosis-Chest-X-Ray">
TensorFlow-FlexUNet-Image-Segmentation-Tuberculosis-Chest-X-Ray
</a> to the master images.
<br><br>
<b>Step 3</b><br>
We generated our own <a href="https://drive.google.com/file/d/1ARgKwVFHnkw8nqnVNYDEUCeYXugBlLJT/view?usp=sharing">TB-Chest-X-Ray-ImageMask-Dataset</a>
 from all pairs of the first pseudo masks and the master images.
<br><br>
<h3>2.3 Train Sample Images and Masks</h3>
<b>Train_sample_images</b><br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_sample_masks</b><br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3. Train TensorFlowFlexUNet Model
</h3>
 We trained the Tuberculosis-Chest-X-Ray TensorFlowFlexUNet model using the following
<a href="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/TB-Chest-X-Ray and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
This runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>
<b>Model parameters</b><br>
Defined a small <b>base_filters=16 </b> and large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers=8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
; You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 3
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_focal_dice_loss" and <a href="./src/dice_coef_multiclass.py">"dice_coef_hybrid"</a>.<br>
<pre>
[model]
loss           = "categorical_focal_dice_loss"
metrics        = ["dice_coef_hybrid"]
hybrid_alpha   = 1.6
hybrid_beta    = 0.4
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled the learning_rate_reducer callback and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with the patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>RGB Color map</b><br>
Specified RGB color map dict for Tuberculosis-Chest-X-Ray 1+2 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;Tuberculosis-Chest-X-Ray RGB color map dict for 1+2 classes.
; Background: black,  Lung:cyan,  Tuberculosis: yellow
rgb_map = {(0,0,0):0,(0,255,255):1, (255,255,0):2}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch change, the inference procedure can be called
 for 6 images in the <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
<!--
As shown below, early in the model training, the predicted masks from our FlexUNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
-->
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 18,19,20)</b><br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 38,39,40)</b><br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was stopped at epoch 40 by EarlyStoppingCallback.<br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/asset/train_console_output_at_epoch40.png" width="1024" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/eval/train_metrics.png" width="520" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4. Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/TB-Chest-X-Ray</b> folder,
and run the following bat file to evaluate the TensorFlowUNet model for Tuberculosis-Chest-X-Ray.<br>
<pre>
>./2.evaluate.bat
</pre>
This runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/asset/evaluate_console_output_at_epoch40.png" width="1024" height="auto">
<br><br>Image-Segmentation-Tuberculosis-Chest-X-Ray

<a href="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_focal_dice_loss) on this Tuberculosis-Chest-X-Ray/test was low, 
and dice_coef_hybrid was high,
 as shown below.
<br>
<pre>
categorical_focal_dice_loss,0.0995
dice_coef_hybrid,0.9632
</pre>
<br>
<h3>
5. Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/TB-Chest-X-Ray</b> folder
and run the following bat file to infer segmentation regions for images using the trained TensorFlowUNet model for Tuberculosis-Chest-X-Ray.<br>
<pre>
>./3.infer.bat
</pre>
This runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for TB-Chest-X-Ray Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks except for the first case.<br>
<br>
<b>class_color_map = {Lung: cyan, Tuberculosis: yellow}</b>
<br><br>
<table>
<tr>
<th width="320" height="auto">Image</th>
<th width="320" height="auto">Mask (ground_truth)</th>
<th width="320" height="auto">Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/images/TB.185.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/masks/TB.185.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test_output/TB.185.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/images/TB.660.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/masks/TB.660.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test_output/TB.660.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/images/TB.1352.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/masks/TB.1352.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test_output/TB.1352.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/images/TB.1447.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/masks/TB.1447.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test_output/TB.1447.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/images/TB.2176.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/masks/TB.2176.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test_output/TB.2176.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/images/TB.2494.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test/masks/TB.2494.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/TB-Chest-X-Ray/mini_test_output/TB.2494.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Efficient and Accurate Pneumonia Detection Using a Novel Multi-Scale Transformer Approach</b><br>
Alireza Saber, Pouria Parhami, Alimihammad Siahkarzadeh, Amirreza Fateh<br>
<a href ="https://arxiv.org/html/2408.04290v1">
https://arxiv.org/html/2408.04290v1
</a>
<br><br>
<b>2. Pneumonia detection from enhanced chest X-Ray images based on Double SGAN model</b><br>
Zhijing Xu & Haoyang Zhang<br>
<a href="https://www.nature.com/articles/s41598-026-39785-w" >
https://www.nature.com/articles/s41598-026-39785-w
</a>
<br><br>
<b>3. Leveraging human expert image annotations to improve pneumonia 
differentiation through human knowledge distillation</b><br>
Daniel Schaudt, Reinhold von Schwerin, Alexander Hafner, Pascal Riedel, <br>
Christian Späte, Manfred Reichert, Andreas Hinteregger, Meinrad Beer & Christopher Kloth<br>
<a href ="https://www.nature.com/articles/s41598-023-36148-7?fromPaywallRec=false">
https://www.nature.com/articles/s41598-023-36148-7?fromPaywallRec=false
</a>
<br><br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Tuberculosis-Chest-X-Ray</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Tuberculosis-Chest-X-Ray">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Tuberculosis-Chest-X-Ray</a>
<br><br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Chest-X-Ray</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Chest-X-Ray">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Chest-X-Ray</a>
<br><br> 
<b>6. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br><br>
