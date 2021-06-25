# CellMaskRCNN
PyTorch Mask-RCNN for Cell Segmentation

## Parameter list:<br/>

*imagePath* - **path to the image dataset. Required.<br/>**
*--model* - **currently only one model exists (zeisscyto).<br/>**
*--outputPath* - **path where output files should be saved to.<br/>**
*--channel* - **channel containing the cytoplasm stain. 0-based indexing.<br/>**
*--threshold* - **a value between 0 and 1 to filter out false detections. Default is 0.6.<br/>**
*--overlap* - **the image is split into overlapping tiles before cytoplasm detection. This parameter specifes the amount of overlap in pixels.<br/>**
*--scalingFactor* - **factor by which to increase/decrease image size by. Default is 1 (no resizing).<br/>**
*--GPU* - **if multiple GPUs are available, this specifies which GPU card to use. Default behavior is the first GPU card (0-based indexing).<br/>**
