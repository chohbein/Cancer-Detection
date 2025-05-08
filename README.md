# Cancer-Detection
CNN Cancer Imaging Detection

Data found here: https://www.kaggle.com/competitions/histopathologic-cancer-detection/data <br>
To replicate, download zip files and follow instructions within 'CNN Cancer Project' notebook


## Approach
For this project, I build a CNN model to predict cancer in histopathologic images and tested multiple hypotheses to determine the optimal architecture. I elected to build the model architecture from scratch without the use of pre-trained weights, (e.g. ImageNet), to not only challenge myself but gain a deeper understanding of optimization and limitations of computational constraints. Models were trained using PyTorch and optimized with GPU-acceleration and more specific approaches such as mixed precision (GradScaler).


## Results
![image.png](attachment:image.png)
Due to the nature of the problem, I also wanted to minimize false-negatives; which I got down to 2%. <br>

## Compute Problems 
This model was trained solely on my laptop. With hundres-of-thousands of images to train on, this was an initial roadblock for my little RTX-3060 Laptop GPU. To address this, I took multiple steps to improve computation costs while also balancing performance. <br>
- CUDA for GPU utilization.
- Mixed precision training (autocast,GradScaler) to tradeoff lower precision while maintaining accuracy.
- Optimized the data-loading process by using multiple persistent workers and downsizing image resolution.
- Tuned batch sizes for training to maximize GPU usage.
- Frequent normalization

## Approach
The model was built from scratch with no use of pre-trained weights. <br>
My modeling approach tested 2 hypotheses which I formulated from data inspection:
##### Iteration 1
Given the absence of obvious visual cues during exploratory data analysis (EDA), we hypothesize that cancerous features exist in fine-grained patterns within the images. Therefore, a model capable of capturing subtle, localized features (e.g., one with small receptive fields) is necessary to detect these indicators. <br>
This entailed:
1. Resized images to 46x46 (96x96 originally)
    - Balancing performance with computational efficiency
2. 5x Convolution layers
    - Small kernel sizes
        - We used 3×3 kernels along with 1×1 kernels specifically to first capture spatial relationships, then more refined features, without losing spatial resolution.
    - Delayed pooling
        - Our delayed pooling approach (only after the final convolutional layer) preserves detail throughout the network.
    - Padding = 1 on 3x3 layers to avoid shrinkage
    - Batch Normalization after each Conv. layer
        - Since this iteration was highly focused on details, it is important to account for overfitting on noise.
    - Dropout (0.3) to avoid overfitting
<br><br>
##### Iteration 2
Compare our first iteration to one that focuses on broader trends and patterns in the images.
This entailed:
1. Resized images to 46x46 (96x96 originally)
    - Balancing performance with computational efficiency
2. 5x Convolution layers
    - Small kernel sizes
        - We used 3×3 kernels along with 1×1 kernels specifically to first capture spatial relationships, then more refined features, without losing spatial resolution.
    - Delayed pooling
        - Our delayed pooling approach (only after the final convolutional layer) preserves detail throughout the network.
    - Padding = 1 on 3x3 layers to avoid shrinkage
    - Batch Normalization after each Conv. layer
        - Since this iteration was highly focused on details, it is important to account for overfitting on noise.
    - Dropout (0.3) to avoid overfitting
  
##### Tuning Approaches
- Early Stopping
    - Both models' validation losses seem to diverge from the training loss, suggesting overfitting.
    - Early stopping showed no improvements on either iteration.
    - Both low and high patience did not see improvement. While it slightly lowered false negatives, the decrease in AUC was not sufficiently accounted for.

- LR Scheduling
    - Attempted this because the 2nd iteration seems to have plateaud on it's validation loss.
    - Additionally, it may help with the overfitting problem.
    - Did not improve the model.
    - Reasons for this may include high amounts of noise in the images, or the limited epochs we are running to account for lack of computational efficiency on my local machine.

- Image Alterations
    - Shuffling image presentation is known to help CNN's pick up on important patterns.
    - This reflected in our data; the following alterations improved performance:
        - Randomized rotation, randomized horizontal/vertical flipping, color jitter

