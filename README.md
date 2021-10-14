# Computer Vision
The goal of these projects was to make us familiar with computer vision concepts and techniques. <br/>
For more information about each project, please check the report included in each project. <br/>
#### Report included in every project :) ####

### Project 1: ###
 * This project focuses on analysing the image and using the data from the cells that have bee detected from a microscope.
    * Task 1 - Cell Counting
        <details>
          <summary> Steps:</summary>
          <summary> 1.Used a median filter to remove the "noise" from the image </summary>
          <summary> 2. Removed salt noise using opening technique (erosion + dialation).</summary>
          <summary> 3. Removed pepper noise using the closing technique(dialetion + erosion). </summary>
          <summary> 4. Used thresholding to transform the image to grayscale. </summary>
          <summary> 5. Contour the cells so that the joined cells are now seperated.</summary>
          <summary> 6. Made all the cells have the same hierarchy, used a mask to depict each cell on it, dialated them to come to their normal size and depicted them in another mask all together (process explained in more detail in the report).</summary>
    * Task 2 - Surface Measuring
      <details>
        <summary> Steps:</summary>
          <summary> 1.Kept count of the cells accepted and declined </summary>
          <summary> 2. Using an empty array for counting looped through all the cells.</summary>
          <summary> 3. If the cell is not in the array with the declined ones use count area to count the pixels the dialated cell is using. </summary>
          <summary> 4. Depict each cell counting process step by step visually. </summary>
    * Task 3 - Mean gradient of gray scale
        <details>
          <summary> Steps:</summary>
          <summary> 1.Created bounding boxes around each cell</summary>
          <summary> 2.Created the sum table where each value is the sum from all the above (process explained in the report visually and theoritically)</summary>
          <summary> 3. Called the cv integral function to help with the scale variance.</summary>
          <summary> 4. Used the following function to calculate the gray scale of each cell:</br>
          gray_sum = integral_image[(y + h), (x + w)] - integral_image[(y + h), x] - integral_image[y, (x + w)] + integral_image[(y - 1), (x - 1)]</summary>
          <summary> 4. Divided the gray_sum with the number of cells in each bounding box to get the mean value . </summary>

### Project 2: ###
* This project focuses on manually creating a panorama from more than 4 different photos.
   * Task 1 - Created a panorama using sift implementation
      
   * Task 2 - Created a panorama using surf implementation </br>
#### For more information about the process, check the report included in project 2 ####


### Project 3: ###
 * This project focuses on multiclass clasification using unsupervised learning.
    * Task 1,2 - Visual Vocabulary & Descriptor Extraction based on the Bag Of Words madel. 
        <details>
          <summary> Steps:</summary>
          <summary> 1.Extracted the charasteristics of every image in the dataset. </summary>
           <summary> 2.Word creation using k-means. </summary>
          <summary> 3.Mached every keypoint with one word</summary>
          <summary> 4. Created a histogram for each image based on the frequency of the appearence of the words(created above) in the image. </summary>
    * Task 3 - Image Classification using the vocabulary created above.
         * Using k-NN without the use of the function cv.ml.KNearest_create() .
         * Using the one-versus-all where each class is trained with an SVM classifier.
    * Task 4 - Evaluated the system
        <details>
          <summary> Evaluated the following:</summary>
          <summary> 1.Using the imagedb_test measured the accuracy of the system(in both classifier cases).</summary>
          <summary> 2.Checked how the number of words(from BOW) affects the result.</summary>
           <summary> 3. Checked how the number of neighbours(k-means) affects the result.</summary>
          <summary> 4. Checked how the kernel size (SVM) affects the result. </summary>

#### For more info, check the report in the project3 folder. ####

### Project 4: ###
Implemented 2 Convolutional Neural Networks
   * A non pretrained one using regularization,augmentation,bach normalization and max-pooling 
   * A pretrained network tuning the parameters to fit our datasets needs.</br>
   Used VGG19,VGG16 and InceptionV3 . </br>
#### For more info, check the report in the project4 folder. ####
