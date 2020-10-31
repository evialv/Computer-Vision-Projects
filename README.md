# Information_Retrival
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
          <summary> 1.Create bounding boxes around each cell</summary>
          <summary> 2.Create the sum table where each value is the sum from all the above (process explained in the report visually and theoritically)</summary>
          <summary> 3. Call the cv integral function to help us with the scale variance.</summary>
          <summary> 4. Use the following function to calculate the gray scale of each cell:</br>
          gray_sum = integral_image[(y + h), (x + w)] - integral_image[(y + h), x] - integral_image[y, (x + w)] + integral_image[(y - 1), (x - 1)]</summary>
          <summary> 4. Divide the gray_sum with the number of cells in each bounding box to get the mean value . </summary>
