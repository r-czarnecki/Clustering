# Machine learning - Clustering

This project was created as part of the University of Warsaw's Machine learning course.

The goal of the project was to create a clustering algorithm for images of characters. The exact description of the task can be found in the [task.pdf](task.pdf) file.

## METHOD

=== Similarity function ===

The similarity function `compute_sim` determines the similarity between two images - the lower returned number, the more similar two images are. The steps it makes are as follow:
1. Since the images can be shifted or have some noise, I am cropping them by removing all topmost and bottommost rows of pixels and leftmost and rightmost columns of pixels whose color is closer to white than a certain value.
   Since I don't know which value will be the most appriopriate, I check all of them starting from the color closest to white on those images (any threshold whiter than that would crop everything off of one of the images) and going to the black color (0,0,0) making steps of 50 to reduce the algorithm's execution time
    a. Crop the images at the determined threshold
    b. Match images' sizes - both images are resized to the `(height, width)` dimensions where `height` is a maximum height of those images and `width` is a maximum width of those images
    c. Subtract values from both images and compute absolute value of the subtraction to get the difference between the images. That way I obtain the image where blacker pixels represent places of greater difference
    d. Blur the image obtained from the previous step using a Gaussian blur. Since the images of the same type of characters will usually not match perfectly in some places and at the same time images of different character types will most likely
       have bigger areas where they differ, that blur will make the importance of separate black pixels less significant, and the importance of bigger areas of black pixels relatively more significant
    e. Compute a `sim` value which is a sum of squares of the pixel values from the image obtained from the previous step
    d. Compute a `ratio_penalty_mult` value which is a ratio of the compared images' ratios of their width and height. The images that represent the same character will probably have similar length to width ratios, so I assume that the more different those
       ratios are, the more different the images are as well. Additionally, if the `ratio_penalty_mult` is higher than 1.5, I assume that the images are definitely representing a different image, so I add 1000 to that value
    e. Compute a `size_reward` which is a square root of the number of image's pixels. Bigger images will have more differences than smaller images due to their size, but it doesn't mean they are less similar to each other, so this value will help mitigate
       this effect
    d. Return (sim * ratio_penalty_mult * 100 / size_reward)**0.5 as the final value
2. After values for all thresholds are computed, I sort them, filter out the first and last 25% of them and return the mean of them as the final similarity value

=== Clustering algorithm ===

1. Take an image. If it's already assigned to some cluster, take the next image and repeat. Otherwise, go to step 2
2. Assign this image to a separate cluster
3. Compute similarities between this image and all the other unassigned images. If a similarity is no greater than 1700 (I found this value to work the best), assign them to the image's cluster from the previous step
4. Compute similarities between centers of the image's cluster and all the other clusters. Center is an image that is just an average of all the other images in the cluster. If the computed difference is lesser or equal to 1700, merge those clusters
5. Repeat step 1 until there are no more unassigned images

## EXPECTED DURATION OF EXECUTION

For 5000 images the algorithm takes around 9.5 minutes.

## HOW TO RUN

The algorithm has been tested on Python 3.8.5 and on Linux. To run it use:

`./run.sh <name_of_env> <file_with_images>`

`<name_of_env>` is the name of the python environment that should be created \
`<file_with_images>` is a path to the input file containing the paths of images that should be clustered

The results will be created in the clusters.html and clusters.txt files.