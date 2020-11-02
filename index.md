## CS 194-26 Project 4: Facial Keypoint Detection with Neural Networks

### Part 1: Nose Tip Detection

In this part, all we had to do was design and train a simple convolutional neural network to detect the point directly under a person's nose.

First, I had to make sure I was loading the ground-truth values correctly. The results of this can be seen below, along with the results from showing just the nose key-point.

<img src="website_imgs/p1_ground_truth.png" alt="Labeled ground truths" width="800"/>
<img src="website_imgs/p1_nose.png" alt="Labeled noses" width="800"/>

Now that I knew I was loading the images and labels properly, I was able to start training. I chose to use a 3-layer network, with 12 7x7 filters for conv1, 16 5x5 filters for conv2, and 32 3x3 filters for conv3. I was first resizing the images to be 60 by 80 (as opposed to 480 by 640) for speed, so this results with the first fully connected layer having input size 32x4x7=896. I chose an output size of 256, leaving the last fc layer to have input 256 and output 2.

My training results are shown below. We can see that around 10 epochs, our model's loss gets worse temporarily, but then recovers to find a lower minima by 25 epochs.

<img src="website_imgs/p1_graph.png" alt="Training graph for part 1" width="500"/>

This simple model seems to work decently, successfully detecting the nose point even on images of faces that are not from straight on. However, it also mislabels a lot of noses. This is probably because of insufficient training time, overfitting to training data and therefore not generalizing well to validation data, or a not complex enough model. Since this model is simply a proof-of-concept for latter parts of the project, though, its performance is sufficient.

Some example results from performance on the validation set are shown below. Green is the ground-truth, and red is the network's output.

<table>
  <tr>
    <td> Works </td>
    <td> <img src="website_imgs/p1_work1.jpg" alt="Part 1 work 1" width="250"/> </td>
    <td> <img src="website_imgs/p1_work2.jpg" alt="Part 1 work 2" width="250"/> </td>
  </tr>
  <tr>
    <td> Do not work </td>
    <td> <img src="website_imgs/p1_broke1.jpg" alt="Part 1 broken 1" width="250"/> </td>
    <td> <img src="website_imgs/p1_broke2.jpg" alt="Part 1 broken 2" width="250"/> </td>
  </tr>
</table>

### Part 2: Full Facial Keypoints Detection

Now that we've shown we can use a convolutional neural network to detect the nose point, we can generalize this to all the facial keypoints by just increasing the output dimension of our final fully connected layer. In this case, we would have an output of size 58x2=116.

This time, we also added some data augmentation to add both robustness and help prevent overfitting to training data. I chose to implement random rotation and random shifting. 

To validate I was doing augmentation correctly, I first tested to see if the new ground-truth labels of augmented data were correct. This can be shown below.

<img src="website_imgs/p2_ground_truth.png" alt="Part 2 ground truth" width="800"/>

I made the model slightly more complicated this time, and increased the input image size. I settled on input size 240x180 and a model with 6 convolutional layers, all with 5x5 filters. The first two had 12 filters, the next two had 20, and the last two had 32. As with before, there were two fully connected layers at the end that ultimately give us the output of 116 numbers corresponding to the predicted keypoints.

The training looks a lot smoother than in part 1. We do see, however, that the model isn't really learning until after epoch 10, after which we see training and validation loss both going down.

<img src="website_imgs/p2_graph.png" alt="Part 2 graph" width="500"/>

<table>
  <tr>
    <td> Works </td>
    <td> <img src="website_imgs/p2_work1.jpg" alt="Part 2 work 1" width="250"/> </td>
    <td> <img src="website_imgs/p2_work2.jpg" alt="Part 2 work 2" width="250"/> </td>
  </tr>
  <tr>
    <td> Do not work </td>
    <td> <img src="website_imgs/p2_broke1.jpg" alt="Part 2 broken 1" width="250"/> </td>
    <td> <img src="website_imgs/p2_broke2.jpg" alt="Part 2 broken 2" width="250"/> </td>
  </tr>
</table>

As with before, we can see that detecting facial keypoints works better in some cases than in others. Some of the more interesting failure cases, which I've included above, seem to come from the model's outputs being affected by shadows, particularly from beneath the chin, pulling all the predicted keypoints downwards.

Something interesting we can do is to try and visualize the filters. 

<table>
  <tr>
    <td> Conv1 </td>
    <td> <img src="website_imgs/p2_conv1.png" alt="Part 2 conv1" width="450"/> </td>
  </tr>
  <tr>
    <td> Conv2 </td>
    <td> <img src="website_imgs/p2_conv2.png" alt="Part 2 conv2" width="450"/> </td>
  </tr>
</table>

Most of the filters actually don't look like much. But, conv1's 3.0 and 4.0 seem to detect something about horizontal edges, and 6.0 responds to vertical edges. For conv2, we see more complicated patterns that are much harder to make any sense of.

### Part 3: Train with Larger Dataset

For this part, we had to train a model using data from the ibug face in the wild dataset, which contains 6666 images, each with 68 facial keypoints. 

We had to do an extra step of cropping to the actual face in this dataset, and also had to deal with the fact that labelling was done differently. Here are my results for verifying that I was intaking all the labels and images correctly. We can see that the crop is pretty good, and all the points line up even after random rotation and shifting.

<img src="website_imgs/p3_ground_truth.png" alt="Part 3 ground truth" width="800"/>

For this part, we were allowed, and advised, to use existing standard computer vision models. I chose to go with ResNet18 for its combination of being relatively light and powerful. Its design is shown in the diagram below.

<img src="website_imgs/resnet18.png" alt="Resnet model layers" width="800"/>

However, I had to make a few modifications. Since my input image was grayscale, I had to change the first convolutional layer to only take 1 input channel. Also, instead of outputting 1000 classes, I needed the output of the fully connected layer to have size 68x2=136, so I changed that.

Also, I chose not to use the pretrained weights, so I trained the model from scratch. I did it with 10 epochs with lr=0.00015, then trained for another 10 epochs with lr=0.00005. After the first 10 epochs, I had a score of 8.88 on Kaggle. With the extra 10 epochs, I was able to drop my score to 7.54.

<table>
  <tr>
    <td> <img src="website_imgs/p3_graph.png" alt="First 10 epochs graph" width="250"/> </td>
    <td> <img src="website_imgs/p3_graph2.png" alt="Second 10 epochs graph" width="250"/> </td>
  </tr>
</table>

We can see that there is steady improvement in the training loss, and some lesser improvement in validation loss. We are probably going closer to overfitting, based on this. To do better, I would probably have to do more data augmentation. I think I could do better by just using the pretrained weights and adding more fully connected layers to the end, and training just the fully connected layers plus a little bit of fine-tuning on the latter ResNet18 layers. Using a larger network would probably also improve results. However, as it stands, the results are still pretty good visually.

We can verify that our results are pretty good by visualizing some images with the keypoints predicted from the test set. It's pretty cool that the model can predict the outline of the faces, even though the faces that are input are not fully in frame.

<img src="website_imgs/p3_pred.png" alt="Predictions on the test set" width="800"/>

We can even test the model on our own images!

<img src="website_imgs/my_collection.png" alt="Predictions on my images" width="800"/>

The model seems to work on all of the photos I provided. In fact, it even worked on Shrek, who certainly has human-like features, but is definitely not a human!
