# CNN-ensemble-classifier-Land-Use-Classification
Convolutional Neural Network combined with ensemble classifier for land use classification

In this project, the architecture of the cnn is same as cifar-10 architecture, thanks to https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

also using keras library, with Theano as backend

The Land Use dataset is from UC Merced Land Use Dataset, thanks to http://vision.ucmerced.edu/datasets/landuse.html Total data used is 2100 images, which consist of 21 classes, each class consist of 100 images, the classes are:

1. agricultural
2. airplane
3. baseballdiamond
4. beach
5. buildings
6. chaparral
7. denseresidential
8. forest
9. freeway
10. golfcourse
11. harbor
12. intersection
13. mediumresidential
14. mobilehomepark
15. overpass
16. parkinglot
17. river
18. runway
19. sparseresidential
20. storagetanks
21. tenniscourt

Each are labelled with number from 0-20 manually, labels are stored in a csv file that will be read during the training and testing.
