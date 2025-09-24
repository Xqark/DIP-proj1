### Goals

Practice the knowledge and skills you have learned so far from this course. The problems that you will solve—“Colour car tracking”—may involve image histograms, spatial domain filtering, and colour image processing.

### Description

Two image sequences, each containing 200+ frames, are provided. One sequence, named “red,” contains a red car, and the other sequence, called “blue,” contains a blue car. You are tasked to track the red and blue cars throughout the two image sequences, respectively.

For each image sequence, you are free to use the first frame for template selection, colour information analysis, etc. Then, you need to design an algorithm to automatically track the objects of interest through the remaining frames. In other words, no manual interference is allowed starting from frame #2.

Your algorithm will be evaluated based on two criteria:

1. How many frames you can go before losing the object.
2. How accurate your tracking results are. (For each frame of a sequence, the predicted object location should be indicated with a bounding box. Tighter bounding boxes are better.)

**Hint:** The normalized cross-correlation (NCC) is a popular option for template matching. You should first convert a colour image into a grayscale one. It is highly recommended that the size of the template should be adjustable to fit the ever-changing object size. Also, colour may play an important role in this task. However, note that the car may change its colour slightly from frame to frame.

### Yeses and Nos

**Yeses:**

* You are expected to use the skills and techniques you have learned so far or will be learning in the next two weeks.
* It is OK to use morphological image processing operations, although they may not be a game changer for this task.

**Nos:**

* You MUST NOT use deep-learning-based object detection and tracking approaches, as they would give you an unfair advantage over other students who do not have deep learning experience.
* Advanced computer vision methods, such as SIFT feature matching and similar technologies, are not permitted for the same reason.
