# AR (BookTrailer)
It is the project of Computer Vision Master Course which projects the trailer of a novel on its cover using feature matching.

## Description
I have a dataset of novel covers and their movie trailer. The input of program is a video, and we should find any book cover that we have and then projects its trailer on it. Additionally, we must consider occlusions and affect projected videos.

It worth mentioning that it is an offline program, and it produce output video after a few minutes.


I used SIFT features in order to detect book covers, also I used RANSAC to attain corresponding projection matrix.

![Workflow](GithubImages/Workflow.png)
