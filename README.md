# Unwrap Labels
## 

Algorithm to unwrap labels using edge markers

## How it works

You need to specify just 6 markers to unapply distortion. Let's take the example below:
![alt text](https://raw.githubusercontent.com/Nepherhotep/unwrap_labels/master/samples/sample1/original.jpg)

Markers depicted on the image below. As one of the markers is out of the image, 
let's just increase canvas a bit.

![alt text](https://raw.githubusercontent.com/Nepherhotep/unwrap_labels/master/samples/sample1/corner-points.jpg)

## Makers detection

Makers detection is not a part of this library, however I'll provide some clues
how to do that:

1. Hough transform to detect vertical lines edges of the bottle (lines A-F, C-D)
2. Due to bottle's symmetry - another Hough transform to detect upper and bottom ellipses.
A-B-C ellipse reduced to just A-B coordinate, as well C-D-F - to just D-F.
