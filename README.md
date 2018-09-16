# Unwrap Labels
## 

Algorithm to unwrap labels using edge markers

## How it works

You need to specify just 6 markers to unapply distortion. Let's take the example below:
![original image](https://raw.githubusercontent.com/Nepherhotep/unwrap_labels/master/samples/sample1/original.jpg)

Markers depicted on the image below. As one of the markers is out of the image, 
let's just increase canvas a bit.

![markers image](https://raw.githubusercontent.com/Nepherhotep/unwrap_labels/master/samples/sample1/corner-points.jpg)

The library creates a mesh, which will be transformed into a plane:
![mesh](https://raw.githubusercontent.com/Nepherhotep/unwrap_labels/master/samples/sample1/original-with-mesh.jpg)

Unwrapped image:
![unwrapped](https://raw.githubusercontent.com/Nepherhotep/unwrap_labels/master/samples/sample1/unwrapped.jpg)

## Makers detection

Makers detection is not a part of this library, however I'll provide some clues
how to do that:

1. Hough transform to detect vertical lines edges of the bottle (lines A-F, C-D)
2. Due to bottle's symmetry - another Hough transform to detect upper and bottom ellipses.
A-B-C ellipse reduced to just A-B coordinate, as well C-D-F - to just D-F.
