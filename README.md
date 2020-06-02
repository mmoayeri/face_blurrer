# Media Scrubber
This code allows for images or videos to be altered to protect the identities of individuals in the pictures. Using opencv face and eye detectors, faces can be blurred out or eyes can be blacked out. 
Blocked Eyes               | Blurred Face
:-------------------------:|:-------------------------:
<img src="/output/block_eyes_ferguson4.jpg" width="75%"> | <img src="/output/blur_faces_ferguson3.jpg" width="75%">

## Requirements
* Python 3.x
* Opencv

## Usage
To alter an image, use the following:
``` python media_scrubber.py --input '/path/to/image.jpg' --mode 'block_eyes' ```
The default mode is to blur faces. Videos can also be altered frame by frame, though the current setting only retains one fifth of the frames. This can be altered with the argument ```--frame_ratio```. Outputs are stored in the output directory with the same name as the input, along with a prefix indicating the kind of alteration. 

## Notes

I whipped this up in a few hours so it's not too great. This heavily relies on the Haar cascade detectors provided in opencv. This is a mainly a proof of concept exercise. The motivation was to demonstrate the capability of technology to allow for visibility of resistance while still protecting the identites of individuals, especially following the tragic fate of some [Ferguson protesters](https://www.chicagotribune.com/nation-world/ct-ferguson-activist-deaths-black-lives-matter-20190317-story.html).

**Black Lives Matter** 
