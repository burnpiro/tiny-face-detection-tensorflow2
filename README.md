# Tiny Face Detection with TensorFlow 2.0

### Dataset

We want to use WIDER FACE dataset. It contain over 32k images with almost 400k faces and is publicly available on
[http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/)

- [Training Data GDrive](https://drive.google.com/file/d/0B6eKvaijfFUDQUUwd21EckhUbWs/view?usp=sharing)
- [Val Data GDrive](https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view?usp=sharing)
- [Test Data GDrive](https://drive.google.com/file/d/0B6eKvaijfFUDbW4tdGpaYjgzZkU/view?usp=sharing)
- [Annotations](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip)

Please put all the data into `./data` folder.

Data structure is described in `./data/wider_face_split/readme.txt`. We only need to use boxes annotations but there is more data available if someone wants to use it.