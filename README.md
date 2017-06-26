# Building a Face Recognition System with OpenCV in the blink of an Eye... Checking your mood with the Microsoft Face API

<table style="width:100%">
  <tr>
    <th><img src="https://github.com/AIAScience/face-recognition-demo-based-on-opencv-and-microsoft-face-api/blob/master/images/Michael.gif?raw=true" alt="Michael"></th>
    <th><img src="https://github.com/AIAScience/face-recognition-demo-based-on-opencv-and-microsoft-face-api/blob/master/images/Dirk.gif?raw=true" alt="Dirk"></th>
    <th><img src="https://github.com/AIAScience/face-recognition-demo-based-on-opencv-and-microsoft-face-api/blob/master/images/Atle.gif?raw=true" alt="Atle"></th>
    <th><img src="https://github.com/AIAScience/face-recognition-demo-based-on-opencv-and-microsoft-face-api/blob/master/images/Patrick.gif?raw=true" alt="Patrick"></th>
    <th><img src="https://github.com/AIAScience/face-recognition-demo-based-on-opencv-and-microsoft-face-api/blob/master/images/OJ.gif?raw=true" alt="Odd-Jostein"></th>  
  </tr>
</table>


* Are you happy?    (◕‿◕✿)
* Are you sad?      (ಥ﹏ಥ)
* Are you afraid?    ⚆ _ ⚆
* Are you surprised? ◉_◉
* Are you angry?     (ง'̀-'́)ง
* ...


This repository is an extension to [the great presentation made by Rodrigo Agundez at PyData 2016](https://github.com/rragundez/PyData). **Thanks Rodriguo!!!**

It extends the face recognition example with simple calls to the Microsft Face API whih returns an estimation of the age, of the gender and of the mood during the collection of faces. The advantage of showing your mood while collecting 20 images of someone's face is to encourage the participants to fake their mood, and as a result, provides a better diversity of the faces collected as training set. 


## Lets try it!!

* you will need to update this file with a valid key from Microsoft. [Just get one here!](https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/vision-api-how-to-topics/howtosubscribe)

```shell
$ cat donthackme.py
API_KEY = '36fb2145d5ad432586376876703b9aa3'
$ 
```

* then simply run: `python start_face_recon_app.py`
