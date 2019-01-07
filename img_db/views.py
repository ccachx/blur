import os
import cv2
from django.shortcuts import render
from img_db.models import IMG
from img_db.function import blur, cus_filter2D, sketch,relief


# Create your views here.


def index(request):
    if request.method == 'POST':
        new_img = IMG(img=request.FILES.get('upimg'))
        new_img.save()

        m11 = request.POST['m11']
        m12 = request.POST['m12']
        m13 = request.POST['m13']
        m21 = request.POST['m21']
        m22 = request.POST['m22']
        m23 = request.POST['m23']
        m31 = request.POST['m31']
        m32 = request.POST['m32']
        m33 = request.POST['m33']

        img_name = str(new_img.img.name).split('/')[1].split('.')[0]

        if not os.path.isdir("media/" + img_name):
            os.mkdir("media/" + img_name)

        rgbimg = cv2.imread(new_img.img.path)
        cvimg = cv2.imread(new_img.img.path, 0)

        img_height= rgbimg.shape[0]
        img_width = rgbimg.shape[1]

        if img_width>400:
            img_height=img_height*400/img_width
            img_width=400
            rgbimg=cv2.resize(rgbimg,(int(img_width),int(img_height)))
            cvimg = cv2.resize(cvimg, (int(img_width), int(img_height)))

        img_blur, img_B, img_C = blur(cvimg, 100, 1 / 4, 1 / 2, 1 / 4)
        img_deblur = cus_filter2D(cvimg)  # {{-1,-1,0},{-1,0,1},{0,1,1}}
        # img_fudiao = cus_filter2D(cvimg, -1, -1, 0, -1, 0, 1, 0, 1, 1)
        img_fudiao = relief(rgbimg)
        img_cus = cus_filter2D(cvimg, m11, m12, m13, m21, m22, m23, m31, m32, m33)

        cv2.imwrite("media/" + img_name + "/source.jpg", cvimg)
        cv2.imwrite("media/" + img_name + "/blur.jpg", img_blur)
        cv2.imwrite("media/" + img_name + "/blur_B.jpg", img_B)
        cv2.imwrite("media/" + img_name + "/blur_C.jpg", img_C)
        cv2.imwrite("media/" + img_name + "/img_deblur.jpg", img_deblur)
        cv2.imwrite("media/" + img_name + "/img_cus.jpg", img_cus)
        cv2.imwrite("media/" + img_name + "/img_fudiao.jpg", img_fudiao)
        sketch(img_name)

        content = {
            'aaa': new_img,
            'sourse': "/media/" + img_name + "/source.jpg",
            'blur': "/media/" + img_name + "/blur.jpg",
            'blur_B': "/media/" + img_name + "/blur_B.jpg",
            'blur_C': "/media/" + img_name + "/blur_C.jpg",
            'img_deblur': "/media/" + img_name + "/img_deblur.jpg",
            'img_cus': "/media/" + img_name + "/img_cus.jpg",
            'img_fudiao': "/media/" + img_name + "/img_fudiao.jpg",
            'img_sketch': "/media/" + img_name + "/img_sketch.jpg",

        }
        return render(request, 'index.html',content)
    return render(request, 'index.html')


