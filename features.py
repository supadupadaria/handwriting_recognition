import cv2 as cv
import math
import matplotlib.pyplot as plt
import numpy as np

color = 255
color2 = 100

# алгоритм брезенхема
def bresenham_line(x0, y0, x1, y1, img, color):
    img[y1, x1] = color
    img[y0, x0] = color
    trflag = False

    if (x0 == x1 and y0 == y1):
        return img

    if (np.abs(y1 - y0) > np.abs(x1 - x0)):
        trflag = True
        x1, y1 = y1, x1
        x0, y0 = y0, x0

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    xar = range(x0, x1 + 1, 1)

    if y0 > y1:
        xar = xar.__reversed__()
        y = y1
    else:
        y = y0

    d = np.abs(y1 - y0)
    dx = np.abs(x1 - x0)
    er = 0

    for x in xar:
        if trflag:
            img[x][y] = color
        else:
            img[y][x] = color

        er = er + d

        if (2 * er) >= dx:
            y = y + np.uint8(er / dx)
            er = er % dx

    return img

# функция отображения изображения в окне
def show_img(img):

    #plt.gray()
    plt.figure()
    plt.imshow(img)
    plt.show()


#длина отрезка:
def segm_len(x0, y0, x1, y1):
    return math.sqrt((x0-x1)**2 + (y0-y1)**2)

#получение конкретного контура
def get_cont(cont, hier, val):

    cont2 = []

    for n in range(len(cont)):
        if hier[0][n][3] == val:
            for c in cont[n]:
                cont2.append(c[0].tolist())

    return cont2

def get_1cont(cont, n):

    cont2 = []

    for c in cont[n]:
        cont2.append(c[0].tolist())

    return cont2

#получение списка вложенных контуров
def get_loops(cont, hier):

    cont_loops = []

    for n in range(len(cont)):

        cont2 = []

        if hier[0][n][3] == -1:

            for c in cont[n]:
                cont2.append(c[0].tolist())

            if len(cont2) > 7 and len(cont2) < 110 :
                #print("LEN:", len(cont2))
                cont_loops.append(cont2)

    return cont_loops

#длина петли
def get_len(cont):

    p1 = cont[0]
    cont_len = 0

    for p in cont[1:]:
        cont_len += segm_len(p1[0], p1[1], p[0], p[1])
        p1 = p

    return cont_len

#площадь петли
def get_area(cont):
    cont.append(cont[0])
    area = 0
    p1 = cont[0]
    for p in cont:
        # S += (X2 - X1) * (Y1 + Y2) / 2
        area += (p[0]-p1[0])*(p[1]+p1[1])
        p1 = p
    return area/2

#ширина петли
def get_width(cont):
    x = np.array([cont[i][0] for i in range(len(cont))])
    return x.max()-x.min()

#высота петли
def get_height(cont):
    y = np.array([cont[i][1] for i in range(len(cont))])
    return y.max()-y.min()

#самая высокая и самая низкая точка
def get_hl_points(cont):

    cont1 = np.array(cont)
    #cont_sorted = np.sort(cont1, axis=0)
    #cont_sorted = cont1.sort(key=lambda x: x[1])
    cont_sorted = sorted(cont1, key=lambda x: x[1])
    l_point = cont_sorted[0]
    h_point = cont_sorted[-1]

    return l_point, h_point

#угол наклона петли
def get_direction(point, intersec):

    #∆ y of the loop / ∆ x of the loop
    if point[0] - intersec[0] != 0:
        arg = math.fabs(point[1] - intersec[1]) / math.fabs(point[0] - intersec[0])
        return math.atan(arg)
    else: return 0


def asc_or_desc(intersec, cont):

    y = np.array([cont[i][1] for i in range(len(cont))])
    if np.abs(y.max() - intersec[1]) < np.abs(y.min() - intersec[1]):
        return True
    else: return False #descending


#угол наклона петли с учетом того, нисходящая она или восходящая
def get_loopdir(cont, intersec, asc=True):

    lp,hp = get_hl_points(cont)
    #print("highest and lowest: ", hp, lp)

    if asc == True:
        p = lp
    else:
        p = hp
    #print("P: ", p)
    return get_direction(p, intersec)


#угол между двумя векторами (с ипользованием скалярного произведения)
def angle_scmult(a, b):
    return math.acos((a[0]*b[0]+a[1]*b[1])/(segm_len(a[0],a[1],0,0)*segm_len(b[0],b[1],0,0)))


#получить список векторов
def get_vectors(cont):

    vectors=[]

    c1 = cont[0]
    for c in cont[1:]:
        vectors.append([c[0]-c1[0], c[1]-c1[1]])
        c1 = c
    return vectors


#получить списком значение углов между векторами
def get_angles(cont):

    vectors = get_vectors(cont)
    vectors.append(vectors[0])
    angles = []

    v1 = vectors[0]

    for v in vectors[1:]:
        angles.append(angle_scmult(v1, v))
        v1 = v

    return angles


#получить точку пересечения
def intersection_point(cont):

    vectors = get_vectors(cont)
    vectors.append(vectors[0])
    angles = []

    v1 = vectors[0]

    for v in vectors[1:]:
        angles.append(angle_scmult(v1, v))
        v1 = v

    av = np.average(angles)
    subs_mod = [np.abs(angle-av) for angle in angles]


    index_intersec = np.argmax(subs_mod) + 1
    #print(index_intersec)
   # print("vecs: ", )

    return cont[index_intersec]


#"кривизна" петли
def get_curvature(cont):

    vectors = get_vectors(cont)
    vectors.append(vectors[0])
    angle = 0
    v1 = vectors[0]

    for v in vectors[1:]:
        angle += angle_scmult(v1, v)
        v1 = v

    return angle/(len(vectors)-1)


def get_averdirection(cont, intersec, asc):

    sumdir = 0
    std_dev = 0
    loopdir = get_loopdir(cont, intersec, asc)

    for p in cont:
        if np.abs(p[0] - intersec[0]) > 1 and np.abs(p[1] - intersec[1]) > 1:
            d = get_direction(p, intersec)
            sumdir += d
            std_dev += (d-loopdir)**2

    return sumdir/len(cont), math.sqrt(std_dev/len(cont))


#отрисовка контуров
def show_bpoints(cont, img):

    img_2 = np.zeros((len(img), len(img[0])))

    for n in range(len(cont)):

        ar = cont[n]
        k1 = ar[0]
        for k in ar[1:]:
            #print(k1, '   ', k)
            img_2 = bresenham_line(k1[0][0], k1[0][1], k[0][0], k[0][1], img_2, color)
            k1 = k

        img_2 = bresenham_line(k1[0][0], k1[0][1], ar[0][0][0], ar[0][0][1], img_2, color2)

    return img_2

#отрисовка контуров
def show_loops(cont, img):

    img_2 = np.zeros((len(img), len(img[0])))

    for ar in cont:

        k1 = ar[0]
        for k in ar[1:]:
            img_2 = bresenham_line(k1[0], k1[1], k[0], k[1], img_2, color2)
            k1 = k

        img_2 = bresenham_line(k1[0], k1[1], ar[0][0], ar[0][1], img_2, color2)

    return img_2

#отрисовка одного контура
def show_contour(cont, img):

    #img_2 = np.zeros((len(img), len(img[0])))

    k1 = cont[0]
    for k in cont[1:]:
         #img[k[1]][k[0]] = 255
         #if hier[0][n][3] == -1:
         img = bresenham_line(k1[0], k1[1], k[0], k[1], img, color)
         k1 = k
    img = bresenham_line(k1[0], k1[1], cont[0][0], cont[0][1], img, color)

    return img


#можно получить все значения параметров петли
def loop_features(loop):


    intersec = intersection_point(loop)
    asc_bool = asc_or_desc(intersec, loop)
    len = get_len(loop)
    area = get_area(loop)
    h = get_height(loop)
    w = get_width(loop)
    dir = get_loopdir(loop, intersec, asc=asc_bool)
    avdir, stdev = get_averdirection(loop, intersec, asc=asc_bool)
    curv = get_curvature(loop)

    #print("intersection point: ", intersec)
    #print("is ascending: ", asc_bool)

    #print("vectors: ", get_vectors(loop))
    #print("angles: ", get_angles(loop))

    features = {'length': len, 'area': area,
                'height': h, 'width': w,
                'direction': dir, 'average_direction': avdir,
                'standart_deviation': stdev, 'curvature': curv}
    #добавить
    return features


def loop_features_normalized(loop, median_h):

    intersec = intersection_point(loop)
    asc_bool = asc_or_desc(intersec, loop)
    len = get_len(loop)
    area = get_area(loop)
    h = get_height(loop)
    w = get_width(loop)
    dir = get_loopdir(loop, intersec, asc=asc_bool)
    avdir, stdev = get_averdirection(loop, intersec, asc=asc_bool)
    curv = get_curvature(loop)

    #print("intersection point: ", intersec)
    #print("is ascending: ", asc_bool)

    #print("vectors: ", get_vectors(loop))
    #print("angles: ", get_angles(loop))

    features = {'length': len/(median_h*math.pi), 'area': area/(median_h**2),
                'height': h/median_h, 'width': w/median_h,
                'direction': dir/math.pi, 'average_direction': avdir/math.pi,
                'standart_deviation': stdev/math.pi, 'curvature': curv}
    #добавить
    return features

def loop_features_normalized2(features, median_h):

    features = {'length': features['length']/(median_h*math.pi), 'area': features['area']/(median_h**2),
                'height': features['height']/median_h, 'width': features['width']/median_h,
                'direction': features['direction']/math.pi, 'average_direction': features['average_direction']/math.pi,
                'standart_deviation': features['standart_deviation']/math.pi, 'curvature': features['curvature']}
    #добавить
    return features


def print_features(features):
    print("total length: ", features['length'])
    print("area: ", features['area'])
    print("loop width: ", features['width'])
    print("loop height: ", features['height'])

    print("direction: ", features['direction'])
    print("Average direction and standard deviation: ", features['average_direction'], features['standart_deviation'])
    print("curvature: ", features['curvature'])


#выделение конкретной точки на изображении
def show_point(img, point):

    img[point[1]][point[0]] = color2
    img[point[1]-1][point[0]] = color2
    img[point[1]][point[0]-1] = color2
    img[point[1]-1][point[0]-1] = color2

    return img


#returns image by its name
def get_line_image(line_id):

    filename = get_line_image_path(line_id)
    #print(filename)
    img = cv.imread(filename)

    return img


def get_line_image_path(line_id):
    parse = line_id.split("-")
    filename = 'data/' + parse[0] + "/" + parse[0] + '-' + parse[1] + "/" + line_id + ".png"  # путь к файлу с картинкой
    return filename


def image_processing(img):

    for i in range(len(img[0])):
        for j in range(len(img)):
            if img[j][i] > 225: img[j][i] = 255

    return img

def image_reshaping(img_name, new_w, new_h):

    img_name = get_line_image_path(img_name)
    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    newimg = cv.resize(img, (new_w, new_h))
    newimg = cv.bitwise_not(newimg)
    #show_img(newimg)
    #print(newimg/255)
    newimg = newimg/255
    #print(newimg.ravel())
    return np.array(newimg.ravel())


#средняя высота строки
def median_height(line_img):

    #2d
    line_img = cv.cvtColor(line_img, cv.COLOR_BGR2GRAY)

    #shape of image
    h, w = len(line_img), len(line_img[0])
    #print(h,w)

    #list of letters' heights
    heights = []

    #проход по каждому пикселю
    for y in range(w):
        max_x = 0
        min_x = 0

        for x in range(h):
            if line_img[x][y]!=255:
                max_x = x
                break

        for x in reversed(range(h)):
            if line_img[x][y]!=255:
                min_x = x
                break

        #print(max_x, min_x)
        if max_x!=0 and min_x!=0: heights.append(np.abs(max_x-min_x))

    #print(heights)
    print(np.median(heights), np.mean(heights))
    #show_img(line_img)

    return np.median(heights), np.mean(heights)

#accelerate
def line_processing(line_id):


    img = get_line_image(line_id)
    med_h, _ = median_height(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_proc = image_processing(img)
    #ret, thresh = cv.threshold(img_proc, 127, 255, 0) #cv.THRESH_BINARY
    thresh = cv.adaptiveThreshold(img_proc, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    img_thresh, cont, hier = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    #print(cont)
    #show_img(img_proc)
    #show_img(img_thresh)
    img2 = img_proc.copy()

    #cv.drawContours(img2, cont, -1, (0, 255, 0), 3, cv.LINE_AA, hier, 1)
    # cv.imshow('cont', img2)
    # cv.waitKey()

    loops = get_loops(cont, hier)
    print('loops number', len(loops))
    #img2 = show_loops(loops, img2)
    #show_img(img2)

    all_features = []
    all_normalized_features = []
    for loop in loops:
        try:
            feat = loop_features(loop)
        except: print("Computing error\n")
        else:
            all_features.append(feat)
            all_normalized_features.append(loop_features_normalized2(feat, med_h))

    return all_features, all_normalized_features


lines = ['n04-022-06', 'l04-087-01']#, 'p03-029-01', 'r06-143-04', 'f02-044-01', 'g04-017-04', 'g06-042h-02']#, 'g06-037k-09', 'k04-106-04', 'm01-026-04', 'n04-022-06', 'l04-087-01', 'p03-029-01', 'r06-143-04'  ] #

#for l in lines:
#    image_reshaping(l, 800, 50)

'''for line in lines:
    f_all = line_processing(line)
    print(line, '\n')

    for f in f_all:
        print_features(f)'''

'''
filename = 'img/g1.png' # путь к файлу с картинкой

img = cv.imread(filename)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img.shape)
ret, thresh = cv.threshold(img, 127, 255, 0)
img_tresh, cont, hier = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

img_line = get_line_image(line_id='a02-000-06')#c01-014-09')#a02-000-06')p03-040-02
median_height(img_line)

img_line = get_line_image(line_id='p03-040-02')#c01-014-09')#a02-000-06')
median_height(img_line)

#print("contours:\n", cont)
#print("\nhier:\n", hier)

#все петли
loops = get_loops(cont, hier)

img_2 = np.zeros((len(img), len(img[0])))

for loop in loops:
print(" : ", loop)
show_contour(loop, img_2)
print(loop_features(loop, 130))
i_pnt = intersection_point(loop)
img_2 = show_point(img_2, i_pnt)
show_img(img_2)'''


