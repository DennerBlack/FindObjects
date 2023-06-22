import math


def custom_sort(countour):
    return -countour.shape[0]

def get_center(countour):
    # вычислим центр тяжести контура
    sum_x = 0.0
    sum_y = 0.0
    for point in countour:
        x = float(point[0][0])
        y = float(point[0][1])
        sum_x += x
        sum_y += y
    xc = sum_x / float(len((countour)))
    yc = sum_y / float(len((countour)))
    return xc,yc

def get_beg_point(countour,xc,yc):
    max = 0
    beg_point = -1
    for i in range(0, len(countour)):
        point = countour[i]
        x = float(point[0][0])
        y = float(point[0][1])
        dx = x - xc
        dy = y - yc
        r = math.sqrt(dx * dx + dy * dy)
        if r > max:
            max = r
            beg_point = i
        return beg_point

def get_polar_coordinates(x0,y0,x,y,xc,yc):
    #Первая координата в полярных координатах - радиус
    dx=xc-x
    dy=yc-y
    r=math.sqrt(dx*dx+dy*dy)

    #Вторая координата в полярных координатах - узел, вычислим относительно начальной точки
    dx0=xc-x0
    dy0=yc-y0
    r0 = math.sqrt(dx0 * dx0 + dy0 * dy0)
    scal_mul=dx0*dx+dy0*dy
    cos_angle=scal_mul/r/r0
    sgn=dx0*dy-dx*dy0 #опредедляем, в какую сторону повернут вектор
    if cos_angle>1:
        if cos_angle>1.0001:
            raise Exception("Что-то пошло не так")
        cos_angle=1
    angle=math.acos(cos_angle)
    if sgn<0:
        angle=2*math.pi-angle
    return angle,r

def polar_to_decart(angle,r):
    x=math.sin(angle)*r
    y=math.cos(angle)*r
    return x,y

def polar_sort(item):
    return item[0]

def get_polar_coordinates_list(countour,xc,yc,beg_point):
    polar_coordinates = []
    x0 = countour[beg_point][0][0]
    y0 = countour[beg_point][0][1]
    for point in countour:
        x = int(point[0][0])
        y = int(point[0][1])
        angle, r = get_polar_coordinates(x0, y0, x, y, xc, yc)
        polar_coordinates.append((angle, r))

    # Создадим вектор описание
    polar_coordinates.sort(key=polar_sort)

    return polar_coordinates
