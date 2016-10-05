import cv2
import sys
import math
import numpy as np

import matplotlib.pyplot as plt

class BezierCurve():
    def __init__(self, inp):
        self.points = np.asarray( map(lambda x : [x[0][0], x[1][0]], inp) )


class Bezier():
    """Find the piece-wise Bezier Curve"""
    def __init__(self, err):
        self.error =  err
        self.curves = []

    def get_control_points(self, points, start, end):
        N = end - start     
        P_O = np.asarray(points[start])
        P_3 = np.asarray(points[end])

        C_1 = np.asarray([[0.0],[0.0]])
        for i in range(N + 1):
            ti = i / (N*1.0)
            C_1 += 3 * ti * ((1-ti)**2) * ( points[start + i] - ((1-ti)**3)*P_O - (ti**3)*P_3 )

        C_2 = np.asarray([[0.0],[0.0]])
        for i in range(N + 1):
            ti = i / (N*1.0)
            C_2 += 3 * (1-ti) * ((ti)**2) * ( points[start + i] - ((1-ti)**3)*P_O - (ti**3)*P_3 )

        A_1 = 0.0
        for i in range(N + 1):
            ti = i / (N*1.0)
            A_1 += (ti**2) * ((1-ti)**4)
        A_1 *= 9

        A_2 = 0.0
        for i in range(N + 1):
            ti = i / (N*1.0)
            A_2 += (ti**4) * ((1-ti)**2)
        A_2 *= 9
        
        A_12 = 0.0
        for i in range(N + 1):
            ti = i / (N*1.0)
            A_12 += (ti**3) * ((1-ti)**3)
        A_12 *= 9
        
        P_1 = (A_2 * C_1 - A_12 * C_2) / (A_1 * A_2 - A_12 * A_12)
        P_2 = (A_1 * C_2 - A_12 * C_1) / (A_1 * A_2 - A_12 * A_12)

        return np.asarray([P_O, P_1, P_2, P_3])

    def get_point(self, CP, t):
        return ((1-t)**3)*CP[0] + 3*t*((1-t)**2)*CP[1] + 3*(1-t)*(t**2)*CP[2] + (t**3)*CP[3]  

    def distance(self, p1, p2):
        return ((p1[0][0] - p2[0][0])**2 + (p1[1][0] - p2[1][0])**2)**0.5

    def get_max_error_index(self, points, start, end, CP):
        error = 0
        index = start
        N = end - start + 1
        if N <= 4 :
            return -1

        for i in range(N):
            t = i / ((N-1)*1.0)
            point = self.get_point(CP, t)
            err = self.distance(points[start + i], point)
            if err > error : 
                error = err
                index = start + i

        print error, index
        if error < self.error:
            return -1 # just don't do any subdivision now

        if index < start + 4:
            return start + 4
        elif index > end - 4:
            return end - 4

        return index

    def gen_curves(self, points, start, end):
        CP = self.get_control_points(points, start, end)

        N = end - start + 1
        if N<8:
            self.curves.append(BezierCurve(CP))
            return

        index = self.get_max_error_index(points, start, end, CP)
        # if index == -1 or (index - start + 1) < 4 or (end - index + 1) < 4:
        if index == -1:
            self.curves.append(BezierCurve(CP))
            return

        self.gen_curves(points, start, index)
        self.gen_curves(points, index, end)
        return

    def plotBZ(self, CP):
        t = 0
        CPP = np.asarray(map(lambda x : [[x[0]], [x[1]]], CP))
        inc = 10**-2
        pts = []
        while True:
            point = self.get_point(CPP, t)
            pts.append([point[0][0], point[1][0]])
            t += inc
            if t >= 1:
                break
        return np.asarray(pts)


points = []
th = 0
r = 2
while th<=2*3.14:
    points.append([r*math.cos(th),r*math.sin(th)])
    th+=0.1

points.append(points[0])
# print points

plt.clf()
# points = np.asarray(points)
# plt.scatter(points[:,0], points[:,1], c = u'r')
points = np.asarray( map(lambda x : np.asarray([ [ x[0] ], [ x[1]] ]), points ) )
bz = Bezier(10**-4)
# print 'len(points)', len(points) - 1
curves = bz.gen_curves(points, 0 , len(points) - 1)
print 'No. : ',len(bz.curves)
for curve in bz.curves:
    pts = bz.plotBZ(curve.points)
    plt.scatter(pts[:,0], pts[:,1], c = u'g')
    # print curve.points
    plt.scatter(curve.points[:,0], curve.points[:,1], c = u'y')
plt.show()