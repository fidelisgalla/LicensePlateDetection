import numpy as np

class ArrayModificationOrder:
    def __init__(self,array):
        self.array = array

    def areaRects(self):
        areaRects = []
        for i in range(self.array.shape[0]):
            coord1x, coord1y, coord2x, coord2y, coord3x, coord3y, coord4x, coord4y = self.array[i][0][0], self.array[i][0][1], \
                                                                                     self.array[i][1][0], self.array[i][1][1], \
                                                                                     self.array[i][2][0], self.array[i][2][1], \
                                                                                     self.array[i][3][0], self.array[i][3][1]
            h = coord3y - coord2y
            w = coord2x - coord1x
            areaRect = w * h
            areaRects.append(areaRect)

        return areaRects

    def coordRects(self):
        coord3ys = []
        for i in range(self.array.shape[0]):
            coord3y = self.array[i][2][1]
            coord3ys.append(coord3y)
        return coord3ys

    def condOne(self):
        locations = []
        for i, coord in enumerate(self.coordRects()):
            delta = np.abs(coord - np.median(self.coordRects()))
            if (delta / np.median(self.coordRects())) < 0.15:
                locations.append(i)
        return locations

    def condTwo(self):
        locations = []
        for i, area in enumerate(self.areaRects()):
            delta = area - np.median(self.areaRects())
            if delta > 0 or (delta / np.median(self.areaRects())) < 0.1:
                locations.append(i)
        return locations

    def newArray(self, valueofpadding=0.05):

        location = self.condOne()
        filteredArrays = []

        for i in location:
            filteredArray = self.array[i]
            filteredArrays.append(filteredArray)

        coord1xs = []
        coord1ys = []
        coord2xs = []
        coord2ys = []
        coord3xs = []
        coord3ys = []
        coord4xs = []
        coord4ys = []

        for array in filteredArrays:
            coord1x = array[0][0]
            coord1y = array[0][1]
            coord2x = array[1][0]
            coord2y = array[1][1]
            coord3x = array[2][0]
            coord3y = array[2][1]
            coord4x = array[3][0]
            coord4y = array[3][1]
            coord1xs.append(coord1x)
            coord1ys.append(coord1y)
            coord2xs.append(coord2x)
            coord2ys.append(coord2y)
            coord3xs.append(coord3x)
            coord3ys.append(coord3y)
            coord4xs.append(coord4x)
            coord4ys.append(coord4y)

        newArray =[[min(coord1xs) + min(coord1xs) * valueofpadding, min(coord1ys) + min(coord1ys) * valueofpadding],
             [max(coord2xs) + max(coord2xs) * valueofpadding, min(coord2ys) + min(coord2ys) * valueofpadding],
             [max(coord3xs) + max(coord3xs) * valueofpadding, max(coord3ys) + max(coord3ys) * valueofpadding],
             [min(coord4xs) + max(coord4xs) * valueofpadding, min(coord4ys) + max(coord4ys) * valueofpadding]]

        return newArray