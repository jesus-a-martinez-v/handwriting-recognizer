from skimage import feature


class HOG(object):
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), transform=3):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.transform = transform

    def describe(self, image):
        histogram = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                                cells_per_block=self.cells_per_block, transform_sqrt=self.transform)

        return histogram
