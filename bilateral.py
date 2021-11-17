import cv2
import numpy as np

### self-bilateral ###
# 窓半径3（つまり窓幅7）、画素値の標準偏差30 （つまりσ2=30）
# Here: filter_diameter=7, sigma1=30 sigma2=30
def bilateral_filter(img, filter_diameter=7, sigma1=30, sigma2=30):
    img_filtered = np.zeros(img.shape)

    for row in range(len(img)):
        for col in range(len(img[0])):
           img_filtered[row][col] = bilateral_filter_kernel(img, row, col, filter_diameter, sigma1, sigma2)

    return img_filtered

def bilateral_filter_kernel(img, row, col, filter_diameter, sigma1, sigma2):
    radius = filter_diameter//2
    filtered_value = []
    weight_sum = []

    for x in range(filter_diameter):
        for y in range(filter_diameter):
            neighbour_row = row - (radius - x)
            neighbour_col = col - (radius - y)

            # boundary process
            if neighbour_row >= len(img):
                neighbour_row -= len(img)
            if neighbour_col >= len(img[0]):
                neighbour_col -= len(img[0])
            if neighbour_row < 0:
                neighbour_row += len(img)
            if neighbour_col < 0:
                neighbour_col += len(img[0])

            # get gaussian weight of image value
            gi = gaussian(int(img[neighbour_row][neighbour_col] - img[row][col]), sigma1)
            # get weight based on distance
            gd = gaussian(distance(neighbour_row, neighbour_col, row, col), sigma2)
            w = gi * gd

            # print(row, col, x,y, neighbour_row, neighbour_col, gi, gd)

            filtered_value.append(img[neighbour_row][neighbour_col] * w)
            weight_sum.append(w)


    filtered_value = sum(filtered_value) / sum(weight_sum) # filtered value

    return filtered_value

def gaussian(x, sigma):
    # return (1.0 / (2*np.pi*(sigma**2))) * np.exp(- (x**2) / (2*(sigma**2)))

    # constant values can be removed, because same function is used at denominator when calculating filtered_value
    return np.exp(-(x**2)/(2*(sigma**2))) 

def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)



# Read the image.
img = cv2.imread('Lena_gauss10.png', cv2.IMREAD_GRAYSCALE)
# opencv result
bilateral_cv = cv2.bilateralFilter(img, 7, 30, 30)

# self implementation result
bilateral_self = bilateral_filter(img, 7, 30, 30)
bilateral_self = bilateral_filter(bilateral_self, 7, 30, 30) # to improve denoising

# Save the output.
cv2.imwrite('bilateral_cv.png', bilateral_cv)
cv2.imwrite('bilateral_self.png', bilateral_self)