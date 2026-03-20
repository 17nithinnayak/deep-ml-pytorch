# Task: Image Brightness Calculator
# In this task, you will implement a function calculate_brightness(img) that calculates the average brightness of a grayscale image. The image is represented as a 2D matrix, where each element represents a pixel value between 0 (black) and 255 (white).
def calculate_brightness(img):
	# Write your code here
	if not img or not img[0]:
		return -1

	rows = len(img)
	cols = len(img[0])

	pixels = 0
	sump = 0

	for row in img:
		if len(row) != cols:
			return -1

		for pixel in row:
			if pixel<0 or pixel>255:
				return -1
			sump+=pixel
			pixels+=1

	avg = sump/pixels

	return round(avg, 2)


