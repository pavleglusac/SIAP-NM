import imageio
import cv2

reader = imageio.get_reader('test.ts')
image = reader.get_data(0)
cv2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.imshow('win', cv2_image)
cv2.waitKey(0)
