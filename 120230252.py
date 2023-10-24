import cv2
import numpy as np

def compute_homography(src_pts, dst_pts):
    A = []
    for (x1, y1), (x2, y2) in zip(src_pts, dst_pts):
        A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def ransac_for_homography(matches, kp1, kp2, threshold=5.0, iterations=1000):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    max_inliers = []
    final_H = None

    if len(src_pts) < 4:
        print("Not enough points to find homography!")
        return None

    for i in range(iterations):
        # Randomly select 4 pairs of points
        indices = np.random.choice(len(src_pts), 4, replace=False)
        src = src_pts[indices]
        dst = dst_pts[indices]

        # Compute the homography matrix
        H = compute_homography(src, dst)

        # Calculate inliers
        transformed_pts = np.dot(H, np.vstack((src_pts.T, np.ones(src_pts.shape[0]))))
        transformed_pts /= transformed_pts[2]
        transformed_pts = transformed_pts[:2].T

        dist = np.sqrt(np.sum((dst_pts - transformed_pts)**2, axis=1))
        inliers = [idx for idx, d in enumerate(dist) if d < threshold]

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            final_H = H

    good_matches = [matches[i] for i in max_inliers]

    return final_H, good_matches

def apply_homography(pt, H):
    x, y = pt
    transformed_pt = np.dot(H, [x, y, 1])
    transformed_pt /= transformed_pt[2]
    return transformed_pt[:2]

def warp_image_simple(img, H, output_shape):
    h, w, _ = img.shape
    output_img = np.zeros(output_shape, dtype=np.uint8)
    
    H_inv = np.linalg.inv(H)
    for y in range(output_shape[0]):
        for x in range(output_shape[1]):
            src_x, src_y = apply_homography([x, y], H_inv)
            src_x, src_y = int(src_x), int(src_y)
            if 0 <= src_x < w and 0 <= src_y < h:
                output_img[y, x] = img[src_y, src_x]
                
    return output_img

def warpImages(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img1_points = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]])
    temp_points = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]])
    
    img2_points = np.array([apply_homography(pt, H) for pt in temp_points])
    
    points = np.vstack([img1_points, img2_points])

    [x_min, y_min] = np.int32(np.min(points, axis=0))
    [x_max, y_max] = np.int32(np.max(points, axis=0))

    output_shape = (y_max - y_min + 1, x_max - x_min + 1, 3)

    H_translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    final_H = np.dot(H_translation, H)
    output_img = warp_image_simple(img2, final_H, output_shape)
    cv2.imwrite("sex.jpg", output_img)
    output_img[-y_min:-y_min+h1, -x_min:-x_min+w1] = img1

    return output_img

def main():
    img1 = cv2.imread('image1.png') # Change to your image path
    img2 = cv2.imread('image2.png') # Change to your image path

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    M, good_matches = ransac_for_homography(matches, kp1, kp2)
    if M is None:
        return
    
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('panorama_match_point_test.jpg', img_matches)
    img_output = warpImages(img2, img1, M)
    cv2.imwrite('panorama_test.jpg', img_output)

if __name__ == '__main__':
    main()
