import cv2
import numpy as np
from numpy.linalg import inv
import random
import  os

# opencv fonts:
# FONT_HERSHEY_SIMPLEX = 0,
# FONT_HERSHEY_DUPLEX = 2,
# FONT_HERSHEY_COMPLEX = 3,
# FONT_HERSHEY_TRIPLEX = 4,
# FONT_HERSHEY_COMPLEX_SMALL = 5,
# FONT_HERSHEY_SCRIPT_COMPLEX = 7,
FONTS = [0, 2, 3, 4, 5, 7]


def show_bbxs(img, bbxs):
    bbxs = bbxs.astype(np.int32)
    for i in range(len(bbxs)):
        # cv2.rectangle(img, (int(bbxs[i, 1]), int(bbxs[i, 0])), (int(bbxs[i, 3]), int(bbxs[i, 2])), (255, 0, 0), 2)
        pts = np.reshape(bbxs[i,:], (4,2))
        cv2.polylines(img, [pts], True, (255, 0, 0), 2)
    cv2.imshow('tmp', img)
    cv2.waitKey()

def rotation_matrix(angle):
    """ Construct a 2D rotation matrix.
    Args
        angle: the angle in radians
    Returns
        the rotation matrix as 2 by 2 numpy array
    """
    angle = angle * np.pi / 180
    return np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle),  np.cos(angle)]
    ])

def noisy(noise_typ,image):
    """
    :param noise_typ: str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.

    :param image: ndarray
    Input image data. Will be converted to float.
    :return: image with one type of noise
    """
    # image = np.array(image)
    if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return np.clip(noisy,0,255)
    elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return np.clip(out,0,255)
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return np.clip(noisy,0,255)
    elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return np.clip(noisy,0,255)

def generate_label1(save_path):
    txt_cfg = RandConfigTxt()

    for _ in range(40):
        label_str = '%010d' % (random.randint(1e8, 1e10))
        txt_cfg.generate_random_config()
        img, bbxs = generate_label_raw(label_str, txt_cfg)
        filename = label_str + '.png'
        cv2.imwrite(os.path.join(save_path, filename), img)


def generate_label2():
    pass


def generate_label_raw(label_str, txt_cfg):

    # get the rectangle size and baseline of the text
    ret1, baseline1 = cv2.getTextSize('1', txt_cfg.font, txt_cfg.scale, txt_cfg.thickness)
    ret2, baseline2 = cv2.getTextSize('12', txt_cfg.font, txt_cfg.scale, txt_cfg.thickness)
    ret, baseline = cv2.getTextSize(label_str, txt_cfg.font, txt_cfg.scale, txt_cfg.thickness)
    # print('ret1: (%d, %d) baseline1: %d' % (ret1[0], ret1[1], baseline1))
    # print('ret2: (%d, %d) baseline2: %d' % (ret2[0], ret2[1], baseline2))
    s = 2 * ret1[0] - ret2[0]
    w = ret1[0] - 2 * s
    img = np.uint8(np.zeros((int(ret[1] * 1.1), int(ret[0])) + (3,)) + txt_cfg.bg_color)
    cv2.putText(img, label_str, (0, int(ret[1])), txt_cfg.font, txt_cfg.scale, txt_cfg.color, txt_cfg.thickness)
    img_vis = np.copy(img)
    bboxes = []
    labels = []
    x1 = int(s / 2)
    y1 = 0
    x2 = int(s / 2) + w
    y2 = int(ret1[1] * 1.1) - 1
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    bboxes.append([x1, y1, x2, y1, x2, y2, x1, y2]) # lt,lb,rb,rt
    labels.append(label_str[0])
    for i in range(len(label_str)-1):
        x1 = x1 + s + w
        x2 = x2 + s + w
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        bboxes.append([x1, y1, x2, y1, x2, y2, x1, y2])
        labels.append(label_str[i+1])

    # cv2.imshow('tmp', img_vis)
    # cv2.waitKey()
    # cv2.imshow('tmp', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    T = Transformation(img, bboxes, margin4proj=100)
    img, bbxs = T.transform()
    TYPES = ['gauss', 'poisson', 's&p', 'speckle']
    n_types_idxs = np.random.randint(0,4,np.random.randint(1,4))
    img = noisy_image(img, np.take(TYPES,n_types_idxs))
    k = np.random.randint(1,3) * 2 + 1
    img = cv2.GaussianBlur(img, (k, k), 0)
    img = np.clip(img, 0, 255)

    return img, bbxs


def noisy_image(img, type_list):
    """

    :param img: input image array
    :param type_list: a list of string indicating noise type, valid type 'gauss', 'poisson', 's&p', 'speckle'
    :return: rendered image
    """
    for type in type_list:
        img = noisy(type, img)

    return img


class RandConfigTxt:
    def __init__(self,
                 font=None,
                 scale=5,
                 thickness=None,
                 margin=20,
                 color=(255, 255, 255),
                 bg_color=(0, 0, 0)):
        self.font = font
        self.scale = scale
        self.thickness = thickness
        self.margin = margin
        self.color = color
        self.bg_color = bg_color

        super(RandConfigTxt, self).__init__()

    def generate_random_config(self):
        self.font =  FONTS[random.randint(0, len(FONTS) - 1)]
        self.thickness = random.randint(5, 10)


class Transformation:
    def __init__(self,
                 img,
                 bboxes,
                 margin4proj=30,
                 corner_ratio=0.1,
                 scale_ratio_max=0.3,
                 rotation=5):
        self.corner_ratio = corner_ratio
        self.scale_ratio_max = scale_ratio_max
        self.rotation = rotation
        self.margin4proj = margin4proj
        self.lu = None
        self.ru = None
        self.lb = None
        self.rb = None
        self.img = img
        self.bboxs = bboxes
        self.src_points = None

    def transform(self):
        """Random transform the image according to the given parameters in init"""

        img, bbxs = self.scale(self.img, self.bboxs)
        # show_bbxs(img, bbxs)
        img, bbxs = self.rotate(img, bbxs)
        # show_bbxs(img, bbxs)
        img, bbx = self.project(img, bbxs)
        # show_bbxs(img, bbxs)
        # cv2.destroyAllWindows()
        return img, bbxs

    def rand_gen_trsf_paras(self, img):
        """Generate the parameters for projection of the image.
           The generated four points should lay in the 4 rectangles which locate in the 4 corner of the image,
           the rectangle size is determined by the parameter corner_ratio.
        """
        rows_, cols_ = img.shape[:2]
        rows, cols = [i + self.margin4proj for i in img.shape[:2]]
        corner_size = np.around(np.array([rows, cols]) * self.corner_ratio)
        offset_x = np.random.rand(4) * corner_size[0]
        offset_y = np.random.rand(4) * corner_size[1]
        m = 10
        n = 30
        self.lu = np.array([np.maximum(offset_x[0],n), np.maximum(offset_y[0], m)])
        self.ru = np.array([np.minimum(cols - offset_x[1],cols - n), np.maximum(offset_y[1], m)])
        self.lb = np.array([np.maximum(offset_x[2], n), np.minimum(rows - offset_y[2], rows - m)])
        self.rb = np.array([np.minimum(cols - offset_x[3], cols - n), np.minimum(rows - offset_y[3], rows-m)])
        self.src_points = np.float32([[0, 0], [cols_ - 1, 0], [0, rows_ - 1], [cols_ - 1, rows_ - 1]])

    def rotate(self, img, bbxs):
        """
        Padding the image with boarders so that all characters are still in the image
        Rotate the image by a random angle with in range [-self.rotation, self.rotation]
        """
        angle = (np.random.rand() - 0.5) * 2 * self.rotation
        rows, cols = img.shape[:2]
        corners = np.array([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        corners_rotated = np.dot(corners, M[:,:2])
        size_new = np.int32(corners_rotated.max(axis=0) - corners_rotated.min(axis=0))
        dx, dy = (size_new - [cols, rows]) // 2 + 5
        img = cv2.copyMakeBorder(img, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rows, cols = img.shape[:2]
        bbxs = bbxs + np.tile([dx-cols//2, dy-rows//2], 4)
        dst = cv2.warpAffine(img, M, (cols, rows))
        for i in range(0,8,2):
            bbxs[:,i:i+2] = np.dot(M[:,:2], np.transpose(bbxs[:,i:i+2])).transpose()
        bbxs = bbxs + np.tile([cols // 2, rows // 2], 4)
        return dst, bbxs

    def translate(self):
        pass

    def scale(self, img, bbxs, ratios=None):
        """
        Scaling an image with the parameter ration.

        Args:
            img: source image to be scaled
            bbxs: bounding boxes of the letters in the image
            ratios: List or Tuple of (f_x, f_y)
                    f_x: scale factor along the horizontal axis
                    f_y: scale factor along the vertical axis

        Returns:
            Image resized to size (f_y*original_h, f_x*original_w)
        """
        if ratios is None:
            ratios = 1 - np.random.rand(2) * self.scale_ratio_max
        dsize = (np.int32(img.shape[1] * ratios[0]), np.int32(img.shape[0] * ratios[1]))
        bbxs = np.array(bbxs) * np.tile(ratios, 4)
        return cv2.resize(img, dsize), bbxs

    def project(self, img, bbxs):
        """Regard the image as 3d Plane and project it randomly from 3d to 2d,
           Besides, this projective matrix is used to transform the bounding boxes.
        """
        self.rand_gen_trsf_paras(img)
        rows, cols = [i + self.margin4proj for i in img.shape[:2]]
        dst_points = np.array([self.lu, self.ru, self.lb, self.rb], dtype=np.float32)
        P = cv2.getPerspectiveTransform(self.src_points, dst_points)
        img_output = cv2.warpPerspective(img, P, (cols, rows))
        for i in range(0,8,2):
            bbxs_i = np.ones((len(bbxs),3))
            bbxs_i[:,:2] = bbxs[:,i:i+2]
            bbxs_i= np.dot(P, np.transpose(bbxs_i)).transpose()
            bbxs[:, i:i + 2] = bbxs_i[:,:2] / np.tile(bbxs_i[:,2],(2,1)).transpose()
        return img_output, bbxs



generate_label1('/home/robotics/OCR_synthetic_data/sythetic_data')



