import numpy as np

from .constants_vuer import grd_yup2grd_zup, hand2inspire
import numpy as np

def mat_update(prev_mat, mat):
    if np.linalg.det(mat) == 0:
        return prev_mat
    else:
        return mat


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


class VuerPreprocessor:
    def __init__(self):
        self.vuer_head_mat = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 1.5],
                                  [0, 0, 1, -0.2],
                                  [0, 0, 0, 1]])
        self.vuer_right_wrist_mat = np.array([[1, 0, 0, 0.5],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, -0.5],
                                         [0, 0, 0, 1]])
        self.vuer_left_wrist_mat = np.array([[1, 0, 0, -0.5],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, -0.5],
                                        [0, 0, 0, 1]])

    def process(self, tv):
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
        # self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())
        # change of basis
        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        # left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)

        # rel_left_wrist_mat = left_wrist_mat @ hand2inspire
        # rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]

        rel_right_wrist_mat = right_wrist_mat @ hand2inspire  # wTr = wThd @ hTr
        # 相对头坐标
        # rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]
        rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] 


        button = tv.button.copy()
        return head_mat, rel_right_wrist_mat, button# , rel_left_fingers, rel_right_fingers



