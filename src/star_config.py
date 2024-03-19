import os
path_male_star = '/home/samk/deformable_shape_space/data/star_1_1/male/model.npz'
path_female_star = '/home/samk/deformable_shape_space/data/star_1_1/female/model.npz'
path_neutral_star = '/home/samk/iterative_shape_registration/data/star_1_1/neutral/model.npz'

data_type = 'float32'

if data_type not in ['float16','float32','float64']:
    raise RuntimeError('Invalid data type %s'%(data_type))

class meta(object):
    pass 

cfg = meta()
cfg.data_type = data_type

cfg.path_male_star    = path_male_star
cfg.path_female_star  = path_female_star
cfg.path_neutral_star = path_neutral_star
