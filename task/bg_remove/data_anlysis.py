# from tifffile import tifffile
# import pickle
# from matplotlib import pyplot as plt
# from utils import *
#
# # mean = {'raw': [], 'gt': []}
# # variance = {'raw': [], 'gt': []}
# # percentile99 = {'raw': [], 'gt': []}
# # percentile98 = {'raw': [], 'gt': []}
# # percentile95 = {'raw': [], 'gt': []}
# # for i in range(2, 7):
# #     path_raw = '/home/wwh/Dataset/simulation/remove_bessel_bg_signal/raw_' + str(i) + '.tif'
# #     path_gt = '/home/wwh/Dataset/simulation/remove_bessel_bg_signal/bg_removed_' + str(i) + '.tif'
# #     raw_mov = tifffile.imread(path_raw)
# #     gt_mov = tifffile.imread(path_gt)
# #
# #     raw_mov = (raw_mov - raw_mov.min()) / (raw_mov.max() - raw_mov.min())
# #     gt_mov = (gt_mov - gt_mov.min()) / (gt_mov.max() - gt_mov.min())
# #
# #     with open('/home/wwh/code/DLT-main/dataset/bg_removed_' + str(i) + '.pkl', 'wb') as f:
# #          pickle.dump(gt_mov, f)
# #     with open('/home/wwh/code/DLT-main/dataset/raw_' + str(i) + '.pkl', 'wb') as f:
# #         pickle.dump(raw_mov, f)
#
# #     mean['raw'].append(raw_mov.mean(axis=(0, 1, 2)))
# #     mean['gt'].append(gt_mov.mean(axis=(0, 1, 2)))
# #     variance['raw'].append(raw_mov.var())
# #     variance['gt'].append(gt_mov.var())
# #     percentile99['raw'].append(np.percentile(raw_mov, 99))
# #     percentile99['gt'].append(np.percentile(gt_mov, 99))
# #     percentile98['raw'].append(np.percentile(raw_mov, 98))
# #     percentile98['gt'].append(np.percentile(gt_mov, 98))
# #     percentile95['raw'].append(np.percentile(raw_mov, 95))
# #     percentile95['gt'].append(np.percentile(gt_mov, 95))
# #
# # print('mean: ', mean)
# # print('variance: ', variance)
# # print('percentile99: ', percentile99)
# # print('percentile98: ', percentile98)
# # print('percentile95: ', percentile95)
# # mean:          {'raw': [0.0475, 0.0336, 0.0417, 0.0261, 0.0466, 0.0430],
# #                 'gt':  [0.0090, 0.0064, 0.0075, 0.0058, 0.0105, 0.0111]}
# # percentile99:  {'raw': [0.1001, 0.0669, 0.0866, 0.0578, 0.0991, 0.0877],
# #                 'gt':  [0.0468, 0.0353, 0.0411, 0.0333, 0.0584, 0.0584]}
# # percentile98:  {'raw': [0.0920, 0.0617, 0.0795, 0.0528, 0.0911, 0.0809],
# #                 'gt':  [0.0393, 0.0291, 0.0335, 0.0270, 0.0471, 0.0489]}
# # percentile95:  {'raw': [0.0811, 0.0547, 0.0699, 0.0462, 0.0803, 0.0715],
# #                 'gt':  [0.0294, 0.0213, 0.0247, 0.0197, 0.0340, 0.0366]}
# # variance:  {'raw': [0.00035171205, 0.00014855048, 0.0002636308, 0.00013145068, 0.00036823045, 0.00026276274],
# #             'gt':  [0.000109559754, 6.336097e-05, 9.4504016e-05, 5.4848424e-05, 0.00015771679, 0.00017056944]}
#
#
# # for i in range(1, 7):
# #     path_raw = '/home/wwh/Dataset/simulation/remove_bessel_bg_signal/raw_' + str(i) + '.tif'
# #     path_gt = '/home/wwh/Dataset/simulation/remove_bessel_bg_signal/bg_removed_' + str(i) + '.tif'
# #     raw_mov = tifffile.imread(path_raw)
# #     gt_mov = tifffile.imread(path_gt)
# #     raw_mov[raw_mov < 0] = 0
# #     gt_mov[gt_mov < 0] = 0
# #     raw_mov_ = np.log(raw_mov+1)[100:500, :, :].flatten()
# #     raw_mov_ = (raw_mov_ - raw_mov_.min()) / (raw_mov_.max() - raw_mov_.min())
# #     gt_mov_ = np.log(gt_mov+1)[100:500, :, :].flatten()
# #     gt_mov_ = (gt_mov_ - gt_mov_.min()) / (gt_mov_.max() - gt_mov_.min())
# #     plt.hist(raw_mov_, bins=100, alpha=0.5, label='raw')
# #     plt.show()
# #     plt.hist(gt_mov_, bins=100, alpha=0.5, label='gt')
# #     plt.show()
#
# from tqdm import tqdm
#
# mean = {'raw': [], 'gt': []}
# variance = {'raw': [], 'gt': []}
# max = {'raw': [], 'gt': []}
# min = {'raw': [], 'gt': []}
# percentile995 = {'raw': [], 'gt': []}
# log_mean = {'raw': [], 'gt': []}
# log_max = {'raw': [], 'gt': []}
# log_variance = {'raw': [], 'gt': []}
# for i in tqdm(range(1, 28)):
#     path_raw = '/home/wwh/Dataset/simulation/segmentation/raw_20um_' + str(i) + '.tif'
#     path_gt = '/home/wwh/Dataset/simulation/segmentation/clean_20um_' + str(i) + '.tif'
#     raw_mov = tifffile.imread(path_raw)
#     gt_mov = tifffile.imread(path_gt)
#
#     # with open('/home/wwh/code/DLT-main/dataset/bg_removed_' + str(i) + '.pkl', 'rb') as f:
#     #      gt_mov = pickle.load(f)
#     # with open('/home/wwh/code/DLT-main/dataset/raw_' + str(i) + '.pkl', 'rb') as f:
#     #      raw_mov = pickle.load(f)
#
#     mean['raw'].append(raw_mov.mean(axis=(0, 1, 2)))
#     mean['gt'].append(gt_mov.mean(axis=(0, 1, 2)))
#     variance['raw'].append(raw_mov.var())
#     variance['gt'].append(gt_mov.var())
#     max['raw'].append(raw_mov.max())
#     max['gt'].append(gt_mov.max())
#     min['raw'].append(raw_mov.min())
#     min['gt'].append(gt_mov.min())
#     percentile995['raw'].append(np.percentile(raw_mov, 99.5))
#
#     raw_mov = preprocess(raw_mov, False)
#     gt_mov = preprocess(gt_mov, True)
#     log_mean['raw'].append(raw_mov.mean(axis=(0, 1, 2)))
#     log_mean['gt'].append(gt_mov.mean(axis=(0, 1, 2)))
#     log_max['raw'].append(raw_mov.max())
#     log_max['gt'].append(gt_mov.max())
#     log_mean['gt'].append(gt_mov.mean(axis=(0, 1, 2)))
#     log_variance['raw'].append(raw_mov.var())
#     log_variance['gt'].append(gt_mov.var())
#
# print('mean: ', mean)
# print('variance: ', variance)
# print('max: ', max)
# print('min: ', min)
# print('percentile995: ', percentile995)
# print('log_mean_gt: ', np.mean(log_mean['gt']))
# print('log_mean_raw: ', np.mean(log_mean['raw']))
# print('log_max_gt: ', np.max(log_max['gt']))
# print('log_max_gt: ', np.max(log_max['raw']))
# print('log_variance_gt: ', np.mean(log_variance['gt']))
# print('log_variance_raw: ', np.mean(log_variance['raw']))
# # mean:          {'raw': [2250.2476, 2375.8533, 2365.6736, 2090.6626, 2274.5188, 2311.9846],
# #                 'gt': [4.5400586, 4.5876675, 4.511928, 4.7343, 4.7000775, 4.399604]}
# # variance:      {'raw': [794319.3, 745742.06, 856255.4, 848647.0, 884616.5, 765517.7],
# #                 'gt': [26.419575, 29.565718, 32.568474, 32.73667, 31.267006, 25.466358]}
# # max:           {'raw': [47512.098, 70841.89, 56979.945, 80338.99, 49000.992, 53963.27],         59772.8642
# #                 'gt': [491.1429, 683.2775, 587.1079, 772.7862, 445.2688, 386.49542]}            561.01312
# # min:           {'raw': [-11.0, -11.0, -11.0, -10.0, -12.845743, -12.0],
# #                 'gt': [0.081879035, 0.18209997, 0.061389394, 0.22242191, 0.015889635, 0.09854069]}
# # log_mean:      {'raw': [7.636733, 7.704783, 7.6868687, 7.5435953, 7.6323075, 7.6728888],        7.64619605
# #                 'gt': [1.442905, 1.4472319, 1.4215324, 1.460484, 1.4555286, 1.4207346]}         1.44140275
# # log_max:       {'raw': [10.768761, 11.16822, 10.950472, 11.294023, 10.799616, 10.896077],       10.979528
# #                 'gt': [6.198769, 6.5283637, 6.3769107, 6.6512957, 6.1009216, 5.959704]}         6.30266078
# # log_variance:  {'raw': [0.1798176, 0.148763, 0.19322723, 0.22508916, 0.23946372, 0.15794264],   0.19071723
# #                 'gt': [0.47058663, 0.46969518, 0.4867229, 0.49169853, 0.4942039, 0.45996764]}   0.47881246
# '''
# segmentation_dataset:
# log_mean_gt:  1.5690196
# log_mean_raw:  7.896325
# log_max_gt:  7.3532133
# log_max_gt:  11.954944
# log_variance_gt:  0.6133263
# log_variance_raw:  0.18054935
# '''
#
#
#
# # mean = {'raw': [], 'gt': []}
# # variance = {'raw': [], 'gt': []}
# # max = {'raw': [], 'gt': []}
# # min = {'raw': [], 'gt': []}
# # log_mean = {'raw': [], 'gt': []}
# # log_max = {'raw': [], 'gt': []}
# # log_variance = {'raw': [], 'gt': []}
# #
# # for i in range(7, 9):
#     # path_raw = '/home/wwh/Dataset/simulation/remove_bessel_bg_signal/test/raw_' + str(i) + '.tif'
#     # path_gt = '/home/wwh/Dataset/simulation/remove_bessel_bg_signal/test/bg_removed_' + str(i) + '.tif'
#     # raw_mov = tifffile.imread(path_raw)
#     # gt_mov = tifffile.imread(path_gt)
#
#     # with open('/home/wwh/code/DLT-main/dataset/bg_removed_' + str(i) + '.pkl', 'rb') as f:
#     #      gt_mov = pickle.load(f)
#     # with open('/home/wwh/code/DLT-main/dataset/raw_' + str(i) + '.pkl', 'rb') as f:
#     #      raw_mov = pickle.load(f)
#
#     # mean['raw'].append(raw_mov.mean(axis=(0, 1, 2)))
#     # mean['gt'].append(gt_mov.mean(axis=(0, 1, 2)))
#     # variance['raw'].append(raw_mov.var())
#     # variance['gt'].append(gt_mov.var())
#     # max['raw'].append(raw_mov.max())
#     # max['gt'].append(gt_mov.max())
#     # min['raw'].append(raw_mov.min())
#     # min['gt'].append(gt_mov.min())
#
#     # gt_mov[gt_mov < 0] = 0
#     # raw_mov[raw_mov < 0] = 0
#     # log_mean['raw'].append(np.log(raw_mov+1).mean(axis=(0, 1, 2)))
#     # log_mean['gt'].append(np.log(gt_mov+1).mean(axis=(0, 1, 2)))
#     # log_max['raw'].append(np.log(raw_mov + 1).max())
#     # log_max['gt'].append(np.log(gt_mov+1).max())
#     # log_variance['raw'].append(np.log(raw_mov+1).var())
#     # log_variance['gt'].append(np.log(gt_mov+1).var())
# #
# # print('mean: ', mean)
# # print('variance: ', variance)
# # print('max: ', max)
# # print('min: ', min)
# # print('log_mean: ', log_mean)
# # print('log_max: ', log_max)
# # print('log_variance: ', log_variance)
# # mean:  {'raw': [2236.3652, 2251.6074],
# #         'gt': [4.526806, 4.828477]}
# # variance:  {'raw': [808804.4, 841930.5],
# #             'gt': [26.816362, 31.871704]}
# # max:  {'raw': [38963.203, 57664.04],
# #        'gt': [332.86972, 583.90015]}
# # min:  {'raw': [-12.0, -11.0],
# #        'gt': [0.05745644, 0.08263296]}
# # log_mean:  {'raw': [7.620234, 7.633151],
# #             'gt': [1.4387136, 1.4873837]}
# # log_max:  {'raw': [10.570398, 10.962406],
# #            'gt': [5.810751, 6.3714414]}
# # log_variance:  {'raw': [0.22821319, 0.18983251],
# #                 'gt': [0.46809676, 0.47723398]}
#
# # for i in range(1, 7):
# #
# #     with open('/home/wwh/code/DLT-main/dataset/bg_removed_' + str(i) + '.pkl', 'rb') as f:
# #          gt_mov = pickle.load(f)
# #     with open('/home/wwh/code/DLT-main/dataset/raw_' + str(i) + '.pkl', 'rb') as f:
# #          raw_mov = pickle.load(f)
# #
# #     # 0-1 max_min normalization
# #     raw_mov_ = (raw_mov - raw_mov.min()) / (raw_mov.max() - raw_mov.min())
# #     gt_mov_ = (gt_mov - gt_mov.min()) / (gt_mov.max() - gt_mov.min())
# #     plt.hist(raw_mov_[:500, :, :].flatten(), bins=100, alpha=0.5, label='raw')
# #     plt.show()
# #     plt.hist(gt_mov_[:500, :, :].flatten(), bins=100, alpha=0.5, label='gt')
# #     plt.show()
# #     # log standardization
# #     gt_mean, gt_std = 1.44140275, 0.47881246
# #     pred_mean, pred_std = 7.64619605, 0.19071723
# #
# #     raw_mov[raw_mov < 0] = 0
# #     gt_mov[gt_mov < 0] = 0
# #     raw_mov_ = (np.log(raw_mov + 1) - pred_mean) / pred_std
# #     gt_mov_ = (np.log(gt_mov + 1) - gt_mean) / gt_std
# #     plt.hist(raw_mov_[:500, :, :].flatten(), bins=100, alpha=0.5, label='raw')
# #     plt.show()
# #     plt.hist(gt_mov_[:500, :, :].flatten(), bins=100, alpha=0.5, label='gt')
# #     plt.show()


