from shutil import copyfile


ori = '/home/xhg/KPCN/tungsten_docker/result/'
dst = '/home/xhg/deep_video/datasets/train/living-room-2/'

for i in range(120):
		
	im1 = ori +  str(i).zfill(10) + '.exr'
	im2 = ori +  str(i + 1).zfill(10) + '.exr'

	ds1 = dst + '1' + str(i).zfill(9) + '.exr'
	ds2 = dst + '1' + str(i).zfill(9) + '_a.exr'

	copyfile(im1, ds1)
	copyfile(im2, ds2)