def intersection(box_a,box_b):
	base_y = box_a[0]
	base_x = box_a[1]
	base_h = box_a[2]
	base_w = box_a[3]

	test_y = box_b[0]
	test_x = box_b[1]
	test_h = box_b[2]
	test_w = box_b[3]

	base_area = base_h * base_w
	test_area = test_h * test_w
	
	vert_below_base = base_y <= test_y
	hori_beyond_base = base_x <= test_x

	overlap_area = 0.

	try:
		assert base_y != test_y or base_x != test_x, 'center point of boxes are equal'
				
		if vert_below_base and hori_beyond_base:
			# test is right below base
			base_corner_y = base_y + base_h / 2.
			test_corner_y = test_y - test_h / 2.
			vert_overlap = base_corner_y - test_corner_y

			base_corner_x = base_x + base_w / 2.
			test_corner_x = test_x - test_w / 2.
			hori_overlap = base_corner_x - test_corner_x
			pass
		elif vert_below_base and not hori_beyond_base:
			# test is left below base
			base_corner_y = base_y + base_h / 2.
			test_corner_y = test_y - test_h / 2.
			vert_overlap = base_corner_y - test_corner_y

			base_corner_x = base_x - base_w / 2.
			test_corner_x = test_x + test_w / 2.
			hori_overlap = test_corner_x - base_corner_x
			pass
		elif not vert_below_base and hori_beyond_base:
			# test is right above base
			base_corner_y = base_y - base_h / 2.
			test_corner_y = test_y + test_h / 2.
			vert_overlap = test_corner_y - base_corner_y

			base_corner_x = base_x + base_w / 2.
			test_corner_x = test_x - test_w / 2.
			hori_overlap = base_corner_x - test_corner_x
			pass
		elif not vert_below_base and not hori_beyond_base:
			# test is left above base
			base_corner_y = base_y - base_h / 2.
			test_corner_y = test_y + test_h / 2.
			vert_overlap = test_corner_y - base_corner_y

			base_corner_x = base_x - base_w / 2.
			test_corner_x = test_x + test_w / 2.
			hori_overlap = test_corner_x - base_corner_x
			pass
		else:
			raise Exception('any case should hold')
			pass

		vert_overlap = max(0.,vert_overlap)
		hori_overlap = max(0.,hori_overlap)
		overlap_area = min(base_h,test_h,vert_overlap)*min(base_w,test_w,hori_overlap)
	except AssertionError:
		# base_y == test_y and base_y == test_x
		overlap_area = min(base_h,test_h)*min(base_w,test_w)
	except Exception as e:
		print(e)
		pass
	
	return base_area,test_area,overlap_area

def jaccard_similarity(box_a,box_b):
	base_area, test_area, overlap_area = intersection(box_a,box_b)
	assert overlap_area <= base_area and overlap_area <= test_area, 'overlap area cannot be larger than one of the boxes'	
	intersec = overlap_area / (base_area + test_area - overlap_area)
	assert 0. <= intersec <= 1., 'IoU value must be between 0 and 1'
	return intersec