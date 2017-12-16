import numpy as np

inf = float('inf')

def width_compatible(a,b):
	if np.isinf(a) or np.isinf(b):
		return False 
	return (a/b < 3 and b/a < 3)


def update_bounds(boundaries,x,y):
	boundaries[0] = min(boundaries[0],x)
	boundaries[1] = min(boundaries[1],y)
	boundaries[2] = max(boundaries[2],x)
	boundaries[3] = max(boundaries[3],y)
	return


def is_letter(boundaries, stroke_vals):
	bound_height = boundaries[3] - boundaries[1]
	bound_width = boundaries[2] - boundaries[0]
	bound_diag = np.sqrt((boundaries[2] - boundaries[0])**2 + (boundaries[3]-boundaries[1])**2)  
	aspect_ratio = bound_height/bound_width if bound_width else 0
	stroke_val_median = np.median(stroke_vals)
	stroke_val_mean = np.mean(stroke_vals)

	val = True
	val = val and (aspect_ratio < 10 and aspect_ratio > 0.5)
	val = val and (bound_diag < 10*stroke_val_median)
	val = val and (np.std(stroke_vals)< 0.5*stroke_val_mean)
	#******* my own conditionss
	val = val and (bound_height > 6)	
	val = val and (len(stroke_vals)/(bound_width*bound_height)>0.3)
	val = val and (1.2*bound_width > stroke_val_median)
	return val


#************************* Region growing implementaion *****************************

def region_growing_util(cc,swt,point,invalid_labels):

	chain_q = [point]
	# boundaries=[min_x,  min_y   ,max_x   ,max_y] 
	boundaries = [inf,inf,0,0]
	stroke_vals = []

	while(len(chain_q)!=0):
		pt  = chain_q.pop(0)
		i = pt[0]
		j = pt[1]

		update_bounds(boundaries,j,i)
		stroke_vals.append(swt[pt])

		neighbors = [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1),(i,j-1)]
		for neighbor in neighbors:
			try:
				if(cc[neighbor] == 0 and width_compatible(swt[pt],swt[neighbor])):
					cc[neighbor] = cc[pt]
					chain_q.append(neighbor)
			except IndexError:
				continue
	
	#Filtering process
	if not is_letter(boundaries,stroke_vals):
		invalid_labels.append(cc[point])		
	else:
		return boundaries

def relabel_rg(cc,invalid_labels):
	label_count = 0
	label_dict = {}
	for i in range(cc.shape[0]):
		for j in range(cc.shape[1]):
			if cc[i,j] == 0:
				continue
			if cc[i,j] in invalid_labels:
				cc[i,j] = 0
			else:
				# relabel all valid components  
				if not (cc[i,j] in label_dict):
					label_count+=1
					label_dict[cc[i,j]] = label_count	

				cc[i,j] = label_dict[cc[i,j]]
	return label_count

def region_growing(swt):
	cc = np.zeros(swt.shape)
	label_count = 0
	invalid_labels = []
	bounding_boxes = {}
	for i in range(swt.shape[0]):
		for j in range(swt.shape[1]):
			if swt[i,j] == inf:
				continue
			if cc[i,j] == 0:
				label_count+=1
				cc[i,j] = label_count
				boundaries = region_growing_util(cc,swt,(i,j),invalid_labels)
				if boundaries:
					bounding_boxes[label_count] = boundaries
	#Do this step in right after declaring a label as invalid ?? any efficient methods??
	label_count = relabel_rg(cc,invalid_labels)

	return cc,label_count,bounding_boxes


#**************************** Implementation by disjoint set ***************************************

def find_set(a,par_set):
	#returns the ultimate parent of a 
	while par_set[a] != a:
		a = par_set[a]
	return a

def union_set(neighbors,par_set):
	#union of sets of all points in neighbors and return the union set value
	#the ranking is such that set label with lower magnitude has higher priority
	 parents = [find_set(i,par_set) for i in neighbors]
	 common_set = min(parents)
	 for set_label in parents	:
	 	par_set[set_label] = common_set
	 return common_set

def disjoint_set(swt):
	cc = -1*np.ones(swt.shape,dtype = np.int32)
	label_count = -1
	par_set = []		# since set_labels are integers starting from 0 list indexing is preferred to dicts

	#NOTE :  we move bottom up while connecting components because in doing so,the list of letters formed are in
	#		 descending order of y (bottom up) and for same y,in ascending order of x (left to right)
	for i_t in range(swt.shape[0]):
		i = swt.shape[0]-i_t-1
		for j in range(swt.shape[1]):
			if swt[i,j] == inf:
				continue

			neighbors = [(i,j-1), (i+1,j-1), (i+1,j), (i+1,j+1)]
			neighbor_set = []

			for neighbor in neighbors:
				try:
					if(width_compatible(swt[i,j], swt[neighbor])):
						neighbor_set.append(find_set(cc[neighbor],par_set))
	
				except IndexError:
					continue

			if(neighbor_set):
				cc[i,j] = union_set(neighbor_set,par_set)
			else:
				label_count+=1
				cc[i,j] = label_count
				par_set.append(label_count)

	# 2nd pass to replace all set vals with ult parent vals and extract stroke widths and boundaries
	stroke_vals = {}
	boundaries = {}

	for it in range(cc.shape[0]):
		i = swt.shape[0]-1-it
		for j in range(cc.shape[1]):
			if cc[i,j] == -1:
				continue
			label = find_set(cc[i,j],par_set)
			cc[i,j] = label 
			if label not in stroke_vals:
				stroke_vals[label] = []
				boundaries[label] = [inf,inf,0,0]

			stroke_vals[label].append(swt[i,j])
			update_bounds(boundaries[label],j,i)

	letters = find_letter_candidates(cc,stroke_vals,boundaries)
	words = find_words(letters)

	return cc,letters,words

def find_letter_candidates(cc,stroke_vals,boundaries):

	##Any way to improve efficieny????
	valid_labels_dict = {}
	label_count = 0
	letters = []
	for label in stroke_vals:
		if is_letter(boundaries[label],stroke_vals[label]):
			label_count+=1
			bounds = [boundaries[label][0], 						# x_min -> x coordinate of bottom left corner 
					  boundaries[label][3],							# y_max -> y coordinate of botton left corner
					  boundaries[label][2] - boundaries[label][0],	# width
					  boundaries[label][3] - boundaries[label][1]]	# height

			letters.append({"stroke_median":np.median(stroke_vals[label]), "boundaries":bounds})

	return letters


#************************************* Finding words *****************************

def connect_letters(A,B):
	[xa , ya , wa , ha] = A['boundaries']
	[xb , yb , wb , hb] = B['boundaries']
	a_str = A['stroke_median']
	b_str = B['stroke_median']
	max_dist = 2*a_str

	val = True

	if(abs(xb-(xa+wa)) < max_dist ):
		dist = xb-(xa+wa)
		direction = 1
	elif(abs(xa - (xb+wb)) < max_dist ):
		dist = xa - (xb+wb)
		direction = -1
	else:
		return False,0,0

	val = val and ((a_str/b_str < 2) and (b_str/a_str <2))
	val = val and ((ha/hb < 2) and (hb/ha < 2))
	#Average color condition???

	return val,direction,dist


def get_word_boundary(word,letters):
	x_min  = letters[word[0]]['boundaries'][0]
	x_max  = letters[word[-1]]['boundaries'][0] + letters[word[-1]]['boundaries'][2]
	y_min  = min([letters[letter]['boundaries'][1] - letters[letter]['boundaries'][3] for letter in word])
	y_max  = max([letters[letter]['boundaries'][1] for letter in word])

	return [x_min, y_max, x_max - x_min, y_max - y_min]


def find_words(letters):
	letter_grps = []
	letter_grps_index = -1
	start_letter_dict = {}	# maps the starting letter of a letter_grp to its index in 'letter_grps'

	#Finds triplets or pairs of letter that are closest to eachother and staify letter connectivity conditions
	for index,letter in enumerate(letters):
		left_letter = (-1,inf)				# tuple is (index in 'letters' , distance from letter under consideration)
		right_letter = (-1,inf)

		[x , y , w , h] = letter['boundaries']
		base_line_diff = 0.2*h 				# acceptable difference in base line for letters that can be joined

		i = index + 1
		while (i< len(letters)) and (y - letters[i]['boundaries'][1] < base_line_diff):

			[connectable,direction,dist] = connect_letters(letter,letters[i])
			
			if not connectable:
				i+=1
				continue

			if direction == 1:
				if(dist<right_letter[1]):
					right_letter = (i,dist)
			else:
				if(dist<left_letter[1]):
					left_letter = (i,dist)
			i+=1

		letter_grp = [i for i in [left_letter[0], index, right_letter[0]] if i>=0]

		if(len(letter_grp)>1):
			letter_grps.append(letter_grp)
			letter_grps_index+=1
			start_letter_dict[letter_grp[0]] = letter_grps_index

	#Now concatenate letter groups to form words 
	used = [False]*len(letter_grps)
	words = []
	for index,letter_grp in enumerate(letter_grps):
		if used[index]:
			continue	

		# Connect the current group to one which starts with ending letter of this group
		common_letter = letter_grp[-1]
		while common_letter in start_letter_dict:
			next_grp_index = start_letter_dict[common_letter]
			start_letter_dict.pop(common_letter,None)
			letter_grps[index] = letter_grps[index][:-1] + letter_grps[next_grp_index]
			used[next_grp_index] = True
			common_letter = letter_grps[next_grp_index][-1] 	
		
		# Since groups are delted after appending , only the indices to final set of words are present in dict
		words = [letter_grps[start_letter_dict[i]] for i in start_letter_dict]	
	
	words_final = [ {"letters" : word , "boundaries" : get_word_boundary(word,letters)} for word in words]	
	
	return words_final

