def generate_opt_args_from_lists(param_lists):
	'''
	Converts nested lists of parameters containing tuples with (option,long-option) pairs
	into an option string and a long-option which serve as input to the getopt function.

	# Args:
		param_lists list of two lists containing option tuples. The first list contains all
		  options that require an argument. The second list contains all options that do not
		  require any argument.

	# Returns:
		arg_string: shortopts arg string meeting the getopt parameter requirements
		arg_list: longopts list meeting the getopt parameter requirements
	'''
	arg_string = ''
	arg_list = []
	for index,paramList in enumerate(param_lists):
		if isinstance(paramList,list):
			for k,v in paramList:
				if index == 0:
					# argumnet required
					arg_string += "{}:".format(k)
					arg_list.append("{}=".format(v))
				else:
					# no argumnet required
					arg_string += k
					arg_list.append(v)
		else:
			raise Exception
	return arg_string,arg_list

def genereate_opt_args_from_nested_lists(param_dicts):
	'''
	Converts a dictionary with nested lists or dictionaries of parameters
	into shortopts,longopts arguments.

	# Args:
		param_dicts dictionary containing nested dictionaries or lists with option tuples.

	# Returns:
		arg_string: shortopts arg string meeting the getopt parameter requirements
		arg_list: longopts list meeting the getopt parameter requirements
	'''
	arg_string = ''
	arg_list = []
	for method,param_lists in param_dicts.items():
		if isinstance(param_lists,list):
			args_string_tmp,arg_list_tmp = generate_opt_args_from_lists(param_lists)
			arg_string += args_string_tmp
			arg_list += arg_list_tmp
		elif isinstance(param_lists,dict):
			for method,tup in param_lists.items():
				arg_string += tup[0]
				arg_list.append(tup[1])
		else:
			raise Exception
	return arg_string,arg_list

def _generate_flag_tuple(flags):
	'''
	Converts a tuple of arbitrary values to a tuple of (option,long-option)
	'''
	return ('-{}'.format(flags[0]),'--{}'.format(flags[1]))

def extract_option(flags,opts):
	'''
	Checks if option flag is present in opts list and extracts the corresponding
	value from the opts list.

	# Args:
		flags tuple with 2 coordinates. 1st coordinate is considered to
		  represent an option, 2nd coordinate is considered to represent
		  a long option
		opts list of tuples containing ((long)option,value) pairs
		
	# Returns
		string value as of an opts tuple as if (long)option is found, else None
	'''
	tup = _generate_flag_tuple(flags)
	for opt in opts:
		if opt[0] in tup:
			return opt[1]
	return None

def check_flag_presence(flags,opts):
	'''
	Checks if option flag is present in opts list.

	# Args:
		flags tuple with 2 coordinates. 1st coordinate is considered to
		  represent an option, 2nd coordinate is considered to represent
		  a long option
		opts list of tuples containing ((long)option,value) pairs

	# Returns
		True if (long)option is found
		False else
	'''
	tup = _generate_flag_tuple(flags)
	for opt in opts:
		if opt[0] in tup:
			return True
	return False