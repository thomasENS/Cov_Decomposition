import os

def _check_folder(pth):
	'''
		Fct that checks if a path leads to an existing folder :
			- If not, creates recursively the path to the folder.
	'''

	try:
		if not os.path.isdir(pth):
			_check_folder(os.path.dirname(pth))
			os.mkdir(pth)
	except:
		pass

	return None
