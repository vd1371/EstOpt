import os
import pandas as pd


base_directory = 'E://OneDrive - The Hong Kong Polytechnic University//Academics//2018-21-Phd-HKPU//Outputs//10- EstOpt//Data'
                
# E://OneDrive - The Hong Kong Polytechnic University//Academics//2018-21-Phd-HKPolyU//Outputs//10- EstOpt//Data

def combine():

	ls = []
	for directory in ['Batch1//', 'Batch2//']:
		for file_name in os.listdir(base_directory + "//" + directory)[:]:

			try:
				print (f"{file_name} from {directory} is about to be loaded")
				direc = f"{base_directory}/{directory}/{file_name}"
				temp_df = pd.read_csv(direc, index_col = 0)
				ls.append(temp_df)
			except:
				print (f'{file_name} is problematic')

	df = pd.concat(ls)
	df.reset_index(inplace = True, drop = True)
	df.drop(columns = 'obj', inplace = True)
	print ("About to save...")
	df.to_csv(base_directory + "EstOptAll.csv")
	print ("EstOptAll Saved.\n-----------------")

def test2():

	df = pd.read_csv(base_directory + "EstOptAll.csv", index_col = 0)
	# df.set_index("index", inplace = True)
	# df.to_csv("EstOptAll.csv")
	print (df.head())
	print (df.describe())

if __name__ == "__main__":
	combine()
	test2()