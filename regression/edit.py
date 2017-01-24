if __name__ == "__main__":
	filepath = './data/minority_population.csv'	# './more-data/pop_density.csv'
	new_filename = './data/minority_population_abbv.csv'	# './more-data/pop_density_abbv.csv'
	col_index_of_state = 0

	with open(filepath) as file:
		with open(new_filename, 'w') as output:
			for line in file:
				pieces = line.split()
				piece = ';'.join(pieces)
				output.write(piece + '\n')