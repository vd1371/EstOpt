import numpy as np
from deap import tools

def mate(chrom1, chrom2, **params):

	crossver_prob = params.pop("crossver_prob")
	mutation_prob = params.pop("mutation_prob")

	# Two point crossover
	if np.random.random() < crossver_prob:
		chrom1, chrom2 = tools.cxTwoPoint(chrom1, chrom2)

	# Flipbit mutation
	chrom1 = tools.mutFlipBit(chrom1, mutation_prob)
	chrom2 = tools.mutFlipBit(chrom2, mutation_prob)

	return np.array(chrom1[0]), np.array(chrom2[0])