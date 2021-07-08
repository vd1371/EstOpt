from collections import Counter
from ._mrr_to_decimal import mrr_to_decimal

RECON = 3
REHAB = 2
MAINT = 1
DONOT = 0

def check_policy_binary(mrr):

	mrr_decimal = mrr_to_decimal(mrr)

	for elem_mrr in mrr_decimal:
		counts = Counter(elem_mrr)

		if counts[RECON] > 2:
			return False
		elif counts[REHAB] > 3:
			return False
		elif counts[MAINT] > 5:
			return False

	for i in range (len(mrr_decimal)):
		for val1, val2 in zip(mrr_decimal[i][:-1], mrr_decimal[i][1:]):
			if val1 * val2 > 0:
				return False

	return True