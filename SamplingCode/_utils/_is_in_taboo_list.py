def _is_in_taboo_list(taboo_list, solut):
	return hash(solut.tobytes()) in taboo_list