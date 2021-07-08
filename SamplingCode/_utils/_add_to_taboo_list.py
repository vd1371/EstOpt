def _add_to_taboo_list(taboo_list, solut):
	taboo_list.append(hash(solut.tobytes()))