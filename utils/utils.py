PAD_SEQ = 'Z'
SA = ['A', 'Y', 'B', 'C', 'D', 'G', 'I', 'L', 'E', 'F', 'H', 'K', 'N', 'S', 'T', 'V', 'W', 'X', 'M', 'P', 'Q', 'R',
        PAD_SEQ]


def convert_to_sa(sa_idx):
    s = ''.join(SA[i] for i in sa_idx)
    s = s.replace(PAD_SEQ, '')
    return s
