import torch


def argmax(vec):
    """
    Return the argmax as an int
    :param vec: tensor
    :return: int
    """
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_idx, device):
    idxs = [to_idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long).to(device)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))