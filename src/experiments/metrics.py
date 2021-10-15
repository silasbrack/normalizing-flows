from src.experiments._psis import psislw


def evaluate_k_hat(log_p, log_q, to_print=True):
    log_w = log_p - log_q

    lw_out, kss = psislw(log_w.cpu().detach().numpy())

    if to_print:
        if kss < 0.7:
            print(f"k hat = {kss:.3f} < 0.7: VI approximation is reliable.")
        else:
            print(f"k hat = {kss:.3f} > 0.7: VI approximation is NOT reliable.")
    return kss
