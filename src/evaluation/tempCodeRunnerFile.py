        return {
            f'{prefix}mAP': adjusted_mAP,
            f'{prefix}rank1': adjusted_cmc_scores[0],
            f'{prefix}rank5': adjusted_cmc_scores[4],
            f'{prefix}rank10': adjusted_cmc_scores[9]
        }