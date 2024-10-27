def train_collate_fn(batch):
    features = {
        "image": batch[:]["features"]["image"],
        "text": batch[:]["features"]["text"],
    }
    targets = batch[:]["target"]

    return {"features": features, "target": targets}


def infer_collate_fn(batch):
    features = {
        "image": batch[:]["features"]["image"],
        "text": batch[:]["features"]["text"],
    }

    return {"features": features}
