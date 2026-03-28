from torchvision.datasets import OxfordIIITPet

from tqdm.auto import tqdm

from torch.utils.data import Subset

def single_cat_OxfordIIITPet(category):
    dataset = OxfordIIITPet(
        root="data",
        split="trainval",
        target_types=("segmentation", "category"),
        download=True
    )

    test_dataset = OxfordIIITPet(
        root="data",
        split="test",
        target_types=("segmentation", "category"),
        download=True
    )

    idx_train = []
    for i, (image, (mask, cat)) in tqdm(enumerate(dataset), total=len(dataset)):
        if cat == category:
            idx_train.append(i)

    idx_test = []
    for i, (image, (mask, cat)) in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        if cat == category:
            idx_test.append(i)
            
    filtered_train_dataset = Subset(dataset, idx_train)
    filtered_test_dataset = Subset(test_dataset, idx_test)
    
    return filtered_train_dataset, filtered_test_dataset