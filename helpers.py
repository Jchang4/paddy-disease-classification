from fastai.text.all import *
from fastai.vision.all import *

from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification

from jupyter_helpers import device

# import kaggle

# base_dir = Path('/notebooks')
base_dir = Path("./")
path = base_dir / "paddy_data"
df = pd.read_csv(path / "train.csv")

# NOTE: all image sizes are (480, 640) or (640, 480)


def get_image_dls(bs: int) -> DataLoaders:
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        n_inp=1,
        item_tfms=Resize(480),
        batch_tfms=aug_transforms(min_scale=0.7, size=224),
        get_items=partial(get_image_files, folders=["train_images"]),
        get_y=parent_label,
        splitter=RandomSubsetSplitter(train_sz=0.8, valid_sz=0.2),
    ).dataloaders(path, bs=bs, shuffle=True)


def get_lm_dls(bs: int) -> DataLoaders:
    return DataBlock(
        blocks=TextBlock.from_df("variety", is_lm=True, seq_len=1),
        get_x=ColReader("text"),
        n_inp=1,
        splitter=RandomSubsetSplitter(train_sz=0.9, valid_sz=0.1),
    ).dataloaders(df, bs=bs)


def get_text_classifier_dls(bs: int, lm_dls=None) -> DataLoaders:
    # def text_get_items(df):
    #     return list(zip(df.variety.str.lower(), df.label))

    if not lm_dls:
        lm_dls = get_lm_dls(bs)

    return TextDataLoaders.from_df(
        df,
        valid_pct=0.2,
        text_col="variety",
        text_vocab=lm_dls.vocab,
        label_col="label",
    )


def get_image_and_text_items(df: pd.DataFrame) -> List[Tuple[Path, str, str]]:
    df["image_path"] = path / "train_images" / df.label / df.image_id
    return list(zip(df.image_path, df.variety.str.lower(), df.label))


def get_image_and_text_dls(bs: int, lm_dls=None, valid_pct: float = 0.2) -> DataLoaders:
    if not lm_dls:
        lm_dls = get_lm_dls(bs)

    return DataBlock(
        blocks=(
            ImageBlock,
            TextBlock.from_df(["variety"], vocab=lm_dls.vocab, seq_len=1),
            CategoryBlock,
        ),
        n_inp=2,
        get_items=get_image_and_text_items,
        get_x=(ItemGetter(0), ItemGetter(1)),
        get_y=ItemGetter(2),
        item_tfms=Resize(480),
        batch_tfms=aug_transforms(min_scale=0.7, size=224),
        splitter=RandomSubsetSplitter(train_sz=1 - valid_pct, valid_sz=valid_pct),
    ).dataloaders(df, bs=bs, shuffle=True)


def train_learner(dls: DataLoaders, model: nn.Module, splitter=noop, 
                  lr_multiplier: float = 0.75, 
                  lr_override: Optional[float] = None,
                  cbs: List[Callback] = []) -> Learner:
    learn = Learner(
        dls,
        model,
        metrics=[accuracy],
        cbs=cbs + [ShowGraphCallback()],
        splitter=splitter,
    ).to_fp16()
    learn.model = learn.model.to(device)
    
    if lr_override:
        lr = lr_override
    else:
        lr_min, lr_steep, lr_valley, lr_slide = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
        print('LR Minimum:', lr_min)
        print('LR Steep:', lr_steep)
        print('LR Valley:', lr_valley)
        print('LR Slide:', lr_slide)
        plt.show()
    
    lr = float(input('Type a learning_rate'))
    print(f"lr: {lr}")
    
    if not learn.splitter or learn.splitter is noop:
        learn.fit_one_cycle(4, lr)
        return learn

    num_splits = len(learn.splitter(learn.model))
    lr_divisors = [2, 2, 5]
    for i in reversed(range(num_splits)):
        num_epochs = 4 if i == 0 else 1
        magic_divisor = 2.6**4      
        learning_rates = slice(*[lr if j == 0 else lr/j/magic_divisor for j in reversed(range(num_splits))])
        
        learn.freeze_to(-i)
        learn.fit_one_cycle(num_epochs, learning_rates)
        
        lr /= lr_divisors[num_splits-1-i]

    return learn


def create_convnext(convnext_arch: str, num_classes: int = 10, with_relu: bool = False) -> nn.Module:
    model = ConvNextForImageClassification.from_pretrained(convnext_arch)
    linear = nn.Linear(model.config.hidden_sizes[-1], num_classes)
    if with_relu:
        linear = nn.Sequential(nn.ReLU(), linear)
    model.classifier = linear
    return model


def get_convnext_feature_extractor(convnext_arch: str) -> torch.Tensor:
    feature_extractor = ConvNextFeatureExtractor.from_pretrained(convnext_arch)
    
    def extract_features(x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            return torch.cat([
                feature_extractor(x_i.cpu(), return_tensors="pt")["pixel_values"]
                for x_i in x
            ], dim=0).to(device)
        return feature_extractor(x.cpu(), return_tensors="pt")["pixel_values"].to(device)
    
    return extract_features