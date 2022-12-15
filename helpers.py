from fastai.text.all import *
from fastai.vision.all import *

from jupyter_helpers import device

# import kaggle

# base_dir = Path('/notebooks')
base_dir = Path("./")
path = base_dir / "paddy_data"
df = pd.read_csv(path / "train.csv")

# NOTE: all image sizes are (480, 640) or (640, 480)


def get_image_dls(bs: int):
    return DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        n_inp=1,
        item_tfms=Resize(480),
        batch_tfms=aug_transforms(min_scale=0.7, size=224),
        get_items=partial(get_image_files, folders=["train_images"]),
        get_y=parent_label,
        splitter=RandomSubsetSplitter(train_sz=0.8, valid_sz=0.2),
    ).dataloaders(path, bs=bs, shuffle=True)


def get_lm_dls(bs: int):
    return DataBlock(
        blocks=TextBlock.from_df("variety", is_lm=True, seq_len=1),
        get_x=ColReader("text"),
        n_inp=1,
        splitter=RandomSubsetSplitter(train_sz=0.9, valid_sz=0.1),
    ).dataloaders(df, bs=bs)


def get_text_classifier_dls(bs: int, lm_dls=None):
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


def get_image_and_text_items(df):
    df["image_path"] = path / "train_images" / df.label / df.image_id
    return list(zip(df.image_path, df.variety.str.lower(), df.label))


def get_image_and_text_dls(bs: int, lm_dls=None, valid_pct: float = 0.2):
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


def train_learner(dls, model, splitter=noop, lr_multiplier: float = 0.75, cbs=[]):
    learn = Learner(
        dls,
        model,
        metrics=[accuracy],
        cbs=cbs + [ShowGraphCallback()],
        splitter=splitter,
    ).to_fp16()
    learn = learn.model.to(device)

    lr_steep, lr_valley = learn.lr_find(suggest_funcs=(steep, valley))
    lr = lr_steep * lr_multiplier
    print(f"Steep: {lr_steep}; Valley: {lr_valley}; lr: {lr}")

    if splitter:
        learn.freeze_to(-1)
        learn.fit_one_cycle(1, lr)

        learn.unfreeze()
        learn.fit_one_cycle(3, lr / 2)
    else:
        learn.fit_one_cycle(4, lr)

    return learn
